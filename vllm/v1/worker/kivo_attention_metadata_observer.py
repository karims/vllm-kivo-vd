# SPDX-License-Identifier: Apache-2.0

"""Source-level observer for attention metadata visibility and shadow planning.

This helper records backend-agnostic metadata at the attention metadata
boundary without mutating runtime behavior.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID as _PAD_SLOT_ID
except Exception:  # pragma: no cover - defensive fallback
    _PAD_SLOT_ID = -1

_OBSERVE_SCHEMA_VERSION = "kivo_source_s3_0b_attention_metadata_observer_v1"
_OBSERVE_POLICY_NAME = "observe_attention_metadata"
_SHADOW_SCHEMA_VERSION = (
    "kivo_source_s3_1a_shadow_selected_attention_metadata_v1"
)
_SHADOW_POLICY_NAME = "shadow_selected_attention_metadata"
_SHADOW_SELECTION_POLICY_NAME = "deterministic_placeholder_block_score"
_SHADOW_SKETCH_SCHEMA_VERSION = (
    "kivo_source_s3_1b_shadow_sketch_selected_attention_metadata_v1"
)
_SHADOW_SKETCH_POLICY_NAME = "shadow_sketch_selected_attention_metadata"
_SHADOW_SKETCH_SELECTION_POLICY_NAME = "slot_coverage_recency_proxy"
_DEFAULT_BUDGET_RATIO = 0.5
_DEFAULT_KEEP_RECENT_BLOCKS = 1
_DEFAULT_COVERAGE_WEIGHT = 0.6
_DEFAULT_RECENCY_WEIGHT = 0.4
_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_ID = 0


def _next_sequence_id() -> int:
    global _SEQUENCE_ID
    with _SEQUENCE_LOCK:
        _SEQUENCE_ID += 1
        return _SEQUENCE_ID


def _env_flag(name: str) -> bool:
    return os.getenv(name) == "1"


def _observation_path() -> Path | None:
    path = os.getenv("KIVO_SOURCE_OBSERVE_PATH") or os.getenv(
        "KIVO_SOURCE_OBS_PATH"
    )
    if not path:
        return None
    return Path(path)


def _current_policy_name() -> str | None:
    return os.getenv("KIVO_SOURCE_POLICY")


def _env_debug() -> dict[str, Any]:
    observe_path = _observation_path()
    return {
        "kivo_source_enable_seen": _env_flag("KIVO_SOURCE_ENABLE"),
        "kivo_source_policy_seen": _current_policy_name(),
        "observe_path_present": observe_path is not None,
    }


def _safe_tensor_info(value: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "present": value is not None,
        "shape": None,
        "dtype": None,
        "device": None,
    }
    if value is None:
        return info
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            info["shape"] = [int(dim) for dim in shape]
        except Exception:
            info["shape"] = str(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        info["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        info["device"] = str(device)
    return info


def _tensor_numel(value: Any) -> int | None:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:
            return None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)
    except Exception:
        return None


def _is_int_dtype(dtype: Any) -> bool:
    dtype_text = str(dtype).lower()
    return "int" in dtype_text and "bool" not in dtype_text


def _bounded_sample(value: Any, limit: int = 16) -> list[Any]:
    if value is None or limit <= 0:
        return []
    try:
        sample = value
        if hasattr(sample, "detach"):
            sample = sample.detach()
        if hasattr(sample, "reshape"):
            sample = sample.reshape(-1)
        if hasattr(sample, "to"):
            try:
                sample = sample.to("cpu")
            except Exception:
                pass
        if hasattr(sample, "tolist"):
            sample_list = sample[:limit].tolist()
            return sample_list if isinstance(sample_list, list) else [sample_list]
        if isinstance(sample, (list, tuple)):
            return list(sample[:limit])
    except Exception:  # pragma: no cover - defensive path
        return []
    return []


def _safe_exact_int_values(
    value: Any,
    *,
    max_numel: int = 256,
) -> tuple[list[int] | None, str | None]:
    if value is None:
        return None, "tensor unavailable"
    dtype = getattr(value, "dtype", None)
    if dtype is not None and not _is_int_dtype(dtype):
        return None, "tensor is not integer dtype"
    numel = _tensor_numel(value)
    if numel is None:
        return None, "tensor size unavailable"
    if numel > max_numel:
        return None, "tensor too large for safe exact scan"
    values = _bounded_sample(value, limit=numel)
    if len(values) != numel:
        return None, "tensor sample unavailable"
    exact_values: list[int] = []
    for item in values:
        try:
            exact_values.append(int(item))
        except Exception:
            return None, "tensor value conversion failed"
    return exact_values, None


def _visible_block_count_estimate(
    slot_mapping: Any,
    *,
    block_size: int,
    pad_slot_id: int,
) -> tuple[int | None, str | None]:
    if slot_mapping is None:
        return None, "slot_mapping unavailable"
    if block_size <= 0:
        return None, "invalid block_size"
    values, value_caveat = _safe_exact_int_values(slot_mapping)
    if values is None:
        return None, value_caveat
    visible_blocks: list[int] = []
    seen: set[int] = set()
    for slot_id in values:
        if slot_id == pad_slot_id or slot_id < 0:
            continue
        block_id = slot_id // block_size
        if block_id not in seen:
            seen.add(block_id)
            visible_blocks.append(block_id)
    return len(visible_blocks), None


def _extract_visible_block_ids(
    block_table_tensor: Any,
) -> tuple[list[int] | None, str | None]:
    values, value_caveat = _safe_exact_int_values(block_table_tensor)
    if values is None:
        return None, value_caveat or "block_table unavailable"
    visible_block_ids: list[int] = []
    seen: set[int] = set()
    for raw_block_id in values:
        if raw_block_id < 0:
            continue
        if raw_block_id not in seen:
            seen.add(raw_block_id)
            visible_block_ids.append(raw_block_id)
    return visible_block_ids, None


def _parse_env_float(
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _parse_env_int(
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _placeholder_block_score(block_id: int) -> int:
    return ((int(block_id) + 1) * 2654435761) & 0xFFFFFFFF


def _slot_coverage_by_block(
    slot_mapping: Any,
    *,
    block_size: int,
    pad_slot_id: int,
) -> tuple[dict[int, int] | None, str | None]:
    if slot_mapping is None:
        return None, "slot_mapping unavailable"
    if block_size <= 0:
        return None, "invalid block_size"
    values, value_caveat = _safe_exact_int_values(slot_mapping)
    if values is None:
        return None, value_caveat
    coverage: dict[int, int] = {}
    for slot_id in values:
        if slot_id == pad_slot_id or slot_id < 0:
            continue
        block_id = slot_id // block_size
        coverage[block_id] = coverage.get(block_id, 0) + 1
    return coverage, None


def _planned_selected_count(
    visible_count: int,
    *,
    budget_ratio: float,
    keep_recent_blocks: int,
) -> int:
    if visible_count <= 0:
        return 0
    target_budget = max(1, int(math.ceil(visible_count * budget_ratio)))
    return min(visible_count, max(min(visible_count, keep_recent_blocks), target_budget))


def _plan_shadow_selection(
    *,
    visible_block_ids: list[int] | None,
    visible_block_count_estimate: int | None,
    budget_ratio: float,
    keep_recent_blocks: int,
) -> tuple[dict[str, Any], list[str]]:
    caveats: list[str] = []
    visible_count = (
        len(visible_block_ids)
        if visible_block_ids is not None
        else int(visible_block_count_estimate or 0)
    )
    if visible_count <= 0:
        return {
            "selected_block_count": 0,
            "selected_block_ids_sample": [],
            "excluded_block_count": 0,
            "excluded_block_ids_sample": [],
            "theoretical_attention_visible_block_reduction": 0,
            "theoretical_attention_visible_block_reduction_ratio": 0.0,
        }, caveats

    keep_recent = min(max(0, keep_recent_blocks), visible_count)
    selected_count = _planned_selected_count(
        visible_count,
        budget_ratio=budget_ratio,
        keep_recent_blocks=keep_recent,
    )
    excluded_count = max(0, visible_count - selected_count)
    reduction_ratio = (
        float(excluded_count) / float(visible_count) if visible_count > 0 else 0.0
    )

    if visible_block_ids is None:
        caveats.append("visible block ids unavailable; counts are estimate-only")
        return {
            "selected_block_count": selected_count,
            "selected_block_ids_sample": [],
            "excluded_block_count": excluded_count,
            "excluded_block_ids_sample": [],
            "theoretical_attention_visible_block_reduction": excluded_count,
            "theoretical_attention_visible_block_reduction_ratio": reduction_ratio,
        }, caveats

    recent_ids = visible_block_ids[-keep_recent:] if keep_recent > 0 else []
    recent_set = set(recent_ids)
    older_ids = [
        block_id for block_id in visible_block_ids if block_id not in recent_set
    ]
    older_budget = max(0, selected_count - len(recent_ids))
    ranked_older_ids = sorted(
        older_ids,
        key=lambda block_id: (-_placeholder_block_score(block_id), block_id),
    )
    selected_older_set = set(ranked_older_ids[:older_budget])
    selected_set = recent_set | selected_older_set
    if not selected_set and visible_block_ids:
        selected_set.add(visible_block_ids[-1])
    selected_ids = [
        block_id for block_id in visible_block_ids if block_id in selected_set
    ]
    excluded_ids = [
        block_id for block_id in visible_block_ids if block_id not in selected_set
    ]
    return {
        "selected_block_count": len(selected_ids),
        "selected_block_ids_sample": selected_ids[:16],
        "excluded_block_count": len(excluded_ids),
        "excluded_block_ids_sample": excluded_ids[:16],
        "theoretical_attention_visible_block_reduction": len(excluded_ids),
        "theoretical_attention_visible_block_reduction_ratio": (
            float(len(excluded_ids)) / float(len(visible_block_ids))
            if visible_block_ids
            else 0.0
        ),
    }, caveats


def _plan_shadow_sketch_selection(
    *,
    visible_block_ids: list[int] | None,
    visible_block_count_estimate: int | None,
    coverage_by_block: dict[int, int] | None,
    budget_ratio: float,
    keep_recent_blocks: int,
    coverage_weight: float,
    recency_weight: float,
) -> tuple[dict[str, Any], list[str]]:
    caveats: list[str] = []
    visible_count = (
        len(visible_block_ids)
        if visible_block_ids is not None
        else int(visible_block_count_estimate or 0)
    )
    if visible_count <= 0:
        return {
            "selected_block_count": 0,
            "selected_block_ids_sample": [],
            "excluded_block_count": 0,
            "excluded_block_ids_sample": [],
            "theoretical_attention_visible_block_reduction": 0,
            "theoretical_attention_visible_block_reduction_ratio": 0.0,
            "block_score_sample": [],
            "fallback_used": False,
            "fallback_reason": None,
        }, caveats

    keep_recent = min(max(0, keep_recent_blocks), visible_count)
    selected_count = _planned_selected_count(
        visible_count,
        budget_ratio=budget_ratio,
        keep_recent_blocks=keep_recent,
    )
    if visible_block_ids is None:
        caveats.append("visible block ids unavailable; counts are estimate-only")
        excluded_count = max(0, visible_count - selected_count)
        return {
            "selected_block_count": selected_count,
            "selected_block_ids_sample": [],
            "excluded_block_count": excluded_count,
            "excluded_block_ids_sample": [],
            "theoretical_attention_visible_block_reduction": excluded_count,
            "theoretical_attention_visible_block_reduction_ratio": (
                float(excluded_count) / float(visible_count)
                if visible_count > 0
                else 0.0
            ),
            "block_score_sample": [],
            "fallback_used": True,
            "fallback_reason": "visible_block_ids_unavailable",
        }, caveats

    recent_ids = visible_block_ids[-keep_recent:] if keep_recent > 0 else []
    recent_set = set(recent_ids)
    older_ids = [
        block_id for block_id in visible_block_ids if block_id not in recent_set
    ]
    older_budget = max(0, selected_count - len(recent_ids))

    fallback_used = False
    fallback_reason = None
    normalized_coverage: dict[int, float] = {}
    if coverage_by_block is None:
        fallback_used = True
        fallback_reason = "coverage_unavailable_fell_back_to_recency"
        caveats.append(fallback_reason)
    else:
        max_coverage = max((coverage_by_block.get(block_id, 0) for block_id in visible_block_ids), default=0)
        if max_coverage <= 0:
            fallback_used = True
            fallback_reason = "coverage_unavailable_fell_back_to_recency"
            caveats.append(fallback_reason)
        else:
            normalized_coverage = {
                block_id: float(coverage_by_block.get(block_id, 0)) / float(max_coverage)
                for block_id in visible_block_ids
            }

    denom = max(1, len(visible_block_ids) - 1)
    recency_scores = {
        block_id: float(index) / float(denom)
        for index, block_id in enumerate(visible_block_ids)
    }
    score_by_block: dict[int, float] = {}
    block_score_sample: list[dict[str, Any]] = []
    for block_id in visible_block_ids:
        coverage_count = (
            int(coverage_by_block.get(block_id, 0))
            if coverage_by_block is not None
            else None
        )
        recency_rank = len(visible_block_ids) - 1 - visible_block_ids.index(block_id)
        recency_score = recency_scores[block_id]
        coverage_score = normalized_coverage.get(block_id, 0.0)
        score = (
            recency_score
            if fallback_used
            else coverage_weight * coverage_score + recency_weight * recency_score
        )
        score_by_block[block_id] = score
        if len(block_score_sample) < 16:
            block_score_sample.append(
                {
                    "block_id": int(block_id),
                    "coverage_count": coverage_count,
                    "recency_rank": int(recency_rank),
                    "recency_score": float(recency_score),
                    "score": float(score),
                }
            )

    ranked_older_ids = sorted(
        older_ids,
        key=lambda block_id: (-score_by_block.get(block_id, 0.0), block_id),
    )
    selected_older_set = set(ranked_older_ids[:older_budget])
    selected_set = recent_set | selected_older_set
    if not selected_set and visible_block_ids:
        selected_set.add(visible_block_ids[-1])
    selected_ids = [
        block_id for block_id in visible_block_ids if block_id in selected_set
    ]
    excluded_ids = [
        block_id for block_id in visible_block_ids if block_id not in selected_set
    ]
    return {
        "selected_block_count": len(selected_ids),
        "selected_block_ids_sample": selected_ids[:16],
        "excluded_block_count": len(excluded_ids),
        "excluded_block_ids_sample": excluded_ids[:16],
        "theoretical_attention_visible_block_reduction": len(excluded_ids),
        "theoretical_attention_visible_block_reduction_ratio": (
            float(len(excluded_ids)) / float(len(visible_block_ids))
            if visible_block_ids
            else 0.0
        ),
        "block_score_sample": block_score_sample,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }, caveats


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        output_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644
    )
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _common_metadata_record(
    *,
    schema_version: str,
    policy_name: str,
    hook_point: str,
    kv_cache_group_id: int,
    block_size: int,
    pad_slot_id: int,
    env_debug: dict[str, Any],
    block_table_info: dict[str, Any],
    slot_mapping_info: dict[str, Any],
    query_start_loc_info: dict[str, Any],
    seq_lens_info: dict[str, Any],
    positions_info: dict[str, Any],
    max_query_len: int | None,
    max_seq_len: int | None,
    visible_block_count_estimate: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "timestamp": time.time(),
        "sequence_id": _next_sequence_id(),
        "pid": os.getpid(),
        "policy_name": policy_name,
        "hook_point": hook_point,
        "kv_cache_group_id": kv_cache_group_id,
        "block_size": block_size,
        "pad_slot_id": pad_slot_id,
        "kivo_source_enable_seen": env_debug["kivo_source_enable_seen"],
        "kivo_source_policy_seen": env_debug["kivo_source_policy_seen"],
        "observe_path_present": env_debug["observe_path_present"],
        "block_table_tensor_present": block_table_info["present"],
        "block_table_tensor_shape": block_table_info["shape"],
        "block_table_tensor_dtype": block_table_info["dtype"],
        "block_table_tensor_device": block_table_info["device"],
        "slot_mapping_present": slot_mapping_info["present"],
        "slot_mapping_shape": slot_mapping_info["shape"],
        "slot_mapping_dtype": slot_mapping_info["dtype"],
        "slot_mapping_device": slot_mapping_info["device"],
        "query_start_loc_shape": query_start_loc_info["shape"],
        "query_start_loc_dtype": query_start_loc_info["dtype"],
        "query_start_loc_device": query_start_loc_info["device"],
        "seq_lens_shape": seq_lens_info["shape"],
        "seq_lens_dtype": seq_lens_info["dtype"],
        "seq_lens_device": seq_lens_info["device"],
        "positions_shape": positions_info["shape"],
        "positions_dtype": positions_info["dtype"],
        "positions_device": positions_info["device"],
        "max_query_len": int(max_query_len) if max_query_len is not None else None,
        "max_seq_len": int(max_seq_len) if max_seq_len is not None else None,
        "visible_block_count_estimate": visible_block_count_estimate,
    }


def _build_observe_record(
    *,
    hook_point: str,
    kv_cache_group_id: int,
    common_attn_metadata: Any,
    kv_cache_spec: Any,
    env_debug: dict[str, Any],
) -> dict[str, Any]:
    block_table_tensor = getattr(common_attn_metadata, "block_table_tensor", None)
    slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
    query_start_loc = getattr(common_attn_metadata, "query_start_loc", None)
    seq_lens = getattr(common_attn_metadata, "seq_lens", None)
    positions = getattr(common_attn_metadata, "positions", None)
    max_query_len = getattr(common_attn_metadata, "max_query_len", None)
    max_seq_len = getattr(common_attn_metadata, "max_seq_len", None)
    block_size = int(
        getattr(kv_cache_spec, "storage_block_size", None)
        or getattr(kv_cache_spec, "block_size", 0)
        or 0
    )
    pad_slot_id = int(_PAD_SLOT_ID) if isinstance(_PAD_SLOT_ID, int) else -1
    visible_block_count_estimate, estimate_caveat = _visible_block_count_estimate(
        slot_mapping,
        block_size=block_size,
        pad_slot_id=pad_slot_id,
    )
    block_table_info = _safe_tensor_info(block_table_tensor)
    slot_mapping_info = _safe_tensor_info(slot_mapping)
    query_start_loc_info = _safe_tensor_info(query_start_loc)
    seq_lens_info = _safe_tensor_info(seq_lens)
    positions_info = _safe_tensor_info(positions)
    record = _common_metadata_record(
        schema_version=_OBSERVE_SCHEMA_VERSION,
        policy_name=_OBSERVE_POLICY_NAME,
        hook_point=hook_point,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
        pad_slot_id=pad_slot_id,
        env_debug=env_debug,
        block_table_info=block_table_info,
        slot_mapping_info=slot_mapping_info,
        query_start_loc_info=query_start_loc_info,
        seq_lens_info=seq_lens_info,
        positions_info=positions_info,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        visible_block_count_estimate=visible_block_count_estimate,
    )
    record.update({
        "visible_block_count_estimate_caveat": estimate_caveat,
        "block_table_tensor_sample": _bounded_sample(block_table_tensor, limit=16),
        "slot_mapping_sample": _bounded_sample(slot_mapping, limit=16),
        "selected_block_count": None,
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "caveats": [
            "observation-only hook; no runtime state is mutated",
            "metadata visibility is recorded at the backend-agnostic boundary",
            "visible block count is estimated only when safe and bounded",
            "no selected attention, KV reduction, or latency claim is made",
        ],
    })
    return record


def _build_shadow_plan_record(
    *,
    hook_point: str,
    kv_cache_group_id: int,
    common_attn_metadata: Any,
    kv_cache_spec: Any,
    env_debug: dict[str, Any],
) -> dict[str, Any]:
    block_table_tensor = getattr(common_attn_metadata, "block_table_tensor", None)
    slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
    query_start_loc = getattr(common_attn_metadata, "query_start_loc", None)
    seq_lens = getattr(common_attn_metadata, "seq_lens", None)
    positions = getattr(common_attn_metadata, "positions", None)
    max_query_len = getattr(common_attn_metadata, "max_query_len", None)
    max_seq_len = getattr(common_attn_metadata, "max_seq_len", None)
    block_size = int(
        getattr(kv_cache_spec, "storage_block_size", None)
        or getattr(kv_cache_spec, "block_size", 0)
        or 0
    )
    pad_slot_id = int(_PAD_SLOT_ID) if isinstance(_PAD_SLOT_ID, int) else -1
    budget_ratio = _parse_env_float(
        "KIVO_SOURCE_BUDGET_RATIO",
        default=_DEFAULT_BUDGET_RATIO,
        minimum=0.0,
        maximum=1.0,
    )
    keep_recent_blocks = _parse_env_int(
        "KIVO_SOURCE_KEEP_RECENT_BLOCKS",
        default=_DEFAULT_KEEP_RECENT_BLOCKS,
        minimum=0,
    )
    visible_block_ids, visible_block_ids_caveat = _extract_visible_block_ids(
        block_table_tensor
    )
    visible_block_count_estimate = (
        len(visible_block_ids) if visible_block_ids is not None else None
    )
    estimate_caveat = None
    if visible_block_count_estimate is None:
        visible_block_count_estimate, estimate_caveat = _visible_block_count_estimate(
            slot_mapping,
            block_size=block_size,
            pad_slot_id=pad_slot_id,
        )
    block_table_info = _safe_tensor_info(block_table_tensor)
    slot_mapping_info = _safe_tensor_info(slot_mapping)
    query_start_loc_info = _safe_tensor_info(query_start_loc)
    seq_lens_info = _safe_tensor_info(seq_lens)
    positions_info = _safe_tensor_info(positions)
    selection_plan, plan_caveats = _plan_shadow_selection(
        visible_block_ids=visible_block_ids,
        visible_block_count_estimate=visible_block_count_estimate,
        budget_ratio=budget_ratio,
        keep_recent_blocks=keep_recent_blocks,
    )
    caveats = [
        "shadow planning only; no runtime state is mutated",
        "deterministic placeholder scoring is not the final sketch selector",
        "no selected attention, KV reduction, or latency claim is made",
    ]
    if visible_block_ids_caveat is not None:
        caveats.append(f"visible block ids caveat: {visible_block_ids_caveat}")
    if estimate_caveat is not None:
        caveats.append(f"visible block estimate caveat: {estimate_caveat}")
    caveats.extend(plan_caveats)
    record = _common_metadata_record(
        schema_version=_SHADOW_SCHEMA_VERSION,
        policy_name=_SHADOW_POLICY_NAME,
        hook_point=hook_point,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
        pad_slot_id=pad_slot_id,
        env_debug=env_debug,
        block_table_info=block_table_info,
        slot_mapping_info=slot_mapping_info,
        query_start_loc_info=query_start_loc_info,
        seq_lens_info=seq_lens_info,
        positions_info=positions_info,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        visible_block_count_estimate=visible_block_count_estimate,
    )
    record.update({
        "visible_block_ids_sample": (visible_block_ids or [])[:16],
        "selection_policy_name": _SHADOW_SELECTION_POLICY_NAME,
        "budget_ratio": budget_ratio,
        "keep_recent_blocks": keep_recent_blocks,
        **selection_plan,
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "caveats": caveats,
    })
    return record


def _build_shadow_sketch_plan_record(
    *,
    hook_point: str,
    kv_cache_group_id: int,
    common_attn_metadata: Any,
    kv_cache_spec: Any,
    env_debug: dict[str, Any],
) -> dict[str, Any]:
    block_table_tensor = getattr(common_attn_metadata, "block_table_tensor", None)
    slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
    query_start_loc = getattr(common_attn_metadata, "query_start_loc", None)
    seq_lens = getattr(common_attn_metadata, "seq_lens", None)
    positions = getattr(common_attn_metadata, "positions", None)
    max_query_len = getattr(common_attn_metadata, "max_query_len", None)
    max_seq_len = getattr(common_attn_metadata, "max_seq_len", None)
    block_size = int(
        getattr(kv_cache_spec, "storage_block_size", None)
        or getattr(kv_cache_spec, "block_size", 0)
        or 0
    )
    pad_slot_id = int(_PAD_SLOT_ID) if isinstance(_PAD_SLOT_ID, int) else -1
    budget_ratio = _parse_env_float(
        "KIVO_SOURCE_BUDGET_RATIO",
        default=_DEFAULT_BUDGET_RATIO,
        minimum=0.0,
        maximum=1.0,
    )
    keep_recent_blocks = _parse_env_int(
        "KIVO_SOURCE_KEEP_RECENT_BLOCKS",
        default=_DEFAULT_KEEP_RECENT_BLOCKS,
        minimum=0,
    )
    coverage_weight = _parse_env_float(
        "KIVO_SOURCE_COVERAGE_WEIGHT",
        default=_DEFAULT_COVERAGE_WEIGHT,
        minimum=0.0,
        maximum=1.0,
    )
    recency_weight = _parse_env_float(
        "KIVO_SOURCE_RECENCY_WEIGHT",
        default=_DEFAULT_RECENCY_WEIGHT,
        minimum=0.0,
        maximum=1.0,
    )
    visible_block_ids, visible_block_ids_caveat = _extract_visible_block_ids(
        block_table_tensor
    )
    visible_block_count_estimate = (
        len(visible_block_ids) if visible_block_ids is not None else None
    )
    estimate_caveat = None
    if visible_block_count_estimate is None:
        visible_block_count_estimate, estimate_caveat = _visible_block_count_estimate(
            slot_mapping,
            block_size=block_size,
            pad_slot_id=pad_slot_id,
        )
    coverage_by_block, coverage_caveat = _slot_coverage_by_block(
        slot_mapping,
        block_size=block_size,
        pad_slot_id=pad_slot_id,
    )
    block_table_info = _safe_tensor_info(block_table_tensor)
    slot_mapping_info = _safe_tensor_info(slot_mapping)
    query_start_loc_info = _safe_tensor_info(query_start_loc)
    seq_lens_info = _safe_tensor_info(seq_lens)
    positions_info = _safe_tensor_info(positions)
    selection_plan, plan_caveats = _plan_shadow_sketch_selection(
        visible_block_ids=visible_block_ids,
        visible_block_count_estimate=visible_block_count_estimate,
        coverage_by_block=coverage_by_block,
        budget_ratio=budget_ratio,
        keep_recent_blocks=keep_recent_blocks,
        coverage_weight=coverage_weight,
        recency_weight=recency_weight,
    )
    caveats = [
        "shadow planning only; no runtime state is mutated",
        "proxy scoring uses metadata-derived coverage and recency only",
        "no selected attention, KV reduction, or latency claim is made",
    ]
    if visible_block_ids_caveat is not None:
        caveats.append(f"visible block ids caveat: {visible_block_ids_caveat}")
    if estimate_caveat is not None:
        caveats.append(f"visible block estimate caveat: {estimate_caveat}")
    if coverage_caveat is not None:
        caveats.append(f"coverage caveat: {coverage_caveat}")
    caveats.extend(plan_caveats)
    record = _common_metadata_record(
        schema_version=_SHADOW_SKETCH_SCHEMA_VERSION,
        policy_name=_SHADOW_SKETCH_POLICY_NAME,
        hook_point=hook_point,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
        pad_slot_id=pad_slot_id,
        env_debug=env_debug,
        block_table_info=block_table_info,
        slot_mapping_info=slot_mapping_info,
        query_start_loc_info=query_start_loc_info,
        seq_lens_info=seq_lens_info,
        positions_info=positions_info,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        visible_block_count_estimate=visible_block_count_estimate,
    )
    record.update({
        "visible_block_ids_sample": (visible_block_ids or [])[:16],
        "selection_policy_name": _SHADOW_SKETCH_SELECTION_POLICY_NAME,
        "budget_ratio": budget_ratio,
        "keep_recent_blocks": keep_recent_blocks,
        "coverage_weight": coverage_weight,
        "recency_weight": recency_weight,
        **selection_plan,
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "caveats": caveats,
    })
    return record


def maybe_observe_attention_metadata(
    *,
    hook_point: str,
    kv_cache_group_id: int,
    common_attn_metadata: Any,
    kv_cache_spec: Any,
) -> dict[str, Any] | None:
    """Record a metadata-only observation if the source hook is enabled."""
    try:
        env_debug = _env_debug()
        if not env_debug["kivo_source_enable_seen"]:
            return None
        output_path = _observation_path()
        if output_path is None:
            return None

        policy_name = env_debug["kivo_source_policy_seen"]
        if policy_name == _OBSERVE_POLICY_NAME:
            record = _build_observe_record(
                hook_point=hook_point,
                kv_cache_group_id=kv_cache_group_id,
                common_attn_metadata=common_attn_metadata,
                kv_cache_spec=kv_cache_spec,
                env_debug=env_debug,
            )
        elif policy_name == _SHADOW_POLICY_NAME:
            if hook_point != "_build_attention_metadata":
                return None
            record = _build_shadow_plan_record(
                hook_point=hook_point,
                kv_cache_group_id=kv_cache_group_id,
                common_attn_metadata=common_attn_metadata,
                kv_cache_spec=kv_cache_spec,
                env_debug=env_debug,
            )
        elif policy_name == _SHADOW_SKETCH_POLICY_NAME:
            if hook_point != "_build_attention_metadata":
                return None
            record = _build_shadow_sketch_plan_record(
                hook_point=hook_point,
                kv_cache_group_id=kv_cache_group_id,
                common_attn_metadata=common_attn_metadata,
                kv_cache_spec=kv_cache_spec,
                env_debug=env_debug,
            )
        else:
            return None

        _append_jsonl(output_path, record)
        return record
    except Exception:
        return None
