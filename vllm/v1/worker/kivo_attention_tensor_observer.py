# SPDX-License-Identifier: Apache-2.0

"""Fail-closed source observers for attention tensors relevant to sketching."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import torch

from vllm.v1.core.kivo_kv_block_score_store import (
    KivoKVBlockScore,
    update_block_scores,
)

SCHEMA_VERSION = "kivo_source_s3_3a_attention_tensor_sketch_observer_v1"
POLICY_NAME = "observe_attention_tensors_for_sketch"
S3_3B_SCHEMA_VERSION = "kivo_source_s3_3b_shadow_kv_block_sketch_v1"
S3_3B_POLICY_NAME = "shadow_kv_block_sketch"
S3_3C_PLAN_SCHEMA_VERSION = "kivo_source_s3_3c_active_sketch_plan_v1"
S3_3C_POLICY_NAME = "active_sketch_kv_metadata_alias"
S3_3B_SKETCH_METHOD = "random_projection_l2"
HOOK_POINT = "unified_attention_with_output"
_PAD_SLOT_ID = -1

_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_ID = 0
_PROJECTION_CACHE_LOCK = threading.Lock()
_PROJECTION_CACHE: dict[tuple[int, int, str, int], torch.Tensor] = {}
_PLAN_CACHE_LOCK = threading.Lock()
_LATEST_SKETCH_PLAN_BY_LAYER: dict[tuple[int, int | None], dict[str, Any]] = {}


def _next_sequence_id() -> int:
    global _SEQUENCE_ID
    with _SEQUENCE_LOCK:
        _SEQUENCE_ID += 1
        return _SEQUENCE_ID


def _observation_path() -> Path | None:
    value = os.getenv("KIVO_SOURCE_OBSERVE_PATH") or os.getenv(
        "KIVO_SOURCE_OBS_PATH"
    )
    return Path(value) if value else None


def _retention_score_bridge_enabled() -> bool:
    return (
        os.getenv("KIVO_KV_RETENTION_ENABLE") == "1"
        and os.getenv("KIVO_KV_RETENTION_POLICY") == "countsketch_online"
    )


def _safe_tensor_info(value: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "present": value is not None,
        "shape": None,
        "dtype": None,
        "device": None,
        "ndim": None,
        "numel": None,
    }
    if value is None:
        return info

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            info["shape"] = [int(dim) for dim in shape]
        except Exception:
            info["shape"] = str(shape)
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        try:
            info["ndim"] = int(ndim)
        except Exception:
            pass
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        info["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        info["device"] = str(device)
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            info["numel"] = int(numel())
        except Exception:
            pass
    return info


def _metadata_tensor(attn_metadata: Any, *names: str) -> Any:
    for name in names:
        value = getattr(attn_metadata, name, None)
        if value is not None:
            return value
    return None


def _layer_index(layer_name: str | None) -> int | None:
    if not layer_name:
        return None
    matches = re.findall(r"(?:layers?|h)\.(\d+)", layer_name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _backend_name(attn_layer: Any) -> str | None:
    backend = getattr(attn_layer, "attn_backend", None)
    get_name = getattr(backend, "get_name", None)
    if callable(get_name):
        try:
            return str(get_name())
        except Exception:
            pass
    impl = getattr(attn_layer, "impl", None)
    return type(impl).__name__ if impl is not None else None


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _maybe_update_retention_score_store(record: dict[str, Any]) -> None:
    if not _retention_score_bridge_enabled():
        return
    if record.get("sketch_computed") is not True:
        return
    block_rows = record.get("block_sketch_sample")
    if not isinstance(block_rows, list):
        return

    scores: list[KivoKVBlockScore] = []
    layer_index = record.get("layer_index")
    layer_id = int(layer_index) if isinstance(layer_index, int) else None
    sequence_id = record.get("sequence_id")
    step = int(sequence_id) if isinstance(sequence_id, int) else None
    source = str(record.get("policy_name") or "kivo_tensor_observer")

    for row in block_rows:
        if not isinstance(row, dict):
            continue
        block_id = row.get("block_id")
        score = row.get("score")
        if not isinstance(block_id, int):
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        scores.append(
            KivoKVBlockScore(
                block_id=block_id,
                score=score_value,
                source=source,
                step=step,
                layer_id=layer_id,
            )
        )
    if scores:
        update_block_scores(scores)


def _can_sketch(info: dict[str, Any]) -> bool:
    return bool(
        info["present"]
        and isinstance(info["shape"], list)
        and info["ndim"] is not None
        and info["ndim"] >= 1
        and info["numel"] is not None
        and info["numel"] > 0
    )


def _parse_env_int(name: str, *, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _parse_env_float(
    name: str,
    *,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return min(maximum, max(minimum, parsed))


def _safe_int_list(
    value: Any,
    *,
    max_numel: int = 256,
) -> list[int] | None:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            size = int(numel())
        except Exception:
            return None
        if size > max_numel:
            return None
    try:
        flat = value.detach().reshape(-1)
        if flat.numel() > max_numel:
            return None
        sample = flat.to("cpu").tolist()
    except Exception:
        return None
    try:
        return [int(item) for item in sample]
    except Exception:
        return None


def _last_valid_slot_id(slot_mapping: Any) -> tuple[int | None, str | None]:
    values = _safe_int_list(slot_mapping)
    if values is None:
        return None, "slot_mapping unavailable or too large"
    valid = [value for value in values if value != _PAD_SLOT_ID and value >= 0]
    if not valid:
        return None, "no valid slot ids in slot_mapping"
    return int(valid[-1]), None


def _projection_tensor(
    input_dim: int,
    sketch_dim: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    key = (input_dim, sketch_dim, str(device), seed)
    with _PROJECTION_CACHE_LOCK:
        cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    row_ids = torch.arange(input_dim, device=device, dtype=torch.int64).unsqueeze(1)
    col_ids = torch.arange(sketch_dim, device=device, dtype=torch.int64).unsqueeze(0)
    hashed = (
        row_ids * 1103515245
        + col_ids * 12345
        + int(seed) * 2654435761
    ) & 1
    projection = hashed.mul(2).sub(1).to(torch.float32)
    projection /= math.sqrt(float(sketch_dim))
    with _PROJECTION_CACHE_LOCK:
        _PROJECTION_CACHE[key] = projection
    return projection


def _store_latest_sketch_plan(record: dict[str, Any]) -> None:
    layer_index = record.get("layer_index")
    key = (int(record.get("pid", os.getpid())), int(layer_index)
           if layer_index is not None else None)
    plan = {
        "schema_version": record.get("schema_version"),
        "policy_name": record.get("policy_name"),
        "timestamp": record.get("timestamp"),
        "sequence_id": record.get("sequence_id"),
        "pid": key[0],
        "layer_name": record.get("layer_name"),
        "layer_index": key[1],
        "sketch_dim": record.get("sketch_dim"),
        "max_sketch_blocks": record.get("max_sketch_blocks"),
        "budget_ratio": record.get("budget_ratio"),
        "current_physical_block_id": record.get("current_physical_block_id"),
        "candidate_block_ids": list(record.get("candidate_block_ids_sample", [])),
        "selected_block_ids": list(record.get("selected_block_ids_sample", [])),
        "excluded_block_ids": list(record.get("excluded_block_ids_sample", [])),
        "sketch_computed": record.get("sketch_computed") is True,
        "sketch_blocker_reason": record.get("sketch_blocker_reason"),
    }
    with _PLAN_CACHE_LOCK:
        _LATEST_SKETCH_PLAN_BY_LAYER[key] = plan
        if len(_LATEST_SKETCH_PLAN_BY_LAYER) > 64:
            oldest_key = min(
                _LATEST_SKETCH_PLAN_BY_LAYER,
                key=lambda item: float(
                    _LATEST_SKETCH_PLAN_BY_LAYER[item].get("timestamp", 0.0) or 0.0
                ),
            )
            _LATEST_SKETCH_PLAN_BY_LAYER.pop(oldest_key, None)


def get_latest_sketch_plan(
    *,
    pid: int | None = None,
    layer_index: int | None = None,
) -> dict[str, Any] | None:
    effective_pid = int(pid if pid is not None else os.getpid())
    with _PLAN_CACHE_LOCK:
        if layer_index is not None:
            plan = _LATEST_SKETCH_PLAN_BY_LAYER.get((effective_pid, int(layer_index)))
            return dict(plan) if plan is not None else None
        candidates = [
            dict(plan)
            for (cached_pid, _), plan in _LATEST_SKETCH_PLAN_BY_LAYER.items()
            if cached_pid == effective_pid
        ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
    return candidates[-1]


def _sketch_block_tensor(
    block_tensor: torch.Tensor,
    *,
    sketch_dim: int,
    seed: int,
) -> tuple[float, list[float]]:
    flat = block_tensor.reshape(-1).to(torch.float32)
    projection = _projection_tensor(
        int(flat.numel()),
        sketch_dim,
        device=flat.device,
        seed=seed,
    )
    sketch = torch.matmul(flat, projection)
    norm = float(torch.linalg.vector_norm(flat).item())
    sample = [float(item) for item in sketch.detach().to("cpu").tolist()[:sketch_dim]]
    return norm, sample


def build_attention_tensor_record(
    *,
    hook_point: str,
    layer_name: str | None,
    attn_layer: Any,
    query: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    attn_metadata: Any,
    slot_mapping: Any,
) -> dict[str, Any]:
    """Build a metadata-only record without copying tensor contents."""
    query_info = _safe_tensor_info(query)
    key_info = _safe_tensor_info(key)
    value_info = _safe_tensor_info(value)
    kv_cache_info = _safe_tensor_info(kv_cache)
    slot_mapping_info = _safe_tensor_info(slot_mapping)
    block_table = _metadata_tensor(
        attn_metadata,
        "block_table",
        "block_table_tensor",
    )
    block_table_info = _safe_tensor_info(block_table)

    can_build_query_sketch = _can_sketch(query_info)
    can_build_key_sketch = _can_sketch(key_info)
    can_build_value_sketch = _can_sketch(value_info)
    can_build_kv_block_sketch = bool(
        _can_sketch(kv_cache_info)
        and kv_cache_info["ndim"] is not None
        and kv_cache_info["ndim"] >= 4
    )
    if can_build_kv_block_sketch:
        recommended_sketch_source = "kv_cache"
    elif can_build_key_sketch:
        recommended_sketch_source = "key"
    elif can_build_query_sketch:
        recommended_sketch_source = "query"
    elif can_build_value_sketch:
        recommended_sketch_source = "value"
    elif block_table_info["present"] or slot_mapping_info["present"]:
        recommended_sketch_source = "metadata_proxy_only"
    else:
        recommended_sketch_source = "unknown"

    caveats = [
        "tensor contents were not copied or sketched",
        "key and value inputs contain only tokens in the current forward pass",
        "KV cache layout is backend-specific and may be quantized",
        "visibility does not establish acceptable runtime sketch overhead",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "sequence_id": _next_sequence_id(),
        "policy_name": POLICY_NAME,
        "hook_point": hook_point,
        "pid": os.getpid(),
        "layer_name": layer_name,
        "layer_index": _layer_index(layer_name),
        "attention_backend": _backend_name(attn_layer),
        "query_present": query_info["present"],
        "query_shape": query_info["shape"],
        "query_dtype": query_info["dtype"],
        "query_device": query_info["device"],
        "query_ndim": query_info["ndim"],
        "query_numel": query_info["numel"],
        "key_present": key_info["present"],
        "key_shape": key_info["shape"],
        "key_dtype": key_info["dtype"],
        "key_device": key_info["device"],
        "key_ndim": key_info["ndim"],
        "key_numel": key_info["numel"],
        "value_present": value_info["present"],
        "value_shape": value_info["shape"],
        "value_dtype": value_info["dtype"],
        "value_device": value_info["device"],
        "value_ndim": value_info["ndim"],
        "value_numel": value_info["numel"],
        "kv_cache_present": kv_cache_info["present"],
        "kv_cache_type": type(kv_cache).__name__ if kv_cache is not None else None,
        "kv_cache_shape": kv_cache_info["shape"],
        "kv_cache_dtype": kv_cache_info["dtype"],
        "kv_cache_device": kv_cache_info["device"],
        "kv_cache_ndim": kv_cache_info["ndim"],
        "kv_cache_numel": kv_cache_info["numel"],
        "slot_mapping_present": slot_mapping_info["present"],
        "slot_mapping_shape": slot_mapping_info["shape"],
        "slot_mapping_dtype": slot_mapping_info["dtype"],
        "slot_mapping_device": slot_mapping_info["device"],
        "block_table_present": block_table_info["present"],
        "block_table_shape": block_table_info["shape"],
        "block_table_dtype": block_table_info["dtype"],
        "block_table_device": block_table_info["device"],
        "attention_metadata_present": attn_metadata is not None,
        "attention_metadata_type": (
            type(attn_metadata).__name__ if attn_metadata is not None else None
        ),
        "can_build_query_sketch": can_build_query_sketch,
        "can_build_key_sketch": can_build_key_sketch,
        "can_build_value_sketch": can_build_value_sketch,
        "can_build_kv_block_sketch": can_build_kv_block_sketch,
        "recommended_sketch_source": recommended_sketch_source,
        "sketch_observer_caveats": caveats,
        "kivo_source_enable_seen": os.getenv("KIVO_SOURCE_ENABLE") == "1",
        "kivo_source_policy_seen": os.getenv("KIVO_SOURCE_POLICY"),
        "observe_path_present": _observation_path() is not None,
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
    }


def _build_kv_block_sketch_record(
    *,
    schema_version: str,
    policy_name: str,
    hook_point: str,
    layer_name: str | None,
    attn_layer: Any,
    query: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    attn_metadata: Any,
    slot_mapping: Any,
) -> dict[str, Any]:
    """Build a bounded real KV-cache block sketch record."""
    query_info = _safe_tensor_info(query)
    key_info = _safe_tensor_info(key)
    value_info = _safe_tensor_info(value)
    kv_cache_info = _safe_tensor_info(kv_cache)
    slot_mapping_info = _safe_tensor_info(slot_mapping)
    sketch_dim = _parse_env_int("KIVO_SOURCE_SKETCH_DIM", default=8, minimum=1)
    max_sketch_blocks = _parse_env_int(
        "KIVO_SOURCE_MAX_SKETCH_BLOCKS", default=4, minimum=1
    )
    budget_ratio = _parse_env_float(
        "KIVO_SOURCE_BUDGET_RATIO",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
    )
    sketch_seed = _parse_env_int("KIVO_SOURCE_SKETCH_SEED", default=123, minimum=0)
    current_slot_id = None
    blocker = None
    block_sketch_sample: list[dict[str, Any]] = []
    candidate_block_ids: list[int] = []
    selected_block_ids: list[int] = []
    excluded_block_ids: list[int] = []
    current_physical_block_id: int | None = None
    sketch_computed = False
    block_size = None

    if not kv_cache_info["present"]:
        blocker = blocker or "kv_cache unavailable"
    elif not isinstance(kv_cache_info["shape"], list):
        blocker = blocker or "kv_cache shape unavailable"
    elif kv_cache_info["ndim"] != 5:
        blocker = blocker or "kv_cache ndim is not 5"
    elif kv_cache_info["shape"][1] != 2:
        blocker = blocker or "kv_cache second dimension is not 2"
    else:
        block_size = int(kv_cache_info["shape"][2])
        if block_size <= 0:
            blocker = blocker or "invalid kv_cache block size"
        else:
            current_slot_id, slot_blocker = _last_valid_slot_id(slot_mapping)
            if current_slot_id is None:
                blocker = blocker or slot_blocker or "current slot id unavailable"
            else:
                current_physical_block_id = int(current_slot_id // block_size)
                num_blocks = int(kv_cache_info["shape"][0])
                if (
                    current_physical_block_id < 0
                    or current_physical_block_id >= num_blocks
                ):
                    blocker = blocker or "current physical block id out of range"
                else:
                    lowest_block_id = max(
                        0,
                        current_physical_block_id - max_sketch_blocks + 1,
                    )
                    candidate_block_ids = list(
                        range(current_physical_block_id, lowest_block_id - 1, -1)
                    )
                    if not candidate_block_ids:
                        blocker = blocker or "no candidate block ids available"
                    else:
                        try:
                            for block_id in candidate_block_ids:
                                k_block = kv_cache[block_id, 0]
                                v_block = kv_cache[block_id, 1]
                                k_l2_norm, k_sketch = _sketch_block_tensor(
                                    k_block,
                                    sketch_dim=sketch_dim,
                                    seed=sketch_seed,
                                )
                                v_l2_norm, v_sketch = _sketch_block_tensor(
                                    v_block,
                                    sketch_dim=sketch_dim,
                                    seed=sketch_seed + 1,
                                )
                                score = float(k_l2_norm + v_l2_norm)
                                block_sketch_sample.append({
                                    "block_id": int(block_id),
                                    "k_l2_norm": k_l2_norm,
                                    "v_l2_norm": v_l2_norm,
                                    "score": score,
                                    "k_sketch_sample": k_sketch[:sketch_dim],
                                    "v_sketch_sample": v_sketch[:sketch_dim],
                                })
                        except Exception as exc:
                            blocker = (
                                "kv block sketch computation failed: "
                                f"{type(exc).__name__}"
                            )

    if blocker is None and block_sketch_sample:
        selected_budget = max(
            1,
            int(math.ceil(len(candidate_block_ids) * budget_ratio)),
        )
        current_block_id = int(candidate_block_ids[0])
        older_candidates = sorted(
            block_sketch_sample[1:],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        selected_set = {current_block_id}
        for item in older_candidates[: max(0, selected_budget - 1)]:
            selected_set.add(int(item["block_id"]))
        selected_block_ids = [
            int(block_id)
            for block_id in candidate_block_ids
            if block_id in selected_set
        ]
        excluded_block_ids = [
            int(block_id)
            for block_id in candidate_block_ids
            if block_id not in selected_set
        ]
        sketch_computed = True

    caveats = [
        "shadow mode only; KV cache and attention metadata are unchanged",
        "candidate blocks are derived from the current slot and bounded recency only",
        "tiny block sketches are recorded as summaries, not full tensor dumps",
        (
            "this does not prove memory reduction, latency reduction, "
            "or selected attention"
        ),
    ]
    if blocker is not None:
        caveats.append(f"blocker: {blocker}")

    return {
        "schema_version": schema_version,
        "timestamp": time.time(),
        "sequence_id": _next_sequence_id(),
        "policy_name": policy_name,
        "hook_point": hook_point,
        "pid": os.getpid(),
        "layer_name": layer_name,
        "layer_index": _layer_index(layer_name),
        "attention_backend": _backend_name(attn_layer),
        "sketch_source": "kv_cache",
        "sketch_method": S3_3B_SKETCH_METHOD,
        "sketch_dim": sketch_dim,
        "max_sketch_blocks": max_sketch_blocks,
        "budget_ratio": budget_ratio,
        "query_present": query_info["present"],
        "key_present": key_info["present"],
        "value_present": value_info["present"],
        "kv_cache_present": kv_cache_info["present"],
        "kv_cache_shape": kv_cache_info["shape"],
        "kv_cache_dtype": kv_cache_info["dtype"],
        "kv_cache_device": kv_cache_info["device"],
        "slot_mapping_present": slot_mapping_info["present"],
        "slot_mapping_shape": slot_mapping_info["shape"],
        "slot_mapping_dtype": slot_mapping_info["dtype"],
        "slot_mapping_device": slot_mapping_info["device"],
        "block_size": block_size,
        "current_slot_id": current_slot_id,
        "current_physical_block_id": current_physical_block_id,
        "candidate_block_count": len(candidate_block_ids),
        "candidate_block_ids_sample": candidate_block_ids[:16],
        "selected_block_count": len(selected_block_ids),
        "selected_block_ids_sample": selected_block_ids[:16],
        "excluded_block_count": len(excluded_block_ids),
        "excluded_block_ids_sample": excluded_block_ids[:16],
        "block_sketch_sample": block_sketch_sample[:16],
        "sketch_computed": sketch_computed,
        "sketch_blocker_reason": blocker,
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "caveats": caveats,
    }


def build_shadow_kv_block_sketch_record(
    *,
    hook_point: str,
    layer_name: str | None,
    attn_layer: Any,
    query: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    attn_metadata: Any,
    slot_mapping: Any,
) -> dict[str, Any]:
    return _build_kv_block_sketch_record(
        schema_version=S3_3B_SCHEMA_VERSION,
        policy_name=S3_3B_POLICY_NAME,
        hook_point=hook_point,
        layer_name=layer_name,
        attn_layer=attn_layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping,
    )


def build_active_sketch_plan_record(
    *,
    hook_point: str,
    layer_name: str | None,
    attn_layer: Any,
    query: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    attn_metadata: Any,
    slot_mapping: Any,
) -> dict[str, Any]:
    return _build_kv_block_sketch_record(
        schema_version=S3_3C_PLAN_SCHEMA_VERSION,
        policy_name=S3_3C_POLICY_NAME,
        hook_point=hook_point,
        layer_name=layer_name,
        attn_layer=attn_layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping,
    )


def maybe_observe_attention_tensors(
    *,
    hook_point: str,
    layer_name: str | None,
    attn_layer: Any,
    query: Any,
    key: Any,
    value: Any,
    kv_cache: Any,
    attn_metadata: Any,
    slot_mapping: Any,
) -> dict[str, Any] | None:
    """Write an S3.3A observation when explicitly enabled.

    Every failure is swallowed so observation cannot break attention.
    """
    try:
        if os.getenv("KIVO_SOURCE_ENABLE") != "1":
            return None
        policy_name = os.getenv("KIVO_SOURCE_POLICY")
        if policy_name not in {
            POLICY_NAME,
            S3_3B_POLICY_NAME,
            S3_3C_POLICY_NAME,
        }:
            return None
        output_path = _observation_path()
        if output_path is None and not _retention_score_bridge_enabled():
            return None
        if policy_name == POLICY_NAME:
            record = build_attention_tensor_record(
                hook_point=hook_point,
                layer_name=layer_name,
                attn_layer=attn_layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
            )
        elif policy_name == S3_3B_POLICY_NAME:
            record = build_shadow_kv_block_sketch_record(
                hook_point=hook_point,
                layer_name=layer_name,
                attn_layer=attn_layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
            )
        else:
            record = build_active_sketch_plan_record(
                hook_point=hook_point,
                layer_name=layer_name,
                attn_layer=attn_layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
            )
            if record.get("sketch_computed") is True:
                _store_latest_sketch_plan(record)
        _maybe_update_retention_score_store(record)
        if output_path is not None:
            _append_jsonl(output_path, record)
        return record
    except Exception:
        return None
