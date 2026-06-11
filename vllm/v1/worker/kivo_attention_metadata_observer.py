# SPDX-License-Identifier: Apache-2.0

"""Source-level observer for attention metadata visibility.

This helper records the backend-agnostic metadata visible at the
``build_attn_metadata`` boundary without mutating runtime behavior.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID as _PAD_SLOT_ID
except Exception:  # pragma: no cover - defensive fallback
    _PAD_SLOT_ID = -1

_SCHEMA_VERSION = "kivo_source_s3_0b_attention_metadata_observer_v1"
_POLICY_NAME = "observe_attention_metadata"
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
    dtype = getattr(slot_mapping, "dtype", None)
    if dtype is not None and not _is_int_dtype(dtype):
        return None, "slot_mapping is not integer dtype"
    numel = _tensor_numel(slot_mapping)
    if numel is None:
        return None, "slot_mapping size unavailable"
    if numel > 4096:
        return None, "slot_mapping too large for safe exact scan"
    values = _bounded_sample(slot_mapping, limit=numel)
    if not values:
        return None, "slot_mapping sample unavailable"
    visible_blocks: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            slot_id = int(value)
        except Exception:
            continue
        if slot_id == pad_slot_id or slot_id < 0:
            continue
        block_id = slot_id // block_size
        if block_id not in seen:
            seen.add(block_id)
            visible_blocks.append(block_id)
    return len(visible_blocks), None


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


def maybe_observe_attention_metadata(
    *,
    hook_point: str,
    kv_cache_group_id: int,
    common_attn_metadata: Any,
    kv_cache_spec: Any,
) -> dict[str, Any] | None:
    """Record a metadata-only observation if the source hook is enabled."""

    if not _env_flag("KIVO_SOURCE_ENABLE"):
        return None
    if os.getenv("KIVO_SOURCE_POLICY") != _POLICY_NAME:
        return None
    output_path = _observation_path()
    if output_path is None:
        return None

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
    block_table_sample = _bounded_sample(block_table_tensor, limit=16)
    slot_mapping_sample = _bounded_sample(slot_mapping, limit=16)
    record = {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": time.time(),
        "sequence_id": _next_sequence_id(),
        "pid": os.getpid(),
        "policy_name": _POLICY_NAME,
        "hook_point": hook_point,
        "kv_cache_group_id": kv_cache_group_id,
        "block_size": block_size,
        "pad_slot_id": pad_slot_id,
        "block_table_tensor_present": block_table_tensor is not None,
        "block_table_tensor_shape": _safe_tensor_info(block_table_tensor)["shape"],
        "block_table_tensor_dtype": _safe_tensor_info(block_table_tensor)["dtype"],
        "block_table_tensor_device": _safe_tensor_info(block_table_tensor)[
            "device"
        ],
        "slot_mapping_present": slot_mapping is not None,
        "slot_mapping_shape": _safe_tensor_info(slot_mapping)["shape"],
        "slot_mapping_dtype": _safe_tensor_info(slot_mapping)["dtype"],
        "slot_mapping_device": _safe_tensor_info(slot_mapping)["device"],
        "query_start_loc_shape": _safe_tensor_info(query_start_loc)["shape"],
        "query_start_loc_dtype": _safe_tensor_info(query_start_loc)["dtype"],
        "query_start_loc_device": _safe_tensor_info(query_start_loc)["device"],
        "seq_lens_shape": _safe_tensor_info(seq_lens)["shape"],
        "seq_lens_dtype": _safe_tensor_info(seq_lens)["dtype"],
        "seq_lens_device": _safe_tensor_info(seq_lens)["device"],
        "positions_shape": _safe_tensor_info(positions)["shape"],
        "positions_dtype": _safe_tensor_info(positions)["dtype"],
        "positions_device": _safe_tensor_info(positions)["device"],
        "max_query_len": int(max_query_len) if max_query_len is not None else None,
        "max_seq_len": int(max_seq_len) if max_seq_len is not None else None,
        "visible_block_count_estimate": visible_block_count_estimate,
        "visible_block_count_estimate_caveat": estimate_caveat,
        "block_table_tensor_sample": block_table_sample,
        "slot_mapping_sample": slot_mapping_sample,
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
    }
    _append_jsonl(output_path, record)
    return record
