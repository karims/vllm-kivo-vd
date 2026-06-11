# SPDX-License-Identifier: Apache-2.0

"""Fail-closed source observer for attention tensors relevant to sketching."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "kivo_source_s3_3a_attention_tensor_sketch_observer_v1"
POLICY_NAME = "observe_attention_tensors_for_sketch"
HOOK_POINT = "unified_attention_with_output"

_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_ID = 0


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


def _can_sketch(info: dict[str, Any]) -> bool:
    return bool(
        info["present"]
        and isinstance(info["shape"], list)
        and info["ndim"] is not None
        and info["ndim"] >= 1
        and info["numel"] is not None
        and info["numel"] > 0
    )


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
        if os.getenv("KIVO_SOURCE_POLICY") != POLICY_NAME:
            return None
        output_path = _observation_path()
        if output_path is None:
            return None
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
        _append_jsonl(output_path, record)
        return record
    except Exception:
        return None
