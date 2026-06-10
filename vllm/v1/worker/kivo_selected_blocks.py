# SPDX-License-Identifier: Apache-2.0

"""Source-level Kivo S1 selected-block observation and mutation helpers."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def _env_flag(name: str) -> bool:
    return os.getenv(name) == "1"


def _safe_summary(value: Any, depth: int = 0) -> dict[str, Any]:
    try:
        value_type = type(value)
        type_name = f"{value_type.__module__}.{value_type.__qualname__}"
        if value is None or isinstance(value, (str, int, float, bool)):
            return {"type": type_name, "value": value}
        if depth >= 2:
            return {"type": type_name}
        if isinstance(value, dict):
            keys = []
            for index, key in enumerate(value):
                if index >= 24:
                    break
                keys.append(str(key))
            return {"type": type_name, "length": len(value), "keys": keys}
        if isinstance(value, (list, tuple)):
            return {
                "type": type_name,
                "length": len(value),
                "items": [
                    _safe_summary(item, depth + 1)
                    for item in value[:8]
                ],
            }
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        if shape is not None or dtype is not None or device is not None:
            summary = {"type": type_name}
            if shape is not None:
                try:
                    summary["shape"] = [int(item) for item in shape]
                except Exception:
                    summary["shape"] = str(shape)[:120]
            if dtype is not None:
                summary["dtype"] = str(dtype)
            if device is not None:
                summary["device"] = str(device)
            return summary
        return {"type": type_name}
    except Exception as exc:  # pragma: no cover - defensive path
        return {"summary_error": f"{type(exc).__name__}: {exc}"}


def _tensor_info(value: Any) -> dict[str, Any]:
    summary = _safe_summary(value)
    return {
        "present": value is not None,
        "type": summary.get("type"),
        "shape": summary.get("shape"),
        "dtype": summary.get("dtype"),
        "device": summary.get("device"),
    }


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
        try:
            return len(value)
        except Exception:
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


def _is_tensor_like(value: Any) -> bool:
    return any(
        getattr(value, attr, None) is not None
        for attr in ("shape", "dtype", "device")
    )


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    parent = output_path.parent
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _slot_mapping_view(instance: Any) -> Any:
    slot_mapping = getattr(instance, "slot_mapping", None)
    return getattr(slot_mapping, "gpu", slot_mapping)


def _block_table_view(instance: Any) -> Any:
    block_table = getattr(instance, "block_table", None)
    return getattr(block_table, "gpu", block_table)


def _maybe_mutate_slot_mapping(slot_mapping: Any) -> tuple[bool, dict[str, Any]]:
    record: dict[str, Any] = {
        "mutation_policy": None,
        "mutation_blocker_reason": None,
        "old_value": None,
        "new_value": None,
        "mutation_index": None,
    }
    if slot_mapping is None:
        record["mutation_blocker_reason"] = (
            "no safe Python-level slot mapping result found"
        )
        return False, record
    if not _is_tensor_like(slot_mapping):
        record["mutation_blocker_reason"] = (
            "slot mapping is not tensor-like; selected-slot state "
            "mutation not attempted"
        )
        return False, record
    slot_info = _tensor_info(slot_mapping)
    record["slot_mapping_shape"] = slot_info["shape"]
    record["slot_mapping_dtype"] = slot_info["dtype"]
    record["slot_mapping_device"] = slot_info["device"]
    if not _is_int_dtype(getattr(slot_mapping, "dtype", None)):
        record["mutation_blocker_reason"] = (
            "tensor-like slot mapping requires integer dtype for safe mutation"
        )
        return False, record
    numel = _tensor_numel(slot_mapping)
    if numel is None or numel < 2:
        record["mutation_blocker_reason"] = (
            "tensor-like slot mapping needs at least two elements"
        )
        return False, record
    try:
        old_tensor_value = slot_mapping[-1]
        new_tensor_value = slot_mapping[-2]
        old_value = (
            old_tensor_value.item()
            if hasattr(old_tensor_value, "item")
            else old_tensor_value
        )
        new_value = (
            new_tensor_value.item()
            if hasattr(new_tensor_value, "item")
            else new_tensor_value
        )
        slot_mapping[-1] = new_tensor_value
        record.update(
            {
                "mutation_policy": "mask_last_slot",
                "mutation_blocker_reason": None,
                "old_value": old_value,
                "new_value": new_value,
                "mutation_index": numel - 1,
            }
        )
        return True, record
    except Exception as exc:
        record["mutation_blocker_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return False, record


def maybe_observe_compute_slot_mapping(
    instance: Any,
    *,
    module_file: str,
    function_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> None:
    if not _env_flag("KIVO_SOURCE_ENABLE"):
        return

    output_path = os.getenv("KIVO_SOURCE_OBS_PATH")
    if not output_path:
        return

    record: dict[str, Any] | None = None
    try:
        active_enabled = _env_flag("KIVO_SOURCE_ACTIVE")
        policy = os.getenv("KIVO_SOURCE_POLICY", "")
        slot_mapping = _slot_mapping_view(instance)
        block_table = _block_table_view(instance)
        mutation_attempted = active_enabled and policy == "mask_last_slot"
        mutation_applied = False
        mutation_record = {
            "mutation_policy": None,
            "mutation_blocker_reason": None,
            "old_value": None,
            "new_value": None,
            "mutation_index": None,
        }
        if active_enabled and not mutation_attempted:
            mutation_record["mutation_blocker_reason"] = (
                f"unsupported mutation policy: {policy or '<unset>'}"
            )
        if mutation_attempted:
            mutation_applied, mutation_record = _maybe_mutate_slot_mapping(
                slot_mapping
            )
        result_summary = _safe_summary(result)
        slot_info = _tensor_info(slot_mapping)
        block_info = _tensor_info(block_table)
        num_blocks_per_row = getattr(instance, "num_blocks_per_row", None)
        try:
            num_blocks_preview = (
                [
                    int(item)
                    for item in list(
                        num_blocks_per_row[:32]  # type: ignore[index]
                    )
                ]
                if num_blocks_per_row is not None
                else None
            )
        except Exception:
            num_blocks_preview = None
        record = {
            "schema_version": "kivo_source_s1_block_table_v1",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "hook_name": "BlockTable.compute_slot_mapping",
            "class_name": type(instance).__qualname__,
            "function_name": function_name,
            "args_summary": [_safe_summary(item) for item in args[:8]],
            "slot_mapping_present": slot_mapping is not None,
            "slot_mapping_type": slot_info["type"],
            "slot_mapping_shape": slot_info["shape"],
            "slot_mapping_dtype": slot_info["dtype"],
            "slot_mapping_device": slot_info["device"],
            "block_table_present": block_table is not None,
            "block_table_type": block_info["type"],
            "block_table_shape": block_info["shape"],
            "block_table_dtype": block_info["dtype"],
            "block_table_device": block_info["device"],
            "result_type": result_summary.get("type"),
            "result_summary": result_summary,
            "block_size": getattr(instance, "block_size", None),
            "num_blocks_per_row": num_blocks_preview,
            "max_num_blocks_per_req": getattr(
                instance, "max_num_blocks_per_req", None
            ),
            "max_num_reqs": getattr(instance, "max_num_reqs", None),
            "active_enabled": active_enabled,
            "mutation_attempted": mutation_attempted,
            "mutation_applied": mutation_applied,
            "mutation_policy": mutation_record["mutation_policy"],
            "mutation_blocker_reason": mutation_record[
                "mutation_blocker_reason"
            ],
            "old_value": mutation_record["old_value"],
            "new_value": mutation_record["new_value"],
            "mutation_index": mutation_record["mutation_index"],
            "runtime_behavior_changed": mutation_applied,
            "active_routing": mutation_applied,
            "measured_runtime_reduction": False,
            "caveats": [
                "source-level experiment hook",
                "fail closed on any hook error",
                "no scheduler, attention, or KV cache mutation",
            ],
        }
        _append_jsonl(output_path, record)
    except Exception as exc:
        if _env_flag("KIVO_SOURCE_FAIL_CLOSED"):
            fallback_record = record or {
                "schema_version": "kivo_source_s1_block_table_v1",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "hook_name": "BlockTable.compute_slot_mapping",
                "class_name": type(instance).__qualname__,
                "function_name": function_name,
                "args_summary": [_safe_summary(item) for item in args[:8]],
                "mutation_attempted": False,
                "mutation_applied": False,
                "runtime_behavior_changed": False,
                "active_routing": False,
                "measured_runtime_reduction": False,
                "caveats": [
                    "source-level experiment hook",
                    "fail closed on any hook error",
                ],
            }
            fallback_record["helper_exception_type"] = type(exc).__name__
            fallback_record["helper_exception_message"] = str(exc)
            try:
                _append_jsonl(output_path, fallback_record)
            except Exception:
                pass
            return
        raise
