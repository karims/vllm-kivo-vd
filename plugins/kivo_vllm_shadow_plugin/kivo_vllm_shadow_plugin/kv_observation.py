# SPDX-License-Identifier: Apache-2.0

"""Copied-metadata observations for KVCacheManager.get_block_ids."""

from __future__ import annotations

import json
import operator
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "phase12_6d_kv_observation_v1"
DEFAULT_PREVIEW_LIMIT = 64
DEFAULT_REPR_LIMIT = 320
MAX_ARRAY_CONVERSION_ITEMS = 4096
_SELF_ATTRS = (
    "block_size",
    "enable_caching",
    "num_gpu_blocks",
    "num_kv_cache_groups",
)


def _bounded_repr(
    value: Any,
    limit: int = DEFAULT_REPR_LIMIT,
    *,
    depth: int = 0,
) -> str:
    if depth < 2 and isinstance(value, Mapping):
        try:
            keys = list(value.keys())[:8]
        except Exception:
            keys = []
        preview = (
            f"<{type(value).__qualname__} len={_safe_length(value)} "
            f"keys={keys!r}>"
        )
        return preview[:limit]
    if depth < 2 and isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        try:
            items = [
                _bounded_repr(item, 80, depth=depth + 1)
                for item in value[:8]
            ]
        except Exception:
            items = []
        preview = (
            f"<{type(value).__qualname__} len={_safe_length(value)} "
            f"preview={items!r}>"
        )
        return preview[:limit]
    try:
        preview = repr(value)
    except Exception as exc:
        preview = f"<repr failed: {type(exc).__name__}>"
    return preview[:limit]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(operator.index(value))
    except TypeError:
        return None


def _as_sequence(value: Any) -> Any:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if isinstance(value, Sequence):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        size = getattr(value, "size", None)
        if callable(size):
            try:
                size = getattr(value, "numel")()
            except Exception:
                size = None
        if isinstance(size, int) and size > MAX_ARRAY_CONVERSION_ITEMS:
            return None
        try:
            return tolist()
        except Exception:
            return None
    return None


def _collect_block_ids(
    value: Any,
    *,
    preview_limit: int,
) -> tuple[list[int], int, int | None, int | None]:
    preview: list[int] = []
    count = 0
    minimum: int | None = None
    maximum: int | None = None
    stack = [value]
    while stack:
        current = stack.pop()
        integer = _as_int(current)
        if integer is not None:
            count += 1
            if len(preview) < preview_limit:
                preview.append(integer)
            minimum = integer if minimum is None else min(minimum, integer)
            maximum = integer if maximum is None else max(maximum, integer)
            continue
        sequence = _as_sequence(current)
        if sequence is None:
            continue
        try:
            items = list(sequence)
        except Exception:
            continue
        stack.extend(reversed(items))
    return preview, count, minimum, maximum


def _safe_length(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _bounded_repr(value, 80)


def _self_attrs_preview(instance: Any) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for name in _SELF_ATTRS:
        try:
            if hasattr(instance, name):
                preview[name] = _safe_scalar(getattr(instance, name))
        except Exception:
            continue
    return preview


def build_kv_observation(
    *,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
) -> dict[str, Any]:
    if preview_limit < 0:
        raise ValueError("preview_limit must be non-negative")
    block_ids, block_id_count, minimum, maximum = _collect_block_ids(
        result,
        preview_limit=preview_limit,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "hook": "KVCacheManager.get_block_ids",
        "class_name": type(instance).__qualname__,
        "method_name": "get_block_ids",
        "result_type": _type_name(result),
        "result_repr_preview": _bounded_repr(result),
        "result_length": _safe_length(result),
        "block_ids_preview": block_ids,
        "block_id_count": block_id_count,
        "block_ids_preview_truncated": block_id_count > len(block_ids),
        "min_block_id": minimum,
        "max_block_id": maximum,
        "args_type_summary": [_type_name(value) for value in args[:8]],
        "kwargs_keys": sorted(str(key) for key in kwargs)[:32],
        "self_type": _type_name(instance),
        "self_attrs_preview": _self_attrs_preview(instance),
        "active_routing": False,
        "measured_runtime_reduction": False,
        "runtime_behavior_changed": False,
        "mutation": False,
        "caveats": [
            "copied metadata only; result returned unchanged",
            "block ID preview is bounded",
            "no KV, scheduler, block-table, slot, or attention mutation",
        ],
    }


def append_kv_observation(
    path: str | Path,
    observation: dict[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(observation, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        output_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def record_kv_get_block_ids_observation(
    *,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    output_path: str | Path,
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
) -> dict[str, Any]:
    observation = build_kv_observation(
        instance=instance,
        args=args,
        kwargs=kwargs,
        result=result,
        preview_limit=preview_limit,
    )
    append_kv_observation(output_path, observation)
    return observation
