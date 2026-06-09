#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Install, inspect, or restore a reversible installed-wheel vLLM patch."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BEGIN_MARKER = "# KIVO_PHASE12_7_BEGIN"
END_MARKER = "# KIVO_PHASE12_7_END"
ACTIVE_BEGIN_MARKER = "# KIVO_PHASE12_8_9_BEGIN"
ACTIVE_END_MARKER = "# KIVO_PHASE12_8_9_END"
BLOCK_TABLE_ACTIVE_BEGIN_MARKER = "# KIVO_PHASE12_10_BEGIN"
BLOCK_TABLE_ACTIVE_END_MARKER = "# KIVO_PHASE12_10_END"
MANIFEST_NAME = "phase12_7_patch_manifest.json"


@dataclass(frozen=True)
class PatchTarget:
    name: str
    relative_path: str
    class_name: str
    method_name: str
    risk: str
    reason: str


TARGETS = (
    PatchTarget(
        "slot_mappings",
        "v1/worker/gpu_model_runner.py",
        "GPUModelRunner",
        "_get_slot_mappings",
        "high",
        "Preferred observation point for runtime slot-mapping results.",
    ),
    PatchTarget(
        "slot_mappings_active_ladder",
        "v1/worker/gpu_model_runner.py",
        "GPUModelRunner",
        "_get_slot_mappings",
        "very high",
        "Guarded shallow-copy mutation ladder for slot-mapping results.",
    ),
    PatchTarget(
        "compute_slot_mapping",
        "v1/worker/block_table.py",
        "BlockTable",
        "compute_slot_mapping",
        "high",
        "Called in the worker path but returns no metadata object.",
    ),
    PatchTarget(
        "block_table_compute_slot_mapping_active",
        "v1/worker/block_table.py",
        "BlockTable",
        "compute_slot_mapping",
        "very high",
        "Guarded lower-level slot-mapping mutation experiment.",
    ),
    PatchTarget(
        "attention_metadata",
        "v1/worker/gpu_model_runner.py",
        "GPUModelRunner",
        "_build_attention_metadata",
        "high",
        "Attention metadata is useful but execution-sensitive.",
    ),
    PatchTarget(
        "block_table_cpu",
        "v1/worker/block_table.py",
        "BlockTable",
        "get_cpu_tensor",
        "medium",
        "Read-oriented block-table view may not be called during generation.",
    ),
    PatchTarget(
        "kv_get_block_ids",
        "v1/core/kv_cache_manager.py",
        "KVCacheManager",
        "get_block_ids",
        "medium",
        "Plugin patch installed but did not observe subprocess calls.",
    ),
    PatchTarget(
        "scheduler_schedule",
        "v1/core/sched/scheduler.py",
        "Scheduler",
        "schedule",
        "high",
        "Fallback inventory target; scheduler mutation remains forbidden.",
    ),
)

PATCH_HELPER = r'''
# KIVO_PHASE12_7_BEGIN
def _kivo_phase12_7_safe_summary(value, depth=0):
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
                    _kivo_phase12_7_safe_summary(item, depth + 1)
                    for item in value[:8]
                ],
            }
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                shape = [int(item) for item in shape]
            except Exception:
                shape = str(shape)[:120]
            return {"type": type_name, "shape": shape}
        return {"type": type_name}
    except Exception as exc:
        return {"summary_error": f"{type(exc).__name__}: {exc}"}


def _kivo_phase12_7_names(value, fragments):
    found = set()
    try:
        names = []
        if isinstance(value, dict):
            for index, key in enumerate(value):
                if index >= 128:
                    break
                names.append(str(key))
        for index, name in enumerate(getattr(value, "__dict__", {})):
            if index >= 128:
                break
            names.append(str(name))
        for name in names:
            lowered = name.lower()
            if any(fragment in lowered for fragment in fragments):
                found.add(name)
    except Exception:
        pass
    return sorted(found)[:48]


def _kivo_phase12_7_observe(
    hook_name,
    module_file,
    function_name,
    instance,
    args,
    kwargs,
    result,
):
    import json as _kivo_json
    import os as _kivo_os
    import threading as _kivo_threading
    import time as _kivo_time

    if _kivo_os.getenv("KIVO_PHASE12_7_ENABLE") != "1":
        return
    output_path = _kivo_os.getenv("KIVO_PHASE12_7_OBS_PATH")
    if not output_path:
        return
    try:
        active = _kivo_os.getenv("KIVO_PHASE12_7_ACTIVE") == "1"
        observed_values = (instance, result)
        block_fields = sorted({
            name
            for value in observed_values
            for name in _kivo_phase12_7_names(value, ("block",))
        })[:48]
        slot_fields = sorted({
            name
            for value in observed_values
            for name in _kivo_phase12_7_names(value, ("slot",))
        })[:48]
        attention_fields = sorted({
            name
            for value in observed_values
            for name in _kivo_phase12_7_names(
                value, ("attn", "attention")
            )
        })[:48]
        kv_fields = sorted({
            name
            for value in observed_values
            for name in _kivo_phase12_7_names(value, ("kv", "cache"))
        })[:48]
        metadata_keys = sorted(
            set(block_fields + slot_fields + attention_fields + kv_fields)
        )
        would_select = []
        result_summary = _kivo_phase12_7_safe_summary(result)
        length = result_summary.get("length")
        if active and isinstance(length, int) and length > 0:
            budget = min(length, max(1, length // 2))
            would_select = list(range(budget))
        blocked = active
        blocker = (
            "Phase 12.7 computes a side-channel decision only; mutating "
            "runtime-consumed metadata is not proven safe."
            if active
            else None
        )
        record = {
            "schema_version": "phase12_7_runtime_observation_v1",
            "timestamp": _kivo_time.time(),
            "pid": _kivo_os.getpid(),
            "thread_id": _kivo_threading.get_ident(),
            "hook_name": hook_name,
            "module_file": module_file,
            "function_name": function_name,
            "self_type": (
                f"{type(instance).__module__}.{type(instance).__qualname__}"
            ),
            "args_summary": [
                _kivo_phase12_7_safe_summary(item) for item in args[:8]
            ],
            "kwargs_keys": sorted(str(key) for key in kwargs)[:32],
            "result_type": (
                f"{type(result).__module__}.{type(result).__qualname__}"
            ),
            "result_summary": result_summary,
            "metadata_keys_found": metadata_keys,
            "block_like_fields_found": block_fields,
            "slot_like_fields_found": slot_fields,
            "attention_like_fields_found": attention_fields,
            "kv_like_fields_found": kv_fields,
            "active_enabled": active,
            "would_select_blocks": would_select,
            "mutation_attempted": active,
            "mutation_applied": False,
            "active_experiment_blocked": blocked,
            "blocker_reason": blocker,
            "runtime_behavior_changed": False,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "caveats": [
                "installed-wheel observation wrapper",
                "side-channel decision only",
                "original result returned unchanged",
                "no KV tensor, scheduler, block-table, or attention mutation",
            ],
        }
        parent = _kivo_os.path.dirname(output_path)
        if parent:
            _kivo_os.makedirs(parent, exist_ok=True)
        encoded = (
            _kivo_json.dumps(record, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor = _kivo_os.open(
            output_path,
            _kivo_os.O_APPEND | _kivo_os.O_CREAT | _kivo_os.O_WRONLY,
            0o644,
        )
        try:
            _kivo_os.write(descriptor, encoded)
        finally:
            _kivo_os.close(descriptor)
    except Exception:
        return
# KIVO_PHASE12_7_END
'''

ACTIVE_PATCH_HELPER = r'''
# KIVO_PHASE12_8_9_BEGIN
_kivo_phase12_8_9_mutation_count = 0
_kivo_phase12_8_9_last_stage = None


def _kivo_phase12_8_9_write_record(record):
    import json as _kivo_json
    import os as _kivo_os
    import time as _kivo_time

    output_path = _kivo_os.getenv("KIVO_PHASE12_8_9_OBS_PATH")
    if not output_path:
        return
    record = {
        "schema_version": "phase12_8_9_active_ladder_v1",
        "timestamp": _kivo_time.time(),
        "measured_runtime_reduction": False,
        **record,
    }
    parent = _kivo_os.path.dirname(output_path)
    if parent:
        _kivo_os.makedirs(parent, exist_ok=True)
    encoded = (_kivo_json.dumps(record, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = _kivo_os.open(
        output_path,
        _kivo_os.O_APPEND | _kivo_os.O_CREAT | _kivo_os.O_WRONLY,
        0o644,
    )
    try:
        _kivo_os.write(descriptor, encoded)
    finally:
        _kivo_os.close(descriptor)


def _kivo_phase12_8_9_apply(hook_name, result):
    import os as _kivo_os

    global _kivo_phase12_8_9_last_stage
    global _kivo_phase12_8_9_mutation_count
    if _kivo_os.getenv("KIVO_PHASE12_8_9_ENABLE") != "1":
        return result
    stage = _kivo_os.getenv("KIVO_PHASE12_8_9_STAGE", "baseline")
    active = _kivo_os.getenv("KIVO_PHASE12_8_9_ACTIVE") == "1"
    if stage != _kivo_phase12_8_9_last_stage:
        _kivo_phase12_8_9_mutation_count = 0
        _kivo_phase12_8_9_last_stage = stage
    try:
        max_mutations = max(
            0, int(_kivo_os.getenv("KIVO_PHASE12_8_9_MAX_MUTATIONS", "1"))
        )
    except ValueError:
        max_mutations = 1
    record = {
        "hook_name": hook_name,
        "mutation_stage": stage,
        "active_enabled": active,
        "mutation_attempted": False,
        "mutation_applied": False,
        "runtime_behavior_changed": False,
        "active_routing": False,
        "removed_key": None,
        "original_layer_count": None,
        "mutated_layer_count": None,
        "selected_slot_key": None,
        "original_selected_slot_count": None,
        "mutated_selected_slot_count": None,
        "blocker_reason": None,
    }
    if stage == "baseline" or not active:
        _kivo_phase12_8_9_write_record(record)
        return result
    if _kivo_phase12_8_9_mutation_count >= max_mutations:
        record["blocker_reason"] = "maximum mutation count reached"
        _kivo_phase12_8_9_write_record(record)
        return result
    record["mutation_attempted"] = True
    if not isinstance(result, tuple) or len(result) < 2:
        record["blocker_reason"] = (
            "_get_slot_mappings result is not a tuple with two items"
        )
        _kivo_phase12_8_9_write_record(record)
        return result

    if stage == "metadata":
        metadata = result[1]
        if not isinstance(metadata, dict) or not metadata:
            record["blocker_reason"] = (
                "second _get_slot_mappings item is not a non-empty dict"
            )
            _kivo_phase12_8_9_write_record(record)
            return result
        copied = dict(metadata)
        removed_key = next(reversed(copied))
        copied.pop(removed_key)
        mutated = (result[0], copied, *result[2:])
        record.update({
            "mutation_stage": "metadata_drop_one_key",
            "mutation_applied": True,
            "runtime_behavior_changed": True,
            "removed_key": str(removed_key),
            "original_layer_count": len(metadata),
            "mutated_layer_count": len(copied),
        })
        _kivo_phase12_8_9_mutation_count += 1
        _kivo_phase12_8_9_write_record(record)
        return mutated

    if stage == "selected_slot":
        mappings = result[0]
        blocker = (
            "no safe Python-level selected-slot/block structure found in "
            "_get_slot_mappings result"
        )
        if not isinstance(mappings, dict):
            record["blocker_reason"] = blocker
            _kivo_phase12_8_9_write_record(record)
            return result
        copied = dict(mappings)
        for key, value in mappings.items():
            if type(value) not in (list, tuple) or len(value) <= 1:
                continue
            copied_value = value[:-1]
            if type(value) is list:
                copied_value = list(copied_value)
            copied[key] = copied_value
            mutated = (copied, result[1], *result[2:])
            record.update({
                "mutation_stage": "selected_slot_drop_one",
                "mutation_applied": True,
                "runtime_behavior_changed": True,
                "active_routing": True,
                "selected_slot_key": str(key),
                "original_selected_slot_count": len(value),
                "mutated_selected_slot_count": len(copied_value),
            })
            _kivo_phase12_8_9_mutation_count += 1
            _kivo_phase12_8_9_write_record(record)
            return mutated
        record["blocker_reason"] = blocker
        _kivo_phase12_8_9_write_record(record)
        return result

    record["blocker_reason"] = f"unsupported mutation stage: {stage}"
    _kivo_phase12_8_9_write_record(record)
    return result
# KIVO_PHASE12_8_9_END
'''

BLOCK_TABLE_ACTIVE_PATCH_HELPER = r'''
# KIVO_PHASE12_10_BEGIN
def _kivo_phase12_10_is_int_like(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _kivo_phase12_10_safe_summary(value, depth=0):
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
                    _kivo_phase12_10_safe_summary(item, depth + 1)
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
    except Exception as exc:
        return {"summary_error": f"{type(exc).__name__}: {exc}"}


def _kivo_phase12_10_is_tensor_like(value):
    return (
        getattr(value, "shape", None) is not None
        or getattr(value, "dtype", None) is not None
        or getattr(value, "device", None) is not None
    )


def _kivo_phase12_10_is_simple_slot_sequence(value):
    return isinstance(value, (list, tuple)) and all(
        _kivo_phase12_10_is_int_like(item) for item in value
    )


def _kivo_phase12_10_write_record(record):
    import json as _kivo_json
    import os as _kivo_os

    output_path = _kivo_os.getenv("KIVO_PHASE12_10_OBS_PATH")
    if not output_path:
        return
    parent = _kivo_os.path.dirname(output_path)
    if parent:
        _kivo_os.makedirs(parent, exist_ok=True)
    encoded = (_kivo_json.dumps(record, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = _kivo_os.open(
        output_path,
        _kivo_os.O_APPEND | _kivo_os.O_CREAT | _kivo_os.O_WRONLY,
        0o644,
    )
    try:
        _kivo_os.write(descriptor, encoded)
    finally:
        _kivo_os.close(descriptor)


def _kivo_phase12_10_apply(instance, hook_name, module_file, function_name,
                           args, kwargs, result):
    import os as _kivo_os
    import threading as _kivo_threading
    import time as _kivo_time

    if _kivo_os.getenv("KIVO_PHASE12_10_ENABLE") != "1":
        return result
    active = _kivo_os.getenv("KIVO_PHASE12_10_ACTIVE") == "1"
    result_summary = _kivo_phase12_10_safe_summary(result)
    self_attrs = sorted(str(key) for key in getattr(instance, "__dict__", {}))
    result_type = type(result)
    type_name = f"{result_type.__module__}.{result_type.__qualname__}"
    slot_like = (
        _kivo_phase12_10_is_simple_slot_sequence(result)
        or "slot" in type_name.lower()
    )
    block_like = "block" in type_name.lower()
    tensor_like = _kivo_phase12_10_is_tensor_like(result)
    python_mutable = isinstance(result, (list, tuple))
    record = {
        "schema_version": "phase12_10_block_table_slot_mapping_v1",
        "timestamp": _kivo_time.time(),
        "pid": _kivo_os.getpid(),
        "thread_id": _kivo_threading.get_ident(),
        "hook_name": hook_name,
        "module_file": module_file,
        "class_name": type(instance).__qualname__,
        "function_name": function_name,
        "self_type": (
            f"{type(instance).__module__}.{type(instance).__qualname__}"
        ),
        "self_attrs_summary": self_attrs[:48],
        "args_summary": [
            _kivo_phase12_10_safe_summary(item) for item in args[:8]
        ],
        "kwargs_keys": sorted(str(key) for key in kwargs)[:32],
        "result_type": type_name,
        "result_summary": result_summary,
        "slot_like_result_found": slot_like,
        "block_like_result_found": block_like,
        "tensor_like_result_found": tensor_like,
        "python_mutable_result_found": python_mutable,
        "mutation_attempted": active,
        "mutation_applied": False,
        "mutation_policy": None,
        "blocker_reason": None,
        "runtime_behavior_changed": False,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "caveats": [
            "installed-wheel observation wrapper",
            "never mutates tensors in place",
            "never mutates scheduler state or attention kernels",
            "returns original result on unsafe structures",
        ],
    }
    if not active:
        _kivo_phase12_10_write_record(record)
        return result
    if tensor_like:
        record["blocker_reason"] = (
            "tensor-like slot mapping requires tensor-safe mutation design"
        )
        _kivo_phase12_10_write_record(record)
        return result
    if result is None:
        record["blocker_reason"] = "no safe Python-level slot mapping result found"
        _kivo_phase12_10_write_record(record)
        return result
    if _kivo_phase12_10_is_simple_slot_sequence(result) and len(result) > 0:
        mutated = result[:-1]
        if isinstance(result, list):
            mutated = list(mutated)
        record.update({
            "mutation_applied": True,
            "mutation_policy": "drop_last_python_slot_entry",
            "runtime_behavior_changed": True,
            "active_routing": True,
        })
        _kivo_phase12_10_write_record(record)
        return mutated
    record["blocker_reason"] = "no safe Python-level slot mapping result found"
    _kivo_phase12_10_write_record(record)
    return result
# KIVO_PHASE12_10_END
'''


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage a reversible installed-wheel vLLM runtime patch."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--install-patch", action="store_true")
    action.add_argument("--restore", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument(
        "--target",
        choices=("auto", *(target.name for target in TARGETS)),
        default="auto",
    )
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/phase12_7_patch_status.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/phase12_7_patch_status.md",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_installed_wheel_path(package_root: str | Path) -> Path:
    root = Path(package_root).resolve()
    text = str(root)
    if "site-packages" not in text and "dist-packages" not in text:
        raise ValueError(
            f"refusing non-installed vLLM package path: {root}"
        )
    return root


def locate_installed_vllm() -> tuple[Path, dict[str, Any]]:
    vllm = importlib.import_module("vllm")
    init_path = Path(str(getattr(vllm, "__file__", ""))).resolve()
    root = assert_installed_wheel_path(init_path.parent)
    return root, {
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_file": str(init_path),
        "package_root": str(root),
    }


def _find_method(
    source: str,
    class_name: str,
    method_name: str,
) -> tuple[ast.ClassDef, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    if child.name == method_name:
                        return node, child
    raise ValueError(f"{class_name}.{method_name} was not found")


def target_available(package_root: Path, target: PatchTarget) -> bool:
    target_path = package_root / target.relative_path
    if not target_path.exists():
        return False
    try:
        _find_method(
            target_path.read_text(encoding="utf-8"),
            target.class_name,
            target.method_name,
        )
    except (SyntaxError, ValueError):
        return False
    return True


def choose_target(package_root: Path, requested: str) -> PatchTarget:
    candidates = (
        TARGETS
        if requested == "auto"
        else tuple(target for target in TARGETS if target.name == requested)
    )
    for target in candidates:
        if target_available(package_root, target):
            return target
    raise ValueError(f"no available patch target for {requested}")


def _wrapper_source(
    target: PatchTarget,
    indent: str,
    original_name: str,
) -> list[str]:
    method = target.method_name
    if target.name == "slot_mappings_active_ladder":
        return [
            f"{indent}def {method}(self, *args, **kwargs):\n",
            f"{indent}    result = self.{original_name}(*args, **kwargs)\n",
            f"{indent}    try:\n",
            f"{indent}        return _kivo_phase12_8_9_apply(\n",
            f"{indent}            {target.name!r}, result\n",
            f"{indent}        )\n",
            f"{indent}    except Exception:\n",
            f"{indent}        return result\n",
            "\n",
        ]
    if target.name == "block_table_compute_slot_mapping_active":
        return [
            f"{indent}def {method}(self, *args, **kwargs):\n",
            f"{indent}    result = self.{original_name}(*args, **kwargs)\n",
            f"{indent}    try:\n",
            f"{indent}        return _kivo_phase12_10_apply(\n",
            f"{indent}            self,\n",
            f"{indent}            {target.name!r},\n",
            f"{indent}            __file__,\n",
            f"{indent}            {method!r},\n",
            f"{indent}            args,\n",
            f"{indent}            kwargs,\n",
            f"{indent}            result,\n",
            f"{indent}        )\n",
            f"{indent}    except Exception:\n",
            f"{indent}        return result\n",
            "\n",
        ]
    return [
        f"{indent}def {method}(self, *args, **kwargs):\n",
        f"{indent}    result = self.{original_name}(*args, **kwargs)\n",
        f"{indent}    try:\n",
        f"{indent}        _kivo_phase12_7_observe(\n",
        f"{indent}            {target.name!r},\n",
        f"{indent}            __file__,\n",
        f"{indent}            {method!r},\n",
        f"{indent}            self,\n",
        f"{indent}            args,\n",
        f"{indent}            kwargs,\n",
        f"{indent}            result,\n",
        f"{indent}        )\n",
        f"{indent}    except Exception:\n",
        f"{indent}        pass\n",
        f"{indent}    return result\n",
        "\n",
    ]


def build_patched_source(source: str, target: PatchTarget) -> str:
    markers = (
        BEGIN_MARKER,
        END_MARKER,
        ACTIVE_BEGIN_MARKER,
        ACTIVE_END_MARKER,
        BLOCK_TABLE_ACTIVE_BEGIN_MARKER,
        BLOCK_TABLE_ACTIVE_END_MARKER,
    )
    if any(marker in source for marker in markers):
        raise ValueError("target file is already patched")
    _, method = _find_method(source, target.class_name, target.method_name)
    lines = source.splitlines(keepends=True)
    def_index = method.lineno - 1
    def_line = lines[def_index]
    indent = def_line[: len(def_line) - len(def_line.lstrip())]
    phase = (
        "phase12_8_9"
        if target.name == "slot_mappings_active_ladder"
        else (
            "phase12_10"
            if target.name == "block_table_compute_slot_mapping_active"
            else "phase12_7"
        )
    )
    original_name = f"_kivo_{phase}_original_{target.method_name}"
    if re.search(rf"\bdef\s+{re.escape(original_name)}\s*\(", source):
        raise ValueError(f"original method alias already exists: {original_name}")
    renamed = re.sub(
        rf"(\bdef\s+){re.escape(target.method_name)}(\s*\()",
        rf"\1{original_name}\2",
        def_line,
        count=1,
    )
    if renamed == def_line:
        raise ValueError("failed to rename target method")
    lines[def_index] = renamed
    insertion_index = min(
        [method.lineno, *(item.lineno for item in method.decorator_list)]
    ) - 1
    lines[insertion_index:insertion_index] = _wrapper_source(
        target,
        indent,
        original_name,
    )
    patched = "".join(lines)
    if not patched.endswith("\n"):
        patched += "\n"
    helper = (
        ACTIVE_PATCH_HELPER
        if target.name == "slot_mappings_active_ladder"
        else (
            BLOCK_TABLE_ACTIVE_PATCH_HELPER
            if target.name == "block_table_compute_slot_mapping_active"
            else PATCH_HELPER
        )
    )
    patched += helper.lstrip("\n")
    ast.parse(patched)
    return patched


def _derive_backup_dir(
    explicit: str | None,
    output_json: str,
) -> Path:
    if explicit:
        return Path(explicit).resolve()
    output = Path(output_json)
    parts = output.parts
    if output.is_absolute() and "outputs" in parts:
        index = parts.index("outputs")
        root = Path(*parts[:index])
        return (
            root / "outputs" / "kivo_vd" / "phase12_7_backups"
        ).resolve()
    return Path("outputs/kivo_vd/phase12_7_backups").resolve()


def _manifest_path(backup_dir: Path) -> Path:
    return backup_dir / MANIFEST_NAME


def _load_manifest(backup_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(backup_dir)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("patch manifest must be a JSON object")
    return value


def install_patch(
    package_root: Path,
    target: PatchTarget,
    backup_dir: Path,
) -> dict[str, Any]:
    target_path = (package_root / target.relative_path).resolve()
    original = target_path.read_bytes()
    source = original.decode("utf-8")
    patched = build_patched_source(source, target).encode("utf-8")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / (
        f"{target.name}_{_sha256(str(target_path).encode())[:12]}.py.bak"
    )
    if backup_path.exists() and backup_path.read_bytes() != original:
        raise ValueError(f"backup already exists with different content: {backup_path}")
    if not backup_path.exists():
        backup_path.write_bytes(original)
    markers = (
        [ACTIVE_BEGIN_MARKER, ACTIVE_END_MARKER]
        if target.name == "slot_mappings_active_ladder"
        else (
            [BLOCK_TABLE_ACTIVE_BEGIN_MARKER, BLOCK_TABLE_ACTIVE_END_MARKER]
            if target.name == "block_table_compute_slot_mapping_active"
            else [BEGIN_MARKER, END_MARKER]
        )
    )
    manifest = {
        "target": asdict(target),
        "target_path": str(target_path),
        "backup_path": str(backup_path),
        "original_sha256": _sha256(original),
        "patched_sha256": _sha256(patched),
        "markers": markers,
    }
    _manifest_path(backup_dir).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path = target_path.with_name(
        f".{target_path.name}.kivo_phase12_7.tmp"
    )
    temporary_path.write_bytes(patched)
    temporary_path.replace(target_path)
    return manifest


def restore_patch(backup_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(backup_dir)
    if manifest is None:
        raise FileNotFoundError("Phase 12.7 patch manifest is missing")
    target_path = Path(manifest["target_path"])
    backup_path = Path(manifest["backup_path"])
    backup = backup_path.read_bytes()
    if _sha256(backup) != manifest["original_sha256"]:
        raise ValueError("backup checksum does not match manifest")
    temporary_path = target_path.with_name(
        f".{target_path.name}.kivo_phase12_7_restore.tmp"
    )
    temporary_path.write_bytes(backup)
    temporary_path.replace(target_path)
    return {
        **manifest,
        "restored_sha256": _sha256(target_path.read_bytes()),
        "restored_exactly": target_path.read_bytes() == backup,
    }


def patch_status(
    package_root: Path,
    backup_dir: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(backup_dir)
    available = [
        {
            **asdict(target),
            "available": target_available(package_root, target),
        }
        for target in TARGETS
    ]
    if manifest is None:
        return {
            "patched": False,
            "manifest_present": False,
            "available_targets": available,
        }
    target_path = Path(manifest["target_path"])
    content = target_path.read_text(encoding="utf-8")
    markers = manifest.get("markers", [BEGIN_MARKER, END_MARKER])
    return {
        "patched": all(marker in content for marker in markers),
        "manifest_present": True,
        "manifest": manifest,
        "current_sha256": _sha256(target_path.read_bytes()),
        "available_targets": available,
    }


def render_markdown(report: dict[str, Any]) -> str:
    operation = report["operation"]
    target = report.get("target")
    active_ladder = bool(
        target and target.get("name") == "slot_mappings_active_ladder"
    )
    block_table_active = bool(
        target
        and target.get("name") == "block_table_compute_slot_mapping_active"
    )
    lines = [
        "# Kivo-VD Phase 12.7 Installed vLLM Patch",
        "",
        f"- Operation: `{operation}`",
        f"- Status: `{report['status']}`",
        f"- vLLM file: `{report.get('vllm_file')}`",
        f"- Package root: `{report.get('package_root')}`",
        f"- Backup directory: `{report.get('backup_dir')}`",
        f"- Patch installed: `{str(report.get('patched', False)).lower()}`",
        "",
        "## Safety Boundary",
        "",
        "- The target is an installed wheel, never repository-local `vllm/`.",
        "- Original bytes are backed up before editing.",
        "- Injected code is disabled unless its phase-specific env flag is set.",
        "- Exceptions fail closed by returning the original method result.",
        (
            "- The active-ladder target mutates shallow copies only."
            if active_ladder
            else (
                "- The block-table target mutates copied Python slot sequences "
                "only."
                if block_table_active
                else (
                    "- Active mode computes side-channel "
                    "`would_select_blocks` only."
                )
            )
        ),
        (
            "- It never mutates tensors in place or changes kernels."
            if active_ladder or block_table_active
            else (
                "- KV tensors, scheduler state, block tables, slots, and "
                "attention are unchanged."
            )
        ),
        "- No measured memory, latency, quality, or active-routing claim is made.",
    ]
    if target:
        lines.extend([
            "",
            "## Target",
            "",
            f"- Name: `{target['name']}`",
            f"- File: `{target['relative_path']}`",
            f"- Method: `{target['class_name']}.{target['method_name']}`",
            f"- Risk: `{target['risk']}`",
            f"- Reason: {target['reason']}",
        ])
    if report.get("error"):
        lines.extend(["", "## Error", "", f"`{report['error']}`"])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    operation = (
        "install" if args.install_patch else "restore" if args.restore else "status"
    )
    backup_dir = _derive_backup_dir(args.backup_dir, args.output_json)
    try:
        package_root, environment = locate_installed_vllm()
        if args.install_patch:
            target = choose_target(package_root, args.target)
            detail = install_patch(package_root, target, backup_dir)
            status = patch_status(package_root, backup_dir)
        elif args.restore:
            detail = restore_patch(backup_dir)
            status = patch_status(package_root, backup_dir)
            target = PatchTarget(**detail["target"])
        else:
            detail = {}
            status = patch_status(package_root, backup_dir)
            target = (
                PatchTarget(**status["manifest"]["target"])
                if status.get("manifest")
                else None
            )
        report = {
            "operation": operation,
            "status": "succeeded",
            **environment,
            "backup_dir": str(backup_dir),
            "target": asdict(target) if target else None,
            "detail": detail,
            **status,
            "runtime_behavior_changed": False,
            "active_routing": False,
            "measured_runtime_reduction": False,
        }
        exit_code = 0
    except Exception as exc:
        report = {
            "operation": operation,
            "status": "failed",
            "backup_dir": str(backup_dir),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "patched": False,
            "runtime_behavior_changed": False,
            "active_routing": False,
            "measured_runtime_reduction": False,
        }
        exit_code = 0 if args.continue_on_error else 1
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "operation": operation,
        "status": report["status"],
        "patched": report.get("patched", False),
        "target": report.get("target", {}).get("name")
        if report.get("target")
        else None,
        "backup_dir": str(backup_dir),
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
