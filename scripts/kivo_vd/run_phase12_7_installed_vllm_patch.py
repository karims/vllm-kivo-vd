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
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BEGIN_MARKER = "# KIVO_PHASE12_7_BEGIN"
END_MARKER = "# KIVO_PHASE12_7_END"
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
        "compute_slot_mapping",
        "v1/worker/block_table.py",
        "BlockTable",
        "compute_slot_mapping",
        "high",
        "Called in the worker path but returns no metadata object.",
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
    if BEGIN_MARKER in source or END_MARKER in source:
        raise ValueError("target file is already patched")
    _, method = _find_method(source, target.class_name, target.method_name)
    lines = source.splitlines(keepends=True)
    def_index = method.lineno - 1
    def_line = lines[def_index]
    indent = def_line[: len(def_line) - len(def_line.lstrip())]
    original_name = f"_kivo_phase12_7_original_{target.method_name}"
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
    patched += PATCH_HELPER.lstrip("\n")
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
    manifest = {
        "target": asdict(target),
        "target_path": str(target_path),
        "backup_path": str(backup_path),
        "original_sha256": _sha256(original),
        "patched_sha256": _sha256(patched),
        "markers": [BEGIN_MARKER, END_MARKER],
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
    return {
        "patched": BEGIN_MARKER in content and END_MARKER in content,
        "manifest_present": True,
        "manifest": manifest,
        "current_sha256": _sha256(target_path.read_bytes()),
        "available_targets": available,
    }


def render_markdown(report: dict[str, Any]) -> str:
    operation = report["operation"]
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
        "- Observation and active-decision code is fail-closed.",
        "- Active mode computes side-channel `would_select_blocks` only.",
        (
            "- KV tensors, scheduler state, block tables, slots, and "
            "attention are unchanged."
        ),
        "- No measured memory, latency, quality, or active-routing claim is made.",
    ]
    target = report.get("target")
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
