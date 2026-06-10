#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Inspect whether a source-built vLLM runtime is visible for Phase S2."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_repo_local_path(module_file: str | None, repo_root: Path | None = None) -> bool:
    if not module_file:
        return False
    root = (repo_root or _repo_root()).resolve()
    try:
        path = Path(module_file).resolve()
    except Exception:
        return False
    return path == root or root in path.parents


def _import_module(
    module_name: str,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> tuple[ModuleType | None, Exception | None]:
    try:
        return importer(module_name), None
    except Exception as exc:  # pragma: no cover - exercised in failure modes
        return None, exc


def _module_status(
    module_name: str,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, Any]:
    module, error = _import_module(module_name, importer=importer)
    record: dict[str, Any] = {
        "module_name": module_name,
        "available": module is not None,
        "module_file": None,
        "module_version": None,
        "repo_local": None,
        "error": None,
    }
    if module is None:
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        return record
    record["module_file"] = getattr(module, "__file__", None)
    record["module_version"] = getattr(module, "__version__", None)
    record["repo_local"] = _is_repo_local_path(record["module_file"])
    return record


def _torch_status(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, Any]:
    record = _module_status("torch", importer=importer)
    if not record["available"]:
        return record
    torch_module, _ = _import_module("torch", importer=importer)
    if torch_module is None:  # pragma: no cover - defensive
        return record
    try:
        record["torch_cuda_version"] = getattr(torch_module.version, "cuda", None)
    except Exception:
        record["torch_cuda_version"] = None
    try:
        record["cuda_available"] = bool(torch_module.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive
        record["cuda_available"] = None
        record["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _inspect_block_table_hook(
    importer: Callable[[str], ModuleType] = importlib.import_module,
    source_getter: Callable[[Any], str] = inspect.getsource,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "available": False,
        "module_file": None,
        "source_contains_kivo_hook": None,
        "source_contains_helper_import": None,
        "source_excerpt": None,
        "error": None,
    }
    module, error = _import_module("vllm.v1.worker.block_table", importer=importer)
    if module is None:
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        return record
    record["available"] = True
    record["module_file"] = getattr(module, "__file__", None)
    try:
        source = source_getter(module.BlockTable.compute_slot_mapping)
        record["source_contains_kivo_hook"] = (
            "maybe_observe_compute_slot_mapping" in source
        )
        record["source_contains_helper_import"] = (
            "kivo_selected_blocks" in source
        )
        record["source_excerpt"] = "\n".join(source.splitlines()[:24])
    except Exception as exc:  # pragma: no cover - defensive
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def collect_runtime_info(
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    source_getter: Callable[[Any], str] = inspect.getsource,
) -> dict[str, Any]:
    repo_root = _repo_root()
    vllm_record = _module_status("vllm", importer=importer)
    helper_record = _module_status("vllm.v1.worker.kivo_selected_blocks", importer=importer)
    compiled_extensions = {
        name: _module_status(name, importer=importer)
        for name in (
            "vllm._C",
            "vllm._C_stable_libtorch",
            "vllm.vllm_flash_attn",
        )
    }
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "repo_root": str(repo_root),
        "torch": _torch_status(importer=importer),
        "vllm": {
            **vllm_record,
            "import_mode": (
                "repo_local_source"
                if vllm_record["repo_local"] is True
                else "site_packages"
                if vllm_record["available"]
                else "unavailable"
            ),
        },
        "compiled_extensions": compiled_extensions,
        "kivo_helper": helper_record,
        "block_table_hook": _inspect_block_table_hook(
            importer=importer,
            source_getter=source_getter,
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect source-built vLLM runtime visibility for Phase S2."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _parse_args(argv)
    info = collect_runtime_info()
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
