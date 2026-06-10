# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_script(filename: str, module_name: str):
    path = _repo_root() / "scripts" / "kivo_vd" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fake_module(name: str, *, file: str | None = None, version=None):
    module = ModuleType(name)
    if file is not None:
        module.__file__ = file
    if version is not None:
        module.__version__ = version
    return module


def test_repo_local_path_helper_detects_workspace():
    helper = _load_script(
        "check_source_vllm_runtime.py",
        "source_s2_runtime_check_test_1",
    )
    repo_root = _repo_root()

    assert helper._is_repo_local_path(str(repo_root / "vllm/__init__.py"), repo_root)
    assert not helper._is_repo_local_path("/usr/local/lib/python3.12/site.py", repo_root)


def test_collect_runtime_info_uses_injected_importer_and_source():
    helper = _load_script(
        "check_source_vllm_runtime.py",
        "source_s2_runtime_check_test_2",
    )
    repo_root = _repo_root()

    torch_module = _fake_module(
        "torch",
        file="/usr/local/lib/python3.12/site-packages/torch/__init__.py",
        version="2.11.0+cu130",
    )
    torch_module.version = SimpleNamespace(cuda="13.0")
    torch_module.cuda = SimpleNamespace(is_available=lambda: True)

    vllm_module = _fake_module(
        "vllm",
        file=str(repo_root / "vllm/__init__.py"),
        version="0.22.1",
    )
    helper_module = _fake_module(
        "vllm.v1.worker.kivo_selected_blocks",
        file=str(repo_root / "vllm/v1/worker/kivo_selected_blocks.py"),
    )
    block_table_module = _fake_module(
        "vllm.v1.worker.block_table",
        file=str(repo_root / "vllm/v1/worker/block_table.py"),
    )
    block_table_module.BlockTable = type(
        "BlockTable",
        (),
        {"compute_slot_mapping": lambda self: None},
    )
    ext_c_module = _fake_module("vllm._C", file="/tmp/vllm/_C.so")
    ext_libtorch_module = _fake_module(
        "vllm._C_stable_libtorch",
        file="/tmp/vllm/_C_stable_libtorch.so",
    )
    ext_flash_attn_module = _fake_module(
        "vllm.vllm_flash_attn",
        file="/tmp/vllm/vllm_flash_attn.so",
    )

    modules = {
        "torch": torch_module,
        "vllm": vllm_module,
        "vllm.v1.worker.kivo_selected_blocks": helper_module,
        "vllm.v1.worker.block_table": block_table_module,
        "vllm._C": ext_c_module,
        "vllm._C_stable_libtorch": ext_libtorch_module,
        "vllm.vllm_flash_attn": ext_flash_attn_module,
    }

    def importer(name: str):
        return modules[name]

    def source_getter(_obj):
        return (
            "def compute_slot_mapping(...):\n"
            "    maybe_observe_compute_slot_mapping(...)\n"
            "    import kivo_selected_blocks\n"
        )

    info = helper.collect_runtime_info(
        importer=importer,
        source_getter=source_getter,
    )

    assert info["vllm"]["import_mode"] == "repo_local_source"
    assert info["torch"]["cuda_available"] is True
    assert info["compiled_extensions"]["vllm._C"]["available"] is True
    assert info["kivo_helper"]["available"] is True
    assert info["block_table_hook"]["source_contains_kivo_hook"] is True
    assert info["block_table_hook"]["source_contains_helper_import"] is True
