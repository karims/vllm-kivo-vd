# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "setup_runtime_source_overlay.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "setup_runtime_source_overlay", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_vllm_tree(tmp_path: Path) -> tuple[Path, Path]:
    installed = tmp_path / "site-packages" / "vllm"
    repo = tmp_path / "repo"
    repo_vllm = repo / "vllm"
    flash_dir = installed / "vllm_flash_attn" / "ops"
    installed.mkdir(parents=True)
    repo_vllm.mkdir(parents=True)
    flash_dir.mkdir(parents=True)

    (installed / "_C.abi3.so").write_text("fake extension")
    (installed / "_version.py").write_text("__version__ = '0.22.0'\n")
    (flash_dir / "flash_attn.so").write_text("fake flash extension")
    return installed, repo


def test_runtime_source_overlay_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "setup_runtime_source_overlay.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--repo-root" in proc.stdout
    assert "--installed-vllm-path" in proc.stdout
    assert "--replace-existing" in proc.stdout


def test_link_runtime_artifacts_with_fake_paths(tmp_path: Path) -> None:
    m = _load_module()
    installed, repo = _make_fake_vllm_tree(tmp_path)

    summary = m.link_runtime_artifacts(
        installed_vllm_path=installed,
        repo_root=repo,
    )

    repo_vllm = repo / "vllm"
    assert summary["num_top_level_extensions_linked"] == 1
    assert summary["num_flash_attn_extensions_linked"] == 1
    assert summary["linked_version_file"] is True
    assert (repo_vllm / "_C.abi3.so").is_symlink()
    assert (repo_vllm / "_version.py").is_symlink()
    assert (repo_vllm / "vllm_flash_attn" / "ops" / "flash_attn.so").is_symlink()


def test_link_runtime_artifacts_is_idempotent(tmp_path: Path) -> None:
    m = _load_module()
    installed, repo = _make_fake_vllm_tree(tmp_path)

    first = m.link_runtime_artifacts(installed_vllm_path=installed, repo_root=repo)
    second = m.link_runtime_artifacts(installed_vllm_path=installed, repo_root=repo)

    assert first["num_top_level_extensions_changed"] == 1
    assert first["num_flash_attn_extensions_changed"] == 1
    assert second["num_top_level_extensions_changed"] == 0
    assert second["num_flash_attn_extensions_changed"] == 0
    assert second["linked_version_file"] is True


def test_runtime_source_overlay_cli_with_fake_paths(tmp_path: Path) -> None:
    installed, repo = _make_fake_vllm_tree(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "setup_runtime_source_overlay.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo),
            "--installed-vllm-path",
            str(installed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["num_top_level_extensions_linked"] == 1
    assert payload["num_flash_attn_extensions_linked"] == 1
    assert payload["linked_version_file"] is True
