# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin():
    plugin_root = _repo_root() / "plugins" / "kivo_vllm_shadow_plugin"
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    module = importlib.import_module("kivo_vllm_shadow_plugin.plugin")
    return importlib.reload(module)


def _load_probe():
    return _load_module(
        _repo_root()
        / "scripts"
        / "kivo_vd"
        / "run_phase12_vllm_plugin_probe.py",
        "run_phase12_vllm_plugin_probe_test",
    )


def _args(module, tmp_path: Path, *extra: str):
    return module._parse_args([
        "--marker-path",
        str(tmp_path / "marker.json"),
        "--output-json",
        str(tmp_path / "report.json"),
        "--output-md",
        str(tmp_path / "report.md"),
        *extra,
    ])


def test_plugin_marker_writer(tmp_path: Path) -> None:
    plugin = _load_plugin()
    marker_path = tmp_path / "nested" / "marker.json"
    state = plugin.KivoShadowPluginState(
        loaded=True,
        plugin_name="kivo_shadow",
        timestamp=1.0,
        python_executable="/fake/python",
        cwd="/tmp",
        sys_path_preview=["/tmp"],
        process_id=123,
        vllm_version="0.test",
        vllm_file="/site-packages/vllm/__init__.py",
        caveats=["marker only"],
    )

    plugin.write_load_marker(marker_path, state)
    payload = json.loads(marker_path.read_text(encoding="utf-8"))

    assert payload["loaded"] is True
    assert payload["plugin_name"] == "kivo_shadow"
    assert payload["vllm_version"] == "0.test"


def test_register_writes_marker_from_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    marker_path = tmp_path / "marker.json"
    fake_vllm = SimpleNamespace(
        __version__="0.test",
        __file__="/site-packages/vllm/__init__.py",
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv("KIVO_SHADOW_PLUGIN_MARKER", str(marker_path))

    plugin.register()

    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert payload["loaded"] is True
    assert payload["plugin_name"] == "kivo_shadow"
    assert "active routing is disabled" in payload["caveats"]


def test_plugin_entry_point_metadata_exists() -> None:
    pyproject = (
        _repo_root()
        / "plugins"
        / "kivo_vllm_shadow_plugin"
        / "pyproject.toml"
    ).read_text(encoding="utf-8")

    assert '[project.entry-points."vllm.general_plugins"]' in pyproject
    assert (
        "kivo_shadow = "
        '"kivo_vllm_shadow_plugin.plugin:register"'
    ) in pyproject


def test_probe_report_handles_marker_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    probe = _load_probe()
    plugin = _load_plugin()
    args = _args(probe, tmp_path, "--skip-generation")

    def load_plugin():
        plugin.write_load_marker(
            args.marker_path,
            plugin.KivoShadowPluginState(
                loaded=True,
                plugin_name="kivo_shadow",
                timestamp=1.0,
                python_executable="/fake/python",
                cwd="/tmp",
                sys_path_preview=["/tmp"],
                process_id=123,
                vllm_version="0.22.1",
                vllm_file="/site-packages/vllm/__init__.py",
                caveats=["marker only"],
            ),
        )
        return {
            "vllm_version": "0.22.1",
            "vllm_file": "/site-packages/vllm/__init__.py",
        }

    report = probe.build_probe_report(args, load_fn=load_plugin)

    assert report["plugin_marker_written"] is True
    assert report["plugin_loaded"] is True
    assert report["generation_status"] == "skipped"
    assert report["phase12_6b_plugin_shadow_hook_candidate"] is True
    assert report["active_routing"] is False
    assert report["runtime_monkeypatch_applied"] is False
    assert os.environ["VLLM_PLUGINS"] == "kivo_shadow"


def test_probe_report_handles_marker_missing(tmp_path: Path) -> None:
    probe = _load_probe()
    args = _args(probe, tmp_path, "--skip-generation")

    report = probe.build_probe_report(
        args,
        load_fn=lambda: {
            "vllm_version": "0.22.1",
            "vllm_file": "/site-packages/vllm/__init__.py",
        },
    )

    assert report["plugin_marker_written"] is False
    assert report["plugin_loaded"] is False
    assert report["phase12_6b_plugin_shadow_hook_candidate"] is False


def test_probe_generation_status_can_succeed(tmp_path: Path) -> None:
    probe = _load_probe()
    plugin = _load_plugin()
    args = _args(probe, tmp_path)

    def load_plugin():
        plugin.write_load_marker(
            args.marker_path,
            plugin.KivoShadowPluginState(
                loaded=True,
                plugin_name="kivo_shadow",
                timestamp=1.0,
                python_executable="/fake/python",
                cwd="/tmp",
                sys_path_preview=["/tmp"],
                process_id=123,
                vllm_version="0.22.1",
                vllm_file="/site-packages/vllm/__init__.py",
                caveats=["marker only"],
            ),
        )
        return {
            "vllm_version": "0.22.1",
            "vllm_file": "/site-packages/vllm/__init__.py",
        }

    report = probe.build_probe_report(
        args,
        load_fn=load_plugin,
        generation_fn=lambda parsed: {
            "status": "succeeded",
            "output_text": " generated",
            "error_type": None,
            "error": None,
        },
    )

    assert report["generation_status"] == "succeeded"
    assert report["phase12_6b_plugin_shadow_hook_candidate"] is True
    assert report["scheduler_behavior_changed"] is False
    assert report["attention_behavior_changed"] is False
    assert report["kv_cache_mutated"] is False
    assert report["block_table_mutated"] is False


def test_load_helper_calls_general_plugin_loader() -> None:
    probe = _load_probe()
    calls: list[str] = []

    def import_module(name: str):
        calls.append(name)
        if name == "vllm":
            return SimpleNamespace(
                __version__="0.test",
                __file__="/site-packages/vllm/__init__.py",
            )
        return SimpleNamespace(load_general_plugins=lambda: calls.append("load"))

    result = probe._load_vllm_and_plugins(import_module)

    assert calls == ["vllm", "vllm.plugins", "load"]
    assert result["vllm_version"] == "0.test"
