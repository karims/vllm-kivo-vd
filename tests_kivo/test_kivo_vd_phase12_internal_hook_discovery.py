# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_discovery():
    plugin_root = _repo_root() / "plugins" / "kivo_vllm_shadow_plugin"
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    module = importlib.import_module(
        "kivo_vllm_shadow_plugin.internal_discovery"
    )
    return importlib.reload(module)


def _load_runner():
    path = (
        _repo_root()
        / "scripts"
        / "kivo_vd"
        / "run_phase12_vllm_internal_hook_discovery.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase12_internal_hook_discovery_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_modules_are_recorded_without_failure() -> None:
    discovery = _load_discovery()
    spec = discovery.CandidateSpec(
        "vllm.missing",
        "MissingClass",
        "missing_method",
        "scheduler_step",
        "high",
        "high",
        "missing test candidate",
    )

    def missing_import(name: str):
        raise ModuleNotFoundError(name)

    report = discovery.discover_internal_hooks(
        import_module=missing_import,
        specs=(spec,),
    )

    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["callable_candidate_count"] == 0
    assert report["missing_modules"][0]["module_path"] == "vllm.missing"
    assert report["patch_installed"] is False


def test_callable_signature_and_source_are_captured() -> None:
    discovery = _load_discovery()

    class FakeMetrics:
        def observe(self, request_id: str, count: int = 0) -> None:
            """Observe copied metadata without mutation."""

    module = SimpleNamespace(FakeMetrics=FakeMetrics)
    spec = discovery.CandidateSpec(
        "vllm.fake.metrics",
        "FakeMetrics",
        "observe",
        "metrics_callback",
        "medium",
        "high",
        "fake metadata callback",
    )
    report = discovery.discover_internal_hooks(
        import_module=lambda name: module,
        specs=(spec,),
    )
    candidate = report["candidates"][0]

    assert candidate["callable"] is True
    assert "request_id" in candidate["signature"]
    assert candidate["source_file"] == __file__
    assert candidate["docstring_preview"].startswith("Observe copied")


def test_ranking_is_deterministic_and_high_risk_is_not_safe() -> None:
    discovery = _load_discovery()

    class FakeHooks:
        def metadata(self):
            return None

        def schedule(self):
            return None

    module = SimpleNamespace(FakeHooks=FakeHooks)
    specs = (
        discovery.CandidateSpec(
            "vllm.fake",
            "FakeHooks",
            "schedule",
            "scheduler_step",
            "high",
            "high",
            "scheduler",
        ),
        discovery.CandidateSpec(
            "vllm.fake",
            "FakeHooks",
            "metadata",
            "metrics_callback",
            "medium",
            "high",
            "metadata",
        ),
    )

    first = discovery.discover_internal_hooks(
        import_module=lambda name: module,
        specs=specs,
    )
    second = discovery.discover_internal_hooks(
        import_module=lambda name: module,
        specs=specs,
    )

    assert [
        item["qualified_name"] for item in first["candidates"]
    ] == [
        item["qualified_name"] for item in second["candidates"]
    ]
    assert first["candidates"][0]["method_name"] == "metadata"
    scheduler = next(
        item
        for item in first["candidates"]
        if item["method_name"] == "schedule"
    )
    assert scheduler["risk_level"] == "high"
    assert scheduler["safe_to_patch_in_phase12_6c"] is False


def test_vllm_source_path_detection() -> None:
    discovery = _load_discovery()

    installed = discovery.classify_vllm_source_path(
        "/usr/local/lib/python3.12/dist-packages/vllm/__init__.py"
    )
    local = discovery.classify_vllm_source_path(
        "/workspace/vllm-kivo-vd/vllm/__init__.py"
    )

    assert installed["installed_wheel_path"] is True
    assert installed["repo_local_source_detected"] is False
    assert local["installed_wheel_path"] is False
    assert local["repo_local_source_detected"] is True


def test_report_keeps_discovery_only_safety_flags() -> None:
    runner = _load_runner()

    def fake_discovery(**kwargs):
        return {
            "candidates": [{
                "rank": 1,
                "qualified_name": "vllm.fake.Fake.observe",
                "callable": True,
                "risk_level": "medium",
                "usefulness_level": "high",
                "category": "metrics_callback",
            }],
            "missing_modules": [],
            "summary": {
                "candidate_count": 1,
                "callable_candidate_count": 1,
                "missing_module_count": 0,
                "risk_counts": {"low": 0, "medium": 1, "high": 0},
                "usefulness_counts": {
                    "low": 0,
                    "medium": 0,
                    "high": 1,
                },
            },
            "recommendations": [],
            "active_routing": False,
            "measured_runtime_reduction": False,
            "runtime_behavior_changed": False,
            "patch_installed": False,
            "discovery_only": True,
        }

    def fake_environment():
        return {
            "python_version": "3.12.0",
            "python_executable": "/usr/bin/python",
            "platform": "test",
            "torch": {
                "version": "2.test",
                "cuda_version": "13.0",
            },
            "vllm_version": "0.22.1",
            "vllm_file": "/site-packages/vllm/__init__.py",
            "installed_wheel_path": True,
            "repo_local_source_detected": False,
            "compiled_extensions": {},
        }

    report = runner.build_report(
        discover_fn=fake_discovery,
        environment_fn=fake_environment,
    )
    markdown = runner.render_markdown(report)

    assert report["patch_installed"] is False
    assert report["runtime_behavior_changed"] is False
    assert report["active_routing"] is False
    assert report["phase12_6d_candidate_review_ready"] is True
    assert "Patch installed: `false`" in markdown
    assert "Runtime behavior changed: `false`" in markdown


def test_cli_help_includes_discovery_options() -> None:
    runner = _load_runner()
    help_text = runner._parse_args(["--include-source-previews"])

    assert help_text.include_source_previews is True
    assert help_text.max_doc_preview_chars == 240
