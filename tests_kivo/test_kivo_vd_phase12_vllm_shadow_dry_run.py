# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts" / "kivo_vd"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / "run_phase12_vllm_shadow_dry_run.py"
    spec = importlib.util.spec_from_file_location(
        "run_phase12_vllm_shadow_dry_run_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(module, tmp_path: Path, *extra: str):
    return module._parse_args([
        "--output-json",
        str(tmp_path / "report.json"),
        "--output-md",
        str(tmp_path / "report.md"),
        "--shadow-output-jsonl",
        str(tmp_path / "events.jsonl"),
        *extra,
    ])


def _fake_import(module_name: str):
    if module_name == "torch":
        return SimpleNamespace(
            __version__="2.test",
            __file__="/fake/torch.py",
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda index: "Fake GPU",
            ),
        )
    if module_name == "vllm":
        return SimpleNamespace(
            __version__="0.test",
            __file__="/fake/vllm/__init__.py",
        )
    return SimpleNamespace(__file__=f"/fake/{module_name}.so")


def _missing_vllm_import(module_name: str):
    if module_name.startswith("vllm"):
        raise ModuleNotFoundError("vllm is unavailable")
    return _fake_import(module_name)


def _successful_generation(args):
    return {
        "status": "succeeded",
        "output_text": " generated",
        "prompt_token_length": 33,
        "error_type": None,
        "error": None,
    }


def test_environment_report_handles_missing_vllm() -> None:
    module = _load_module()

    report = module.collect_environment_report(_missing_vllm_import)

    assert report["torch"]["ok"] is True
    assert report["vllm"]["ok"] is False
    assert report["vllm"]["error_type"] == "ModuleNotFoundError"
    assert report["extensions"]["vllm._C"]["ok"] is False


def test_shadow_disabled_creates_no_events(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(
        module,
        tmp_path,
        "--skip-vllm-generation",
        "--continue-on-error",
    )

    report = module.build_report(
        args,
        import_module=_missing_vllm_import,
    )

    assert report["generation"]["status"] == "skipped"
    assert report["shadow"]["status"] == "disabled"
    assert (tmp_path / "events.jsonl").exists() is False


def test_shadow_enabled_emits_validator_compatible_events(
    tmp_path: Path,
) -> None:
    module = _load_module()
    args = _args(module, tmp_path, "--enable-shadow")

    report = module.build_report(
        args,
        import_module=_fake_import,
        generation_fn=_successful_generation,
    )

    assert report["shadow"]["status"] == "succeeded"
    assert report["shadow"]["events_written"] == 4
    assert report["shadow"]["validation"]["validation_passed"] is True
    assert report["readiness"]["phase12_6_runtime_hook_ready"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert all(event["shadow_only"] is True for event in events)
    assert all(event["active_routing"] is False for event in events)


def test_report_contains_required_caveats(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(
        module,
        tmp_path,
        "--skip-vllm-generation",
        "--continue-on-error",
    )

    report = module.build_report(
        args,
        import_module=_missing_vllm_import,
    )

    assert report["dry_run_only"] is True
    assert report["shadow_only"] is True
    assert report["active_routing"] is False
    assert report["measured_runtime_reduction"] is False
    assert report["no_attention_kernel_change"] is True
    assert report["no_kv_cache_mutation"] is True
    assert report["no_scheduler_change"] is True


def test_generation_failure_is_fail_closed_with_continue(
    tmp_path: Path,
) -> None:
    module = _load_module()
    args = _args(
        module,
        tmp_path,
        "--enable-shadow",
        "--continue-on-error",
    )

    def fail_generation(args):
        raise RuntimeError("synthetic generation failure")

    report = module.build_report(
        args,
        import_module=_fake_import,
        generation_fn=fail_generation,
    )

    assert report["generation"]["status"] == "failed"
    assert report["generation"]["error"] == "synthetic generation failure"
    assert report["shadow"]["status"] == "succeeded"
    assert report["readiness"]["phase12_6_runtime_hook_ready"] is False


def test_generation_failure_raises_without_continue(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(module, tmp_path)

    def fail_generation(args):
        raise RuntimeError("synthetic generation failure")

    try:
        module.build_report(
            args,
            import_module=_fake_import,
            generation_fn=fail_generation,
        )
    except RuntimeError as exc:
        assert str(exc) == "synthetic generation failure"
    else:
        raise AssertionError("generation failure was unexpectedly swallowed")
