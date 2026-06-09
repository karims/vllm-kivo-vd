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


def _fake_import_with_vllm_path(vllm_path: str):
    def fake_import(module_name: str):
        if module_name == "vllm":
            return SimpleNamespace(
                __version__="0.test",
                __file__=vllm_path,
            )
        return _fake_import(module_name)

    return fake_import


def _fake_sanitize(repo_root: Path):
    return {
        "prefer_installed_vllm": True,
        "sys_path_sanitized": True,
        "removed_paths": [str(repo_root)],
        "kept_paths_preview": ["/fake/site-packages"],
    }


def test_environment_report_handles_missing_vllm() -> None:
    module = _load_module()

    report = module.collect_environment_report(_missing_vllm_import)

    assert report["torch"]["ok"] is True
    assert report["vllm"]["ok"] is False
    assert report["vllm"]["error_type"] == "ModuleNotFoundError"
    assert report["extensions"]["vllm._C"]["ok"] is False


def test_prefer_installed_vllm_flag_is_parsed(tmp_path: Path) -> None:
    module = _load_module()

    args = _args(module, tmp_path, "--prefer-installed-vllm")

    assert args.prefer_installed_vllm is True


def test_sanitizer_removes_repo_root_and_preserves_scripts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    scripts_path = repo_root / "scripts"
    kivo_scripts_path = scripts_path / "kivo_vd"
    site_packages = tmp_path / "site-packages"
    for path in (
        repo_root,
        scripts_path,
        kivo_scripts_path,
        site_packages,
    ):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sys,
        "path",
        [
            str(repo_root),
            str(scripts_path),
            str(kivo_scripts_path),
            str(site_packages),
        ],
    )

    report = module.sanitize_sys_path_for_installed_vllm(repo_root)

    assert report["removed_paths"] == [str(repo_root)]
    assert str(repo_root) not in sys.path
    assert str(scripts_path) in sys.path
    assert str(kivo_scripts_path) in sys.path
    assert str(site_packages) in sys.path


def test_sanitizer_removes_empty_entry_when_cwd_is_repo(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    scripts_path = repo_root / "scripts"
    scripts_path.mkdir(parents=True)
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(sys, "path", ["", str(scripts_path)])

    report = module.sanitize_sys_path_for_installed_vllm(repo_root)

    assert report["removed_paths"] == [""]
    assert "" not in sys.path
    assert str(scripts_path) in sys.path


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


def test_preferred_repo_local_vllm_is_not_ready(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(
        module,
        tmp_path,
        "--prefer-installed-vllm",
        "--enable-shadow",
    )
    repo_vllm = str(module.REPO_ROOT / "vllm" / "__init__.py")

    report = module.build_report(
        args,
        import_module=_fake_import_with_vllm_path(repo_vllm),
        generation_fn=_successful_generation,
        sanitize_fn=_fake_sanitize,
    )

    assert report["environment"]["vllm_source_is_repo_local"] is True
    assert report["environment"]["vllm_source_is_site_packages"] is False
    assert report["readiness"]["environment_ready"] is False
    assert report["readiness"]["phase12_6_runtime_hook_ready"] is False
    assert "Installed vLLM was requested" in report["readiness"][
        "recommendation"
    ]


def test_preferred_site_packages_vllm_is_ready(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(
        module,
        tmp_path,
        "--prefer-installed-vllm",
        "--enable-shadow",
    )
    wheel_vllm = (
        "/usr/local/lib/python3.12/site-packages/vllm/__init__.py"
    )

    report = module.build_report(
        args,
        import_module=_fake_import_with_vllm_path(wheel_vllm),
        generation_fn=_successful_generation,
        sanitize_fn=_fake_sanitize,
    )

    environment = report["environment"]
    assert environment["prefer_installed_vllm"] is True
    assert environment["sys_path_sanitized"] is True
    assert environment["vllm_import_source"] == wheel_vllm
    assert environment["vllm_source_is_repo_local"] is False
    assert environment["vllm_source_is_site_packages"] is True
    assert report["readiness"]["environment_ready"] is True
    assert report["readiness"]["phase12_6_runtime_hook_ready"] is True


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
