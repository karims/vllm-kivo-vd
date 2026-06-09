# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def _load_patcher():
    return _load_script(
        "run_phase12_7_installed_vllm_patch.py",
        "phase12_7_runtime_patcher_test",
    )


def _load_validator():
    return _load_script(
        "validate_phase12_7_runtime_observation.py",
        "phase12_7_runtime_validator_test",
    )


def _load_probe():
    return _load_script(
        "run_phase12_7_runtime_generation_probe.py",
        "phase12_7_runtime_generation_probe_test",
    )


def _fake_package(tmp_path: Path) -> tuple[Path, object, Path]:
    patcher = _load_patcher()
    root = tmp_path / "site-packages" / "vllm"
    target = patcher.TARGETS[0]
    path = root / target.relative_path
    path.parent.mkdir(parents=True)
    path.write_text(
        "class GPUModelRunner:\n"
        "    def _get_slot_mappings(self, value, flag=False):\n"
        "        if hasattr(self, 'result'):\n"
        "            return self.result\n"
        "        return ({'slot_mapping': value}, flag)\n",
        encoding="utf-8",
    )
    return root, target, path


def _valid_observation(active: bool = False) -> dict:
    return {
        "schema_version": "phase12_7_runtime_observation_v1",
        "timestamp": 1.0,
        "pid": 123,
        "hook_name": "slot_mappings",
        "module_file": "/site-packages/vllm/gpu_model_runner.py",
        "function_name": "_get_slot_mappings",
        "self_type": "fake.GPUModelRunner",
        "args_summary": [],
        "kwargs_keys": [],
        "result_type": "builtins.tuple",
        "result_summary": {"type": "builtins.tuple", "length": 2},
        "metadata_keys_found": ["slot_mapping"],
        "block_like_fields_found": [],
        "slot_like_fields_found": ["slot_mapping"],
        "attention_like_fields_found": [],
        "kv_like_fields_found": [],
        "active_enabled": active,
        "would_select_blocks": [0] if active else [],
        "mutation_attempted": active,
        "mutation_applied": False,
        "active_experiment_blocked": active,
        "blocker_reason": "side-channel only" if active else None,
        "runtime_behavior_changed": False,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "caveats": ["original result returned unchanged"],
    }


def test_refuses_repo_local_vllm_path(tmp_path: Path) -> None:
    patcher = _load_patcher()

    with pytest.raises(ValueError, match="refusing non-installed"):
        patcher.assert_installed_wheel_path(tmp_path / "repo" / "vllm")


def test_install_backs_up_and_restore_is_exact(tmp_path: Path) -> None:
    patcher = _load_patcher()
    package_root, target, target_path = _fake_package(tmp_path)
    backup_dir = tmp_path / "backups"
    original = target_path.read_bytes()

    manifest = patcher.install_patch(package_root, target, backup_dir)
    patched = target_path.read_text(encoding="utf-8")

    assert Path(manifest["backup_path"]).read_bytes() == original
    assert patcher.BEGIN_MARKER in patched
    assert patcher.END_MARKER in patched
    assert "_kivo_phase12_7_original__get_slot_mappings" in patched
    compile(patched, str(target_path), "exec")

    restored = patcher.restore_patch(backup_dir)
    assert restored["restored_exactly"] is True
    assert target_path.read_bytes() == original


def test_status_detects_patched_and_restored(tmp_path: Path) -> None:
    patcher = _load_patcher()
    package_root, target, _ = _fake_package(tmp_path)
    backup_dir = tmp_path / "backups"

    before = patcher.patch_status(package_root, backup_dir)
    patcher.install_patch(package_root, target, backup_dir)
    during = patcher.patch_status(package_root, backup_dir)
    patcher.restore_patch(backup_dir)
    after = patcher.patch_status(package_root, backup_dir)

    assert before["patched"] is False
    assert during["patched"] is True
    assert after["patched"] is False


def test_generated_patch_contains_required_guards(tmp_path: Path) -> None:
    patcher = _load_patcher()
    _, target, target_path = _fake_package(tmp_path)

    patched = patcher.build_patched_source(
        target_path.read_text(encoding="utf-8"),
        target,
    )

    assert "KIVO_PHASE12_7_ENABLE" in patched
    assert "KIVO_PHASE12_7_OBS_PATH" in patched
    assert "KIVO_PHASE12_7_ACTIVE" in patched
    assert '"mutation_applied": False' in patched
    assert '"active_routing": False' in patched


def test_generated_wrapper_returns_exact_result_and_writes_record(
    monkeypatch,
    tmp_path: Path,
) -> None:
    patcher = _load_patcher()
    _, target, target_path = _fake_package(tmp_path)
    patched = patcher.build_patched_source(
        target_path.read_text(encoding="utf-8"),
        target,
    )
    namespace = {"__file__": str(target_path)}
    exec(compile(patched, str(target_path), "exec"), namespace)
    result = ({"slot_mapping": [1, 2]}, False)
    observations = tmp_path / "observations.jsonl"
    monkeypatch.setenv("KIVO_PHASE12_7_ENABLE", "1")
    monkeypatch.setenv("KIVO_PHASE12_7_OBS_PATH", str(observations))

    runner = namespace["GPUModelRunner"]()
    runner.result = result
    returned = runner._get_slot_mappings([1, 2], flag=False)
    record = json.loads(observations.read_text(encoding="utf-8"))

    assert returned is result
    assert record["hook_name"] == "slot_mappings"
    assert record["mutation_applied"] is False
    assert record["runtime_behavior_changed"] is False


def test_auto_target_prefers_slot_mappings(tmp_path: Path) -> None:
    patcher = _load_patcher()
    package_root, target, _ = _fake_package(tmp_path)

    selected = patcher.choose_target(package_root, "auto")

    assert selected.name == target.name
    assert selected.relative_path == target.relative_path


def test_validator_passes_observation_and_active_blocker() -> None:
    validator = _load_validator()

    observation_only = validator.validate_observations([
        _valid_observation(False)
    ])
    active = validator.validate_observations([_valid_observation(True)])

    assert observation_only["validation_passed"] is True
    assert active["validation_passed"] is True
    assert active["mutation_applied_records"] == 0


def test_validator_missing_file_is_clear(tmp_path: Path) -> None:
    validator = _load_validator()

    with pytest.raises(FileNotFoundError, match="input is missing"):
        validator.load_observations(tmp_path / "missing.jsonl")


def test_validator_rejects_runtime_behavior_change() -> None:
    validator = _load_validator()
    record = _valid_observation()
    record["runtime_behavior_changed"] = True

    report = validator.validate_observations([record])

    assert report["validation_passed"] is False


def test_generation_probe_summarizes_observations(tmp_path: Path) -> None:
    probe = _load_probe()
    path = tmp_path / "observations.jsonl"
    path.write_text(
        json.dumps(_valid_observation(True)) + "\n",
        encoding="utf-8",
    )

    records = probe.load_observations(path)
    summary = probe.summarize_records(records)

    assert summary["observations_written"] == 1
    assert summary["mutation_attempted"] is True
    assert summary["mutation_applied"] is False
    assert summary["active_experiment_blocked"] is True


def test_generation_probe_build_report_with_fake_generation(
    tmp_path: Path,
) -> None:
    probe = _load_probe()
    args = probe._parse_args([
        "--active",
        "--observations-jsonl",
        str(tmp_path / "observations.jsonl"),
        "--output-json",
        str(tmp_path / "report.json"),
        "--output-md",
        str(tmp_path / "report.md"),
    ])

    def generate(parsed):
        Path(parsed.observations_jsonl).write_text(
            json.dumps(_valid_observation(True)) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "succeeded",
            "output_text": " generated",
            "error_type": None,
            "error": None,
        }

    report = probe.build_report(args, generation_fn=generate)

    assert report["generation_status"] == "succeeded"
    assert report["observations_written"] == 1
    assert report["phase12_8_active_selected_attention_candidate"] is True
    assert report["runtime_behavior_changed"] is False


def test_observation_only_does_not_open_phase12_8_gate(
    tmp_path: Path,
) -> None:
    probe = _load_probe()
    args = probe._parse_args([
        "--observations-jsonl",
        str(tmp_path / "observations.jsonl"),
        "--output-json",
        str(tmp_path / "report.json"),
        "--output-md",
        str(tmp_path / "report.md"),
    ])

    def generate(parsed):
        Path(parsed.observations_jsonl).write_text(
            json.dumps(_valid_observation(False)) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "succeeded",
            "output_text": " generated",
            "error_type": None,
            "error": None,
        }

    report = probe.build_report(args, generation_fn=generate)

    assert report["observations_written"] == 1
    assert report["phase12_8_active_selected_attention_candidate"] is False


def test_no_repository_vllm_files_are_patch_targets() -> None:
    patcher = _load_patcher()

    assert all(
        not target.relative_path.startswith("vllm/")
        for target in patcher.TARGETS
    )
