# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import sys
from pathlib import Path


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
        "phase12_8_9_patcher_test",
    )


def _load_probe():
    return _load_script(
        "run_phase12_8_9_active_ladder_probe.py",
        "phase12_8_9_probe_test",
    )


def _load_validator():
    return _load_script(
        "validate_phase12_8_9_active_ladder.py",
        "phase12_8_9_validator_test",
    )


def _active_target(patcher):
    return next(
        target
        for target in patcher.TARGETS
        if target.name == "slot_mappings_active_ladder"
    )


def _patched_namespace(tmp_path: Path):
    patcher = _load_patcher()
    target = _active_target(patcher)
    source = (
        "class GPUModelRunner:\n"
        "    def _get_slot_mappings(self):\n"
        "        return self.result\n"
    )
    target_path = tmp_path / "gpu_model_runner.py"
    target_path.write_text(source, encoding="utf-8")
    patched = patcher.build_patched_source(source, target)
    namespace = {"__file__": str(target_path)}
    exec(compile(patched, str(target_path), "exec"), namespace)
    return patcher, target, patched, namespace


def _run_wrapper(monkeypatch, tmp_path: Path, result, stage, active=True):
    _, _, _, namespace = _patched_namespace(tmp_path)
    observations = tmp_path / f"{stage}.jsonl"
    monkeypatch.setenv("KIVO_PHASE12_8_9_ENABLE", "1")
    monkeypatch.setenv("KIVO_PHASE12_8_9_STAGE", stage)
    monkeypatch.setenv("KIVO_PHASE12_8_9_OBS_PATH", str(observations))
    monkeypatch.setenv("KIVO_PHASE12_8_9_MAX_MUTATIONS", "1")
    if active:
        monkeypatch.setenv("KIVO_PHASE12_8_9_ACTIVE", "1")
    else:
        monkeypatch.delenv("KIVO_PHASE12_8_9_ACTIVE", raising=False)
    runner = namespace["GPUModelRunner"]()
    runner.result = result
    returned = runner._get_slot_mappings()
    records = [
        json.loads(line)
        for line in observations.read_text(encoding="utf-8").splitlines()
    ]
    return returned, records


def test_disabled_environment_returns_exact_original(monkeypatch, tmp_path):
    _, _, _, namespace = _patched_namespace(tmp_path)
    monkeypatch.delenv("KIVO_PHASE12_8_9_ENABLE", raising=False)
    result = ({"request": [1, 2]}, {"layer.0": object()})
    runner = namespace["GPUModelRunner"]()
    runner.result = result

    returned = runner._get_slot_mappings()

    assert returned is result


def test_metadata_mutation_copies_and_removes_one_key(monkeypatch, tmp_path):
    metadata = {"layer.0": object(), "layer.1": object()}
    result = ({"request": [1, 2]}, metadata)

    returned, records = _run_wrapper(
        monkeypatch, tmp_path, result, "metadata"
    )

    assert returned is not result
    assert returned[0] is result[0]
    assert returned[1] is not metadata
    assert list(metadata) == ["layer.0", "layer.1"]
    assert list(returned[1]) == ["layer.0"]
    assert records[0]["mutation_stage"] == "metadata_drop_one_key"
    assert records[0]["removed_key"] == "layer.1"
    assert records[0]["mutation_applied"] is True


def test_selected_slot_blocks_without_plain_sequence(monkeypatch, tmp_path):
    class TensorLike:
        shape = (2,)

    result = ({"request": TensorLike()}, {"layer.0": object()})

    returned, records = _run_wrapper(
        monkeypatch, tmp_path, result, "selected_slot"
    )

    assert returned is result
    assert records[0]["mutation_attempted"] is True
    assert records[0]["mutation_applied"] is False
    assert records[0]["blocker_reason"] == (
        "no safe Python-level selected-slot/block structure found in "
        "_get_slot_mappings result"
    )


def test_selected_slot_mutates_only_copied_list(monkeypatch, tmp_path):
    slots = [10, 11, 12]
    mappings = {"request": slots, "other": "unchanged"}
    result = (mappings, {"layer.0": object()})

    returned, records = _run_wrapper(
        monkeypatch, tmp_path, result, "selected_slot"
    )

    assert returned is not result
    assert returned[0] is not mappings
    assert returned[0]["request"] == [10, 11]
    assert mappings["request"] is slots
    assert slots == [10, 11, 12]
    assert records[0]["active_routing"] is True
    assert records[0]["mutation_applied"] is True


def test_active_patch_backup_restore_is_exact(tmp_path):
    patcher = _load_patcher()
    target = _active_target(patcher)
    package_root = tmp_path / "site-packages" / "vllm"
    target_path = package_root / target.relative_path
    target_path.parent.mkdir(parents=True)
    target_path.write_text(
        "class GPUModelRunner:\n"
        "    def _get_slot_mappings(self):\n"
        "        return ({}, {})\n",
        encoding="utf-8",
    )
    original = target_path.read_bytes()
    backup_dir = tmp_path / "backups"

    manifest = patcher.install_patch(package_root, target, backup_dir)

    patched = target_path.read_text(encoding="utf-8")
    assert patcher.ACTIVE_BEGIN_MARKER in patched
    assert manifest["markers"] == [
        patcher.ACTIVE_BEGIN_MARKER,
        patcher.ACTIVE_END_MARKER,
    ]
    restored = patcher.restore_patch(backup_dir)
    assert restored["restored_exactly"] is True
    assert target_path.read_bytes() == original


def _record(stage, *, attempted=False, applied=False, blocker=None):
    return {
        "schema_version": "phase12_8_9_active_ladder_v1",
        "mutation_stage": stage,
        "mutation_attempted": attempted,
        "mutation_applied": applied,
        "blocker_reason": blocker,
        "measured_runtime_reduction": False,
    }


def test_validator_accepts_success_and_blocked_selected_slot():
    validator = _load_validator()

    success = validator.validate_ladder(
        [_record("baseline")],
        [_record("metadata_drop_one_key", attempted=True, applied=True)],
        [_record("selected_slot_drop_one", attempted=True, applied=True)],
    )
    blocked = validator.validate_ladder(
        [_record("baseline")],
        [_record("metadata_drop_one_key", attempted=True, applied=True)],
        [
            _record(
                "selected_slot",
                attempted=True,
                blocker="no safe structure",
            )
        ],
    )
    crashed_or_skipped = validator.validate_ladder(
        [_record("baseline")],
        [_record("metadata_drop_one_key", attempted=True, applied=True)],
        [],
    )

    assert success["validation_passed"] is True
    assert success["active_routing"] is True
    assert blocked["validation_passed"] is True
    assert blocked["active_routing"] is False
    assert crashed_or_skipped["validation_passed"] is True
    assert crashed_or_skipped["warnings"]


def test_probe_captures_crash_and_skips_selected_slot(tmp_path):
    probe = _load_probe()
    args = probe._parse_args([
        "--baseline-obs-jsonl",
        str(tmp_path / "baseline.jsonl"),
        "--metadata-obs-jsonl",
        str(tmp_path / "metadata.jsonl"),
        "--selected-slot-obs-jsonl",
        str(tmp_path / "selected.jsonl"),
    ])
    calls = 0

    def generate(_args):
        nonlocal calls
        calls += 1
        path = Path(os.environ["KIVO_PHASE12_8_9_OBS_PATH"])
        stage = os.environ["KIVO_PHASE12_8_9_STAGE"]
        if stage == "baseline":
            path.write_text(json.dumps(_record("baseline")) + "\n")
            return {"status": "succeeded", "output_text": "same", "error": None}
        raise RuntimeError("metadata crash")

    report = probe.build_report(args, generation_fn=generate)

    assert calls == 2
    assert report["metadata_generation_status"] == "failed"
    assert "metadata crash" in report["metadata_error"]
    assert report["selected_slot_generation_status"] == "skipped"
    assert report["phase13_selected_attention_candidate"] is False


def test_probe_runs_selected_stage_after_metadata_success(tmp_path):
    probe = _load_probe()
    args = probe._parse_args([
        "--baseline-obs-jsonl",
        str(tmp_path / "baseline.jsonl"),
        "--metadata-obs-jsonl",
        str(tmp_path / "metadata.jsonl"),
        "--selected-slot-obs-jsonl",
        str(tmp_path / "selected.jsonl"),
    ])

    def generate(_args):
        path = Path(os.environ["KIVO_PHASE12_8_9_OBS_PATH"])
        stage = os.environ["KIVO_PHASE12_8_9_STAGE"]
        record = {
            "baseline": _record("baseline"),
            "metadata": _record(
                "metadata_drop_one_key", attempted=True, applied=True
            ),
            "selected_slot": _record(
                "selected_slot_drop_one", attempted=True, applied=True
            ),
        }[stage]
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return {"status": "succeeded", "output_text": "same", "error": None}

    report = probe.build_report(args, generation_fn=generate)

    assert report["selected_slot_generation_status"] == "succeeded"
    assert report["selected_slot_mutation_applied"] is True
    assert report["active_routing"] is True
    assert report["phase13_selected_attention_candidate"] is True


def test_no_repository_vllm_file_is_an_active_target():
    patcher = _load_patcher()
    target = _active_target(patcher)

    assert not target.relative_path.startswith("vllm/")
