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
        "phase12_10_patcher_test",
    )


def _load_probe():
    return _load_script(
        "run_phase12_10_block_table_mutation_probe.py",
        "phase12_10_probe_test",
    )


def _load_validator():
    return _load_script(
        "validate_phase12_10_block_table_mutation.py",
        "phase12_10_validator_test",
    )


def _active_target(patcher):
    return next(
        target
        for target in patcher.TARGETS
        if target.name == "block_table_compute_slot_mapping_active"
    )


def _patched_namespace(tmp_path: Path):
    patcher = _load_patcher()
    target = _active_target(patcher)
    source = (
        "class BlockTable:\n"
        "    def compute_slot_mapping(self):\n"
        "        return self.result\n"
    )
    target_path = tmp_path / "block_table.py"
    target_path.write_text(source, encoding="utf-8")
    patched = patcher.build_patched_source(source, target)
    namespace = {"__file__": str(target_path)}
    exec(compile(patched, str(target_path), "exec"), namespace)
    return patcher, target, patched, namespace


def _run_wrapper(monkeypatch, tmp_path: Path, result, *, active: bool):
    _, _, _, namespace = _patched_namespace(tmp_path)
    observations = tmp_path / ("active.jsonl" if active else "baseline.jsonl")
    monkeypatch.setenv("KIVO_PHASE12_10_ENABLE", "1")
    monkeypatch.setenv("KIVO_PHASE12_10_OBS_PATH", str(observations))
    if active:
        monkeypatch.setenv("KIVO_PHASE12_10_ACTIVE", "1")
    else:
        monkeypatch.delenv("KIVO_PHASE12_10_ACTIVE", raising=False)
    table = namespace["BlockTable"]()
    table.result = result
    returned = table.compute_slot_mapping()
    records = [
        json.loads(line)
        for line in observations.read_text(encoding="utf-8").splitlines()
    ]
    return returned, records


def _record(*, attempted=False, applied=False, blocker=None, tensor=False):
    return {
        "schema_version": "phase12_10_block_table_slot_mapping_v1",
        "timestamp": 1.0,
        "pid": 1,
        "hook_name": "block_table_compute_slot_mapping",
        "module_file": "/tmp/block_table.py",
        "class_name": "BlockTable",
        "function_name": "compute_slot_mapping",
        "self_type": "fake.BlockTable",
        "args_summary": [],
        "kwargs_keys": [],
        "result_type": "builtins.list",
        "result_summary": {"type": "builtins.list", "length": 3},
        "slot_like_result_found": True,
        "block_like_result_found": False,
        "tensor_like_result_found": tensor,
        "python_mutable_result_found": True,
        "mutation_attempted": attempted,
        "mutation_applied": applied,
        "mutation_policy": (
            "drop_last_python_slot_entry" if applied else None
        ),
        "blocker_reason": blocker,
        "runtime_behavior_changed": applied,
        "active_routing": applied,
        "measured_runtime_reduction": False,
        "caveats": [],
    }


def test_patch_target_detection_and_restore(tmp_path: Path):
    patcher = _load_patcher()
    target = _active_target(patcher)
    package_root = tmp_path / "site-packages" / "vllm"
    target_path = package_root / target.relative_path
    target_path.parent.mkdir(parents=True)
    target_path.write_text(
        "class BlockTable:\n"
        "    def compute_slot_mapping(self):\n"
        "        return [1, 2, 3]\n",
        encoding="utf-8",
    )
    backup_dir = tmp_path / "backups"
    original = target_path.read_bytes()

    manifest = patcher.install_patch(package_root, target, backup_dir)
    patched = target_path.read_text(encoding="utf-8")
    restored = patcher.restore_patch(backup_dir)

    assert manifest["target"]["name"] == target.name
    assert patcher.BLOCK_TABLE_ACTIVE_BEGIN_MARKER in patched
    assert restored["restored_exactly"] is True
    assert target_path.read_bytes() == original


def test_disabled_mode_returns_exact_original(monkeypatch, tmp_path):
    _, _, _, namespace = _patched_namespace(tmp_path)
    monkeypatch.delenv("KIVO_PHASE12_10_ENABLE", raising=False)
    result = [1, 2, 3]
    table = namespace["BlockTable"]()
    table.result = result

    returned = table.compute_slot_mapping()

    assert returned is result


def test_list_result_mutates_copied_list(monkeypatch, tmp_path):
    result = [1, 2, 3]

    returned, records = _run_wrapper(
        monkeypatch, tmp_path, result, active=True
    )

    assert returned == [1, 2]
    assert returned is not result
    assert result == [1, 2, 3]
    assert records[0]["mutation_applied"] is True
    assert records[0]["mutation_policy"] == "drop_last_python_slot_entry"


def test_tuple_result_mutates_copied_tuple(monkeypatch, tmp_path):
    result = (1, 2, 3)

    returned, records = _run_wrapper(
        monkeypatch, tmp_path, result, active=True
    )

    assert returned == (1, 2)
    assert returned is not result
    assert result == (1, 2, 3)
    assert records[0]["mutation_applied"] is True


def test_tensor_like_result_is_blocked(monkeypatch, tmp_path):
    class TensorLike:
        shape = (4,)
        dtype = "int32"
        device = "cuda:0"

    result = TensorLike()

    returned, records = _run_wrapper(
        monkeypatch, tmp_path, result, active=True
    )

    assert returned is result
    assert records[0]["tensor_like_result_found"] is True
    assert records[0]["mutation_applied"] is False
    assert records[0]["blocker_reason"] == (
        "tensor-like slot mapping requires tensor-safe mutation design"
    )


def test_unsupported_result_records_blocker(monkeypatch, tmp_path):
    returned, records = _run_wrapper(
        monkeypatch, tmp_path, {"slot": [1, 2]}, active=True
    )

    assert returned == {"slot": [1, 2]}
    assert records[0]["mutation_applied"] is False
    assert records[0]["blocker_reason"] == (
        "no safe Python-level slot mapping result found"
    )


def test_validator_passes_applied_and_tensor_blocked_cases():
    validator = _load_validator()

    applied = validator.validate_records(
        [_record()],
        [_record(attempted=True, applied=True)],
    )
    blocked = validator.validate_records(
        [_record()],
        [
            _record(
                attempted=True,
                blocker="tensor-like slot mapping requires tensor-safe mutation design",
                tensor=True,
            )
        ],
    )

    assert applied["validation_passed"] is True
    assert blocked["validation_passed"] is True
    assert blocked["tensor_like_blocker_records"] == 1


def test_probe_reports_applied_mutation(tmp_path):
    probe = _load_probe()
    args = probe._parse_args([
        "--baseline-obs-jsonl",
        str(tmp_path / "baseline.jsonl"),
        "--active-obs-jsonl",
        str(tmp_path / "active.jsonl"),
    ])

    def generate(_args):
        path = Path(os.environ["KIVO_PHASE12_10_OBS_PATH"])
        if os.environ.get("KIVO_PHASE12_10_ACTIVE") == "1":
            path.write_text(
                json.dumps(_record(attempted=True, applied=True)) + "\n",
                encoding="utf-8",
            )
            return {"status": "succeeded", "output_text": "active", "error": None}
        path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        return {"status": "succeeded", "output_text": "base", "error": None}

    report = probe.build_report(args, generation_fn=generate)

    assert report["mutation_applied"] is True
    assert report["output_changed"] is True
    assert report["phase13_selected_attention_candidate"] is True


def test_no_repository_vllm_files_are_modified_targets():
    patcher = _load_patcher()
    target = _active_target(patcher)

    assert not target.relative_path.startswith("vllm/")
