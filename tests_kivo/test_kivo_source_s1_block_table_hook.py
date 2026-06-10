# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


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


def _load_helper():
    from vllm.v1.worker import kivo_selected_blocks as helper

    return helper


def _load_probe():
    return _load_script(
        "run_source_s1_gpt2_probe.py",
        "source_s1_probe_test",
    )


def _load_validator():
    return _load_script(
        "validate_source_s1_observations.py",
        "source_s1_validator_test",
    )


def _fake_instance(slot_mapping, *, block_table=None):
    if block_table is None:
        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    return type(
        "FakeBlockTable",
        (),
        {
            "slot_mapping": type("SlotBuffer", (), {"gpu": slot_mapping})(),
            "block_table": type("BlockBuffer", (), {"gpu": block_table})(),
            "block_size": 16,
            "max_num_batched_tokens": 8,
            "num_blocks_per_row": torch.tensor([2, 1], dtype=torch.int32).numpy(),
            "max_num_blocks_per_req": 8,
            "max_num_reqs": 2,
            "pcp_world_size": 1,
            "pcp_rank": 0,
            "dcp_world_size": 1,
            "dcp_rank": 0,
            "cp_kv_cache_interleave_size": 1,
        },
    )()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_record(*, active: bool, applied: bool, blocker=None) -> dict:
    return {
        "schema_version": "kivo_source_s1_block_table_v1",
        "timestamp": 1.0,
        "pid": 1,
        "hook_name": "BlockTable.compute_slot_mapping",
        "class_name": "BlockTable",
        "function_name": "compute_slot_mapping",
        "args_summary": [],
        "slot_mapping_present": True,
        "slot_mapping_type": "torch.Tensor",
        "slot_mapping_shape": [3],
        "slot_mapping_dtype": "torch.int64",
        "slot_mapping_device": "cpu",
        "block_table_present": True,
        "block_table_type": "torch.Tensor",
        "block_table_shape": [2, 2],
        "block_table_dtype": "torch.int32",
        "block_table_device": "cpu",
        "result_type": "builtins.NoneType",
        "result_summary": {"type": "builtins.NoneType", "value": None},
        "block_size": 16,
        "num_blocks_per_row": [1, 1],
        "max_num_blocks_per_req": 4,
        "max_num_reqs": 2,
        "active_enabled": active,
        "mutation_attempted": active,
        "mutation_applied": applied,
        "mutation_policy": "mask_last_slot" if applied else None,
        "mutation_blocker_reason": blocker,
        "old_value": 3 if applied else None,
        "new_value": 2 if applied else None,
        "mutation_index": 2 if applied else None,
        "runtime_behavior_changed": applied,
        "active_routing": applied,
        "measured_runtime_reduction": False,
        "caveats": [],
    }


def test_disabled_mode_writes_nothing(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([10, 11, 12], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.delenv("KIVO_SOURCE_ENABLE", raising=False)
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1]), torch.tensor([0, 1])),
        kwargs={},
        result=None,
    )

    assert not path.exists()
    assert slot_mapping.tolist() == [10, 11, 12]


def test_observation_record_contains_required_fields(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([10, 11, 12], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1]), torch.tensor([0, 1])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert record["schema_version"] == "kivo_source_s1_block_table_v1"
    assert record["slot_mapping_present"] is True
    assert record["block_table_present"] is True
    assert record["measured_runtime_reduction"] is False
    assert record["active_routing"] is False


def test_mask_last_slot_mutates_final_integer_entry(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([10, 11, 12], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_last_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1]), torch.tensor([0, 1])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [10, 11, 11]
    assert record["mutation_attempted"] is True
    assert record["mutation_applied"] is True
    assert record["old_value"] == 12
    assert record["new_value"] == 11
    assert record["mutation_index"] == 2
    assert record["runtime_behavior_changed"] is True
    assert record["active_routing"] is True


def test_policy_refuses_too_short_tensor(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([10], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_last_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0]), torch.tensor([0])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [10]
    assert record["mutation_applied"] is False
    assert "at least two elements" in record["mutation_blocker_reason"]


def test_policy_refuses_non_integer_tensor(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([1.0, 2.0], dtype=torch.float32)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_last_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1]), torch.tensor([0, 1])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [1.0, 2.0]
    assert record["mutation_applied"] is False
    assert "integer dtype" in record["mutation_blocker_reason"]


def test_policy_refuses_missing_slot_mapping(tmp_path, monkeypatch):
    helper = _load_helper()
    instance = type(
        "FakeBlockTable",
        (),
        {
            "block_table": type(
                "BlockBuffer",
                (),
                {"gpu": torch.tensor([[0]], dtype=torch.int32)},
            )(),
            "block_size": 16,
            "num_blocks_per_row": [1],
            "max_num_blocks_per_req": 4,
            "max_num_reqs": 1,
        },
    )()
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_last_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0]), torch.tensor([0])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert record["mutation_applied"] is False
    assert "no safe Python-level slot mapping result found" in record[
        "mutation_blocker_reason"
    ]


def test_compute_slot_mapping_integration_records_and_mutates(
    tmp_path, monkeypatch
):
    try:
        from vllm.v1.worker import block_table as block_table_mod
    except ImportError:
        pytest.skip("vllm.worker.block_table import not available")

    class FakeKernel:
        def __getitem__(self, _):
            def launch(*_args, **_kwargs):
                return None

            return launch

    monkeypatch.setattr(
        block_table_mod,
        "_compute_slot_mapping_kernel",
        FakeKernel(),
    )
    instance = _fake_instance(torch.tensor([10, 11, 12], dtype=torch.int64))
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_last_slot")
    monkeypatch.setenv("KIVO_SOURCE_FAIL_CLOSED", "1")

    block_table_mod.BlockTable.compute_slot_mapping(
        instance,
        1,
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int64),
    )

    record = _read_jsonl(path)[0]

    assert record["mutation_applied"] is True
    assert record["active_routing"] is True
    assert instance.slot_mapping.gpu.tolist() == [10, 11, 11]


def test_validator_accepts_success_and_blocked_cases():
    validator = _load_validator()

    base = _source_record(active=False, applied=False)
    applied = _source_record(active=True, applied=True)
    blocked = _source_record(
        active=True,
        applied=False,
        blocker=(
            "tensor-like slot mapping requires tensor-safe mutation design"
        ),
    )

    report = validator.validate_observations([base], [base], [applied])
    blocked_report = validator.validate_observations([base], [base], [blocked])

    assert report["validation_passed"] is True
    assert blocked_report["validation_passed"] is True
    assert blocked_report["errors"] == []


def test_probe_reports_source_candidate(tmp_path):
    probe = _load_probe()
    args = probe._parse_args([
        "--baseline-obs-jsonl",
        str(tmp_path / "baseline.jsonl"),
        "--observation-obs-jsonl",
        str(tmp_path / "observation.jsonl"),
        "--active-obs-jsonl",
        str(tmp_path / "active.jsonl"),
    ])

    def generate(_args):
        path = Path(os.environ["KIVO_SOURCE_OBS_PATH"])
        if os.environ.get("KIVO_SOURCE_ACTIVE") == "1":
            record = _source_record(active=True, applied=True)
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            return {"status": "succeeded", "output_text": "active", "error": None}
        record = _source_record(active=False, applied=False)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return {"status": "succeeded", "output_text": "base", "error": None}

    report = probe.build_report(args, generation_fn=generate)

    assert report["source_selected_block_candidate"] is True
    assert report["mutation_applied"] is True
