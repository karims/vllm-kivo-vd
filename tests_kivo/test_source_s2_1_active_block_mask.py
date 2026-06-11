# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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


def _fake_instance(slot_mapping, *, block_size=16):
    block_table = torch.tensor([[4, 7, 9, 12]], dtype=torch.int32)
    return type(
        "FakeBlockTable",
        (),
        {
            "slot_mapping": type("SlotBuffer", (), {"gpu": slot_mapping})(),
            "block_table": type("BlockBuffer", (), {"gpu": block_table})(),
            "block_size": block_size,
        },
    )()


def _good_report() -> dict:
    return {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "active_success_count": 1,
        "output_changed_count": 0,
        "mutation_applied_prompt_count": 1,
        "total_remapped_slot_count": 2,
        "max_visible_block_count": 4,
        "max_selected_block_count": 3,
        "max_unselected_block_count": 1,
        "active_routing_count": 1,
        "runtime_behavior_changed_count": 1,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s2_1_active_mask_passed": True,
        "prompt_results": [
            {
                "prompt_index": 0,
                "prompt": "alpha",
                "baseline_status": "succeeded",
                "active_status": "succeeded",
                "baseline_output": "baseline",
                "active_output": "active",
                "baseline_error": None,
                "active_error": None,
                "output_changed": False,
                "records_written": 1,
                "max_visible_block_count": 4,
                "max_selected_block_count": 3,
                "max_unselected_block_count": 1,
                "total_remapped_slot_count": 2,
                "mutation_attempted_count": 1,
                "mutation_applied_count": 1,
                "active_routing_count": 1,
                "runtime_behavior_changed_count": 1,
            }
        ],
    }


def _mixed_report() -> dict:
    return {
        "total_prompts": 3,
        "baseline_success_count": 3,
        "active_success_count": 3,
        "output_changed_count": 2,
        "mutation_applied_prompt_count": 2,
        "total_remapped_slot_count": 32,
        "max_visible_block_count": 3,
        "max_selected_block_count": 2,
        "max_unselected_block_count": 1,
        "active_routing_count": 2,
        "runtime_behavior_changed_count": 2,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s2_1_active_mask_passed": True,
        "prompt_results": [
            {
                "prompt_index": 0,
                "prompt": "p0",
                "baseline_status": "succeeded",
                "active_status": "succeeded",
                "baseline_output": "baseline0",
                "active_output": "active0",
                "baseline_error": None,
                "active_error": None,
                "output_changed": False,
                "records_written": 1,
                "max_visible_block_count": 1,
                "max_selected_block_count": 1,
                "max_unselected_block_count": 0,
                "total_remapped_slot_count": 0,
                "mutation_attempted_count": 0,
                "mutation_applied_count": 0,
                "active_routing_count": 0,
                "runtime_behavior_changed_count": 0,
            },
            {
                "prompt_index": 1,
                "prompt": "p1",
                "baseline_status": "succeeded",
                "active_status": "succeeded",
                "baseline_output": "baseline1",
                "active_output": "active1",
                "baseline_error": None,
                "active_error": None,
                "output_changed": True,
                "records_written": 1,
                "max_visible_block_count": 2,
                "max_selected_block_count": 1,
                "max_unselected_block_count": 1,
                "total_remapped_slot_count": 16,
                "mutation_attempted_count": 1,
                "mutation_applied_count": 1,
                "active_routing_count": 1,
                "runtime_behavior_changed_count": 1,
            },
            {
                "prompt_index": 2,
                "prompt": "p2",
                "baseline_status": "succeeded",
                "active_status": "succeeded",
                "baseline_output": "baseline2",
                "active_output": "active2",
                "baseline_error": None,
                "active_error": None,
                "output_changed": True,
                "records_written": 1,
                "max_visible_block_count": 3,
                "max_selected_block_count": 2,
                "max_unselected_block_count": 1,
                "total_remapped_slot_count": 16,
                "mutation_attempted_count": 1,
                "mutation_applied_count": 1,
                "active_routing_count": 1,
                "runtime_behavior_changed_count": 1,
            },
        ],
    }


def test_visible_selected_unselected_block_derivation():
    helper = _load_helper()
    slots = torch.tensor([-1, 0, 1, 16, 17, 32, -1], dtype=torch.int64)

    valid_slots, visible_blocks = helper._visible_block_ids(
        slots,
        block_size=16,
        pad_slot_id=-1,
    )
    selected = helper._select_shadow_blocks(
        visible_blocks,
        budget_ratio=0.5,
        keep_recent_blocks=1,
    )

    assert valid_slots == [0, 1, 16, 17, 32]
    assert visible_blocks == [0, 1, 2]
    assert selected[-1] == 2
    assert len(set(visible_blocks) - set(selected)) >= 0


def test_recent_block_is_not_remapped(tmp_path, monkeypatch):
    helper = _load_helper()
    slots = torch.tensor([-1, 0, 1, 16, 17, 32, 33], dtype=torch.int64)
    original = slots.clone()
    instance = _fake_instance(slots)
    output = tmp_path / "active.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(output))
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "active_mask_unselected_blocks")
    monkeypatch.setenv("KIVO_SOURCE_BUDGET_RATIO", "0.5")
    monkeypatch.setenv("KIVO_SOURCE_KEEP_RECENT_BLOCKS", "1")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(),
        kwargs={},
        result=None,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert torch.equal(slots[-2:], original[-2:])
    assert torch.equal(slots[:4], original[:4]) is False
    assert record["schema_version"] == "kivo_source_s2_1_active_block_mask_v1"
    assert record["policy_name"] == "active_mask_unselected_blocks"
    assert record["mutation_attempted"] is True
    assert record["mutation_applied"] is True
    assert record["active_routing"] is True
    assert record["runtime_behavior_changed"] is True
    assert record["remapped_slot_count"] > 0
    assert record["selected_block_count"] >= 1
    assert record["unselected_block_count"] >= 1


def test_no_mutation_when_visible_block_count_is_one(tmp_path, monkeypatch):
    helper = _load_helper()
    slots = torch.tensor([-1, 0, 1, -1], dtype=torch.int64)
    instance = _fake_instance(slots)
    output = tmp_path / "active.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(output))
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "active_mask_unselected_blocks")
    monkeypatch.setenv("KIVO_SOURCE_BUDGET_RATIO", "0.5")
    monkeypatch.setenv("KIVO_SOURCE_KEEP_RECENT_BLOCKS", "1")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(),
        kwargs={},
        result=None,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["visible_block_count"] == 1
    assert record["mutation_attempted"] is False
    assert record["mutation_applied"] is False
    assert record["active_routing"] is False
    assert record["runtime_behavior_changed"] is False


def test_no_mutation_when_all_blocks_selected(tmp_path, monkeypatch):
    helper = _load_helper()
    slots = torch.tensor([0, 1, 16, 17], dtype=torch.int64)
    instance = _fake_instance(slots)
    output = tmp_path / "active.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(output))
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "active_mask_unselected_blocks")
    monkeypatch.setenv("KIVO_SOURCE_BUDGET_RATIO", "1.0")
    monkeypatch.setenv("KIVO_SOURCE_KEEP_RECENT_BLOCKS", "1")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(),
        kwargs={},
        result=None,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["mutation_attempted"] is False
    assert record["mutation_applied"] is False
    assert record["remapped_slot_count"] == 0


def test_validator_passes_good_report():
    validator = _load_script(
        "validate_source_s2_1_active_block_mask.py",
        "source_s2_1_validator_good_test",
    )

    result = validator.validate_report(_good_report())

    assert result["validation_passed"] is True
    assert result["errors"] == []


def test_validator_passes_mixed_report_with_noop_prompt():
    validator = _load_script(
        "validate_source_s2_1_active_block_mask.py",
        "source_s2_1_validator_mixed_test",
    )

    result = validator.validate_report(_mixed_report())

    assert result["validation_passed"] is True
    assert result["errors"] == []


def test_validator_rejects_missing_remaps():
    validator = _load_script(
        "validate_source_s2_1_active_block_mask.py",
        "source_s2_1_validator_missing_test",
    )
    report = _good_report()
    report["total_remapped_slot_count"] = 0

    result = validator.validate_report(report)

    assert result["validation_passed"] is False
    assert "total_remapped_slot_count must be > 0" in result["errors"]


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("measured_runtime_reduction", "must be false"),
        ("performance_claim_allowed", "must be false"),
    ],
)
def test_validator_rejects_memory_or_performance_claim(field, expected):
    validator = _load_script(
        "validate_source_s2_1_active_block_mask.py",
        f"source_s2_1_validator_claim_{field}",
    )
    report = _good_report()
    report[field] = True

    result = validator.validate_report(report)

    assert result["validation_passed"] is False
    assert any(expected in error for error in result["errors"])
