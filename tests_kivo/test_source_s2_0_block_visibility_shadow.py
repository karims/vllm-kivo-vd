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
        "shadow_success_count": 1,
        "total_records": 2,
        "mutation_applied_count": 0,
        "active_routing_count": 0,
        "runtime_behavior_changed_count": 0,
        "max_visible_block_count": 4,
        "max_selected_block_count": 2,
        "max_theoretical_visible_block_reduction": 2,
        "max_theoretical_visible_block_reduction_ratio": 0.5,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s2_shadow_passed": True,
        "prompt_results": [{"prompt_index": 0}],
    }


def test_valid_slots_to_visible_blocks_ignores_padding():
    helper = _load_helper()
    slots = torch.tensor(
        [-1, 0, 15, 16, 17, 47, -1],
        dtype=torch.int64,
    )

    valid_slots, visible_blocks = helper._visible_block_ids(
        slots,
        block_size=16,
        pad_slot_id=-1,
    )

    assert valid_slots == [0, 15, 16, 17, 47]
    assert visible_blocks == [0, 1, 2]


def test_shadow_selection_keeps_recent_block_and_budget():
    helper = _load_helper()

    selected = helper._select_shadow_blocks(
        [10, 11, 12, 13, 14],
        budget_ratio=0.5,
        keep_recent_blocks=1,
    )

    assert 14 in selected
    assert len(selected) == 3
    assert set(selected).issubset({10, 11, 12, 13, 14})


def test_shadow_record_computes_reduction_without_mutation(
    tmp_path,
    monkeypatch,
):
    helper = _load_helper()
    slots = torch.tensor(
        [-1, 16, 17, 32, 48, -1],
        dtype=torch.int64,
    )
    original = slots.clone()
    instance = _fake_instance(slots)
    output = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(output))
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "sketch_shadow_blocks")
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
    assert torch.equal(slots, original)
    assert record["schema_version"] == (
        "kivo_source_s2_0_block_visibility_shadow_v1"
    )
    assert record["visible_block_count"] == 3
    assert record["selected_block_count"] == 2
    assert record["theoretical_visible_block_reduction"] == 1
    assert record["theoretical_visible_block_reduction_ratio"] == pytest.approx(
        1 / 3
    )
    assert record["selected_block_ids_sample"][-1] == 3
    assert record["mutation_attempted"] is False
    assert record["mutation_applied"] is False
    assert record["runtime_behavior_changed"] is False
    assert record["active_routing"] is False
    assert record["measured_runtime_reduction"] is False


def test_selected_count_never_exceeds_visible_count():
    helper = _load_helper()

    selected = helper._select_shadow_blocks(
        [3, 4],
        budget_ratio=2.0,
        keep_recent_blocks=5,
    )

    assert selected == [3, 4]


def test_runner_aggregates_non_mutating_shadow_report(tmp_path):
    runner = _load_script(
        "run_source_s2_0_block_visibility_shadow.py",
        "source_s2_0_runner_test",
    )
    args = runner._parse_args(
        [
            "--prompt",
            "test prompt",
            "--observation-dir",
            str(tmp_path / "observations"),
            "--output-json",
            str(tmp_path / "report.json"),
            "--output-md",
            str(tmp_path / "report.md"),
        ]
    )

    def fake_generation(generation_args):
        return {
            "status": "succeeded",
            "output_text": f"output:{generation_args.prompt}",
            "error": None,
        }

    def fake_records(_path):
        return [
            {
                "visible_block_count": 4,
                "selected_block_count": 2,
                "theoretical_visible_block_reduction": 2,
                "theoretical_visible_block_reduction_ratio": 0.5,
                "mutation_applied": False,
                "active_routing": False,
                "runtime_behavior_changed": False,
            }
        ]

    report = runner.build_report(
        args,
        generation_fn=fake_generation,
        record_loader=fake_records,
    )

    assert report["baseline_success_count"] == 1
    assert report["shadow_success_count"] == 1
    assert report["output_changed_count"] == 0
    assert report["total_records"] == 1
    assert report["max_visible_block_count"] == 4
    assert report["max_selected_block_count"] == 2
    assert report["mutation_applied_count"] == 0
    assert report["active_routing_count"] == 0
    assert report["runtime_behavior_changed_count"] == 0
    assert report["s2_shadow_passed"] is True


def test_validator_passes_good_report():
    validator = _load_script(
        "validate_source_s2_0_block_visibility_shadow.py",
        "source_s2_0_validator_good_test",
    )

    result = validator.validate_report(_good_report())

    assert result["validation_passed"] is True
    assert result["errors"] == []


def test_validator_rejects_mutation():
    validator = _load_script(
        "validate_source_s2_0_block_visibility_shadow.py",
        "source_s2_0_validator_mutation_test",
    )
    report = _good_report()
    report["mutation_applied_count"] = 1

    result = validator.validate_report(report)

    assert result["validation_passed"] is False
    assert "mutation_applied_count must be 0" in result["errors"]


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("measured_runtime_reduction", "must be false"),
        ("performance_claim_allowed", "must be false"),
    ],
)
def test_validator_rejects_memory_or_performance_claim(field, expected):
    validator = _load_script(
        "validate_source_s2_0_block_visibility_shadow.py",
        f"source_s2_0_validator_claim_{field}",
    )
    report = _good_report()
    report[field] = True

    result = validator.validate_report(report)

    assert result["validation_passed"] is False
    assert any(expected in error for error in result["errors"])
