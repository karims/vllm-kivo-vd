# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


def test_mask_oldest_valid_slot_mutates_oldest_entry(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([16, 17, 18, 19], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_oldest_valid_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1]), torch.tensor([0, 1])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [17, 17, 18, 19]
    assert record["mutation_applied"] is True
    assert record["mutation_target_position"] == "oldest"
    assert record["old_value"] == 16
    assert record["new_value"] == 17
    assert record["valid_slot_count"] == 4
    assert record["previous_valid_index"] is None
    assert record["next_valid_index"] == 1
    assert record["old_new_differ"] is True


def test_mask_middle_valid_slot_mutates_middle_entry(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([-1, 5, 6, 7, -1], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_middle_valid_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [-1, 5, 5, 7, -1]
    assert record["mutation_applied"] is True
    assert record["mutation_target_position"] == "middle"
    assert record["old_value"] == 6
    assert record["new_value"] == 5
    assert record["valid_slot_count"] == 3
    assert record["previous_valid_index"] == 1
    assert record["next_valid_index"] == 3
    assert record["old_new_differ"] is True


def test_mask_middle_valid_slot_requires_three_valid_entries(tmp_path, monkeypatch):
    helper = _load_helper()
    slot_mapping = torch.tensor([-1, 5, -1], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_middle_valid_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [-1, 5, -1]
    assert record["mutation_applied"] is False
    assert "fewer than three valid slot entries" in record[
        "mutation_blocker_reason"
    ]


def test_mask_oldest_valid_slot_requires_differing_candidate(
    tmp_path, monkeypatch
):
    helper = _load_helper()
    slot_mapping = torch.tensor([5, 5], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "mask_oldest_valid_slot")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1]), torch.tensor([0, 1])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [5, 5]
    assert record["mutation_applied"] is False
    assert "no differing valid slot pair found" in record[
        "mutation_blocker_reason"
    ]


def test_noop_valid_slot_shadow_records_candidate_without_mutation(
    tmp_path, monkeypatch
):
    helper = _load_helper()
    slot_mapping = torch.tensor([16, 17, 18], dtype=torch.int64)
    instance = _fake_instance(slot_mapping)
    path = tmp_path / "obs.jsonl"
    monkeypatch.setenv("KIVO_SOURCE_ENABLE", "1")
    monkeypatch.setenv("KIVO_SOURCE_OBS_PATH", str(path))
    monkeypatch.setenv("KIVO_SOURCE_ACTIVE", "1")
    monkeypatch.setenv("KIVO_SOURCE_POLICY", "noop_valid_slot_shadow")

    helper.maybe_observe_compute_slot_mapping(
        instance,
        module_file="/tmp/block_table.py",
        function_name="compute_slot_mapping",
        args=(1, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        kwargs={},
        result=None,
    )

    record = _read_jsonl(path)[0]

    assert slot_mapping.tolist() == [16, 17, 18]
    assert record["mutation_attempted"] is True
    assert record["mutation_applied"] is False
    assert record["mutation_target_position"] == "shadow"
    assert record["runtime_behavior_changed"] is False
    assert record["active_routing"] is False
    assert record["old_new_differ"] is True


def test_policy_drift_aggregation_and_best_policy(tmp_path, monkeypatch):
    module = _load_script(
        "run_source_s1_2_quality_sanity.py",
        "source_s1_3_policy_drift_runner_test",
    )

    args = module._parse_args([
        "--prompt",
        "alpha",
        "--prompt",
        "beta",
        "--output-json",
        str(tmp_path / "report.json"),
        "--output-md",
        str(tmp_path / "report.md"),
    ])

    def fake_run_generation(probe_args):
        policy = os.getenv("KIVO_SOURCE_POLICY", "")
        active = os.getenv("KIVO_SOURCE_ACTIVE") == "1"
        if not active:
            return {
                "status": "succeeded",
                "output_text": f"baseline:{probe_args.prompt}",
                "error": None,
            }
        outputs = {
            "mask_oldest_valid_slot": "drift-oldest",
            "mask_middle_valid_slot": "drift-middle",
            "mask_last_valid_slot": "drift-last",
        }
        if probe_args.prompt == "alpha":
            return {
                "status": "succeeded",
                "output_text": f"baseline:{probe_args.prompt}",
                "error": None,
            }
        return {
            "status": "succeeded",
            "output_text": outputs[policy],
            "error": None,
        }

    def fake_load_records(path):
        path_text = str(path)
        policy = (
            "mask_oldest_valid_slot"
            if "mask_oldest_valid_slot" in path_text
            else "mask_middle_valid_slot"
            if "mask_middle_valid_slot" in path_text
            else "mask_last_valid_slot"
            if "mask_last_valid_slot" in path_text
            else "unknown"
        )
        return [
            {
                "mutation_attempted": True,
                "mutation_applied": True,
                "active_routing": True,
                "runtime_behavior_changed": True,
                "old_new_differ": True,
                "valid_slot_count": 4 if policy != "mask_last_valid_slot" else 5,
                "mutation_blocker_reason": None,
            }
        ]

    monkeypatch.setattr(module, "_run_generation", fake_run_generation)
    monkeypatch.setattr(module, "_load_prompt_records", fake_load_records)

    report = module.build_drift_report(args)

    assert report["total_prompts"] == 2
    assert report["baseline_success_count"] == 2
    assert report["policies"] == [
        "mask_oldest_valid_slot",
        "mask_middle_valid_slot",
        "mask_last_valid_slot",
    ]
    assert report["per_policy"]["mask_oldest_valid_slot"][
        "mutation_applied_prompt_count"
    ] == 2
    assert report["per_policy"]["mask_middle_valid_slot"][
        "mutation_applied_prompt_count"
    ] == 2
    assert report["per_policy"]["mask_last_valid_slot"][
        "mutation_applied_prompt_count"
    ] == 2
    assert report["per_policy"]["mask_oldest_valid_slot"]["output_changed_count"] == 1
    assert report["per_policy"]["mask_middle_valid_slot"]["output_changed_count"] == 1
    assert report["per_policy"]["mask_last_valid_slot"]["output_changed_count"] == 1
    assert report["best_drift_policy"]["policy"] == "mask_oldest_valid_slot"
    assert report["quality_sanity_passed"] is True
    assert report["selected_attention_claim_allowed"] is False
    assert report["performance_claim_allowed"] is False


def test_validator_rejects_claims():
    validator = _load_script(
        "validate_source_s1_3_policy_drift.py",
        "source_s1_3_policy_drift_validator_test",
    )

    good = {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "policies": ["mask_oldest_valid_slot", "mask_last_valid_slot"],
        "per_policy": {
            "mask_oldest_valid_slot": {
                "active_success_count": 1,
                "mutation_applied_prompt_count": 1,
                "output_changed_count": 0,
                "output_unchanged_count": 1,
                "total_mutation_applied_records": 1,
                "total_active_records": 1,
                "old_new_differ_count": 1,
                "blocker_reasons": [],
                "max_valid_slot_count": 4,
                "min_valid_slot_count": 4,
                "measured_runtime_reduction": False,
                "prompt_results": [
                    {
                        "active_status": "succeeded",
                        "active_output": "x",
                        "active_error": None,
                        "output_changed": False,
                        "mutation_attempted_count": 1,
                        "mutation_applied_count": 1,
                        "active_routing_count": 1,
                        "runtime_behavior_changed_count": 1,
                        "old_new_differ_count": 1,
                        "max_valid_slot_count": 4,
                        "min_valid_slot_count": 4,
                        "blocker_reasons": [],
                        "active_records_written": 1,
                        "measured_runtime_reduction": False,
                    }
                ],
            },
            "mask_last_valid_slot": {
                "active_success_count": 1,
                "mutation_applied_prompt_count": 1,
                "output_changed_count": 1,
                "output_unchanged_count": 0,
                "total_mutation_applied_records": 1,
                "total_active_records": 1,
                "old_new_differ_count": 1,
                "blocker_reasons": [],
                "max_valid_slot_count": 4,
                "min_valid_slot_count": 4,
                "measured_runtime_reduction": False,
                "prompt_results": [
                    {
                        "active_status": "succeeded",
                        "active_output": "y",
                        "active_error": None,
                        "output_changed": True,
                        "mutation_attempted_count": 1,
                        "mutation_applied_count": 1,
                        "active_routing_count": 1,
                        "runtime_behavior_changed_count": 1,
                        "old_new_differ_count": 1,
                        "max_valid_slot_count": 4,
                        "min_valid_slot_count": 4,
                        "blocker_reasons": [],
                        "active_records_written": 1,
                        "measured_runtime_reduction": False,
                    }
                ],
            },
        },
        "best_drift_policy": {"policy": "mask_oldest_valid_slot"},
        "measured_runtime_reduction": False,
        "quality_sanity_passed": True,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "prompt_results": [
            {
                "prompt_index": 0,
                "prompt": "alpha",
                "baseline_status": "succeeded",
                "baseline_output": "baseline",
                "baseline_error": None,
                "measured_runtime_reduction": False,
                "per_policy": {
                    "mask_oldest_valid_slot": {
                        "active_status": "succeeded",
                        "active_output": "x",
                        "active_error": None,
                        "output_changed": False,
                        "mutation_attempted_count": 1,
                        "mutation_applied_count": 1,
                        "active_routing_count": 1,
                        "runtime_behavior_changed_count": 1,
                        "old_new_differ_count": 1,
                        "max_valid_slot_count": 4,
                        "min_valid_slot_count": 4,
                        "blocker_reasons": [],
                        "active_records_written": 1,
                        "measured_runtime_reduction": False,
                    },
                    "mask_last_valid_slot": {
                        "active_status": "succeeded",
                        "active_output": "y",
                        "active_error": None,
                        "output_changed": True,
                        "mutation_attempted_count": 1,
                        "mutation_applied_count": 1,
                        "active_routing_count": 1,
                        "runtime_behavior_changed_count": 1,
                        "old_new_differ_count": 1,
                        "max_valid_slot_count": 4,
                        "min_valid_slot_count": 4,
                        "blocker_reasons": [],
                        "active_records_written": 1,
                        "measured_runtime_reduction": False,
                    },
                },
            }
        ],
    }
    assert validator.validate_report(good)["validation_passed"] is True

    assert validator.validate_report(dict(good, performance_claim_allowed=True))[
        "validation_passed"
    ] is False
    assert validator.validate_report(dict(good, selected_attention_claim_allowed=True))[
        "validation_passed"
    ] is False

    bad = dict(good)
    bad["per_policy"] = dict(good["per_policy"])
    bad["per_policy"]["mask_oldest_valid_slot"] = dict(
        bad["per_policy"]["mask_oldest_valid_slot"],
        mutation_applied_prompt_count=0,
        blocker_reasons=[],
    )
    assert validator.validate_report(bad)["validation_passed"] is False
