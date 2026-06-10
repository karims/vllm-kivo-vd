# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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


def _sample_prompt_result(
    *,
    prompt_index: int = 0,
    prompt: str = "prompt",
    baseline_status: str = "succeeded",
    active_status: str = "succeeded",
    output_changed: bool = False,
    mutation_attempted_count: int = 1,
    mutation_applied_count: int = 1,
    active_routing_count: int = 1,
    runtime_behavior_changed_count: int = 1,
    max_valid_slot_count: int = 12,
    min_valid_slot_count: int = 12,
    old_new_differ_count: int = 1,
    blocker_reasons: list[str] | None = None,
) -> dict:
    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "baseline_status": baseline_status,
        "active_status": active_status,
        "baseline_output": "baseline",
        "active_output": "active",
        "output_changed": output_changed,
        "active_error": None,
        "mutation_attempted_count": mutation_attempted_count,
        "mutation_applied_count": mutation_applied_count,
        "active_routing_count": active_routing_count,
        "runtime_behavior_changed_count": runtime_behavior_changed_count,
        "max_valid_slot_count": max_valid_slot_count,
        "min_valid_slot_count": min_valid_slot_count,
        "old_new_differ_count": old_new_differ_count,
        "blocker_reasons": blocker_reasons or [],
        "baseline_records_written": 1,
        "active_records_written": 1,
        "measured_runtime_reduction": False,
    }


def test_aggregate_quality_sanity_counts():
    module = _load_script(
        "run_source_s1_2_quality_sanity.py",
        "source_s1_2_quality_sanity_test_1",
    )

    args = module._parse_args([
        "--prompt",
        "prompt one",
        "--prompt",
        "prompt two",
    ])

    def probe_runner(probe_args):
        index = int(probe_args.prompt.endswith("two"))
        return {
            "baseline_status": "succeeded",
            "active_status": "succeeded",
            "baseline_output": "baseline",
            "active_output": "active" if index == 0 else "active2",
            "output_changed": index == 1,
            "active_error": None,
            "baseline_records_written": 1,
            "active_records_written": 2,
        }

    def load_records(path):
        if "prompt_00" in str(path):
            return [
                {
                    "mutation_attempted": True,
                    "mutation_applied": True,
                    "active_routing": True,
                    "runtime_behavior_changed": True,
                    "valid_slot_count": 12,
                    "old_new_differ": True,
                    "mutation_blocker_reason": None,
                }
            ]
        return [
            {
                "mutation_attempted": True,
                "mutation_applied": False,
                "active_routing": False,
                "runtime_behavior_changed": False,
                "valid_slot_count": 11,
                "old_new_differ": False,
                "mutation_blocker_reason": "fewer than two valid slot entries",
            },
            {
                "mutation_attempted": False,
                "mutation_applied": False,
                "active_routing": False,
                "runtime_behavior_changed": False,
                "valid_slot_count": 11,
                "old_new_differ": False,
                "mutation_blocker_reason": None,
            },
        ]

    module._load_records = load_records
    report = module.build_report(args, probe_runner=probe_runner)

    assert report["total_prompts"] == 2
    assert report["baseline_success_count"] == 2
    assert report["active_success_count"] == 2
    assert report["mutation_applied_prompt_count"] == 1
    assert report["output_changed_count"] == 1
    assert report["output_unchanged_count"] == 1
    assert report["total_mutation_applied_records"] == 1
    assert report["total_active_records"] == 3
    assert report["quality_sanity_passed"] is True
    assert report["selected_attention_claim_allowed"] is False
    assert report["performance_claim_allowed"] is False
    assert report["prompt_results"][0]["blocker_reasons"] == []
    assert "fewer than two valid slot entries" in report["prompt_results"][1][
        "blocker_reasons"
    ]


def test_validator_accepts_good_report_and_rejects_claims():
    module = _load_script(
        "validate_source_s1_2_quality_sanity.py",
        "source_s1_2_quality_sanity_validator_test",
    )

    good = {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "active_success_count": 1,
        "mutation_applied_prompt_count": 1,
        "output_changed_count": 1,
        "output_unchanged_count": 0,
        "total_mutation_applied_records": 1,
        "total_active_records": 1,
        "measured_runtime_reduction": False,
        "quality_sanity_passed": True,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "prompt_results": [
            _sample_prompt_result(
                output_changed=True,
                blocker_reasons=[],
            )
        ],
    }
    report = module.validate_report(good)
    assert report["validation_passed"] is True

    bad_perf = dict(good, performance_claim_allowed=True)
    assert module.validate_report(bad_perf)["validation_passed"] is False

    bad_selected = dict(good, selected_attention_claim_allowed=True)
    assert module.validate_report(bad_selected)["validation_passed"] is False

    bad_no_mutation = dict(
        good,
        total_mutation_applied_records=0,
        mutation_applied_prompt_count=0,
        prompt_results=[
            _sample_prompt_result(
                mutation_applied_count=0,
                active_routing_count=0,
                runtime_behavior_changed_count=0,
                old_new_differ_count=0,
                blocker_reasons=["blocked"],
            )
        ],
    )
    assert module.validate_report(bad_no_mutation)["validation_passed"] is False


def test_validator_requires_blockers_when_no_mutation():
    module = _load_script(
        "validate_source_s1_2_quality_sanity.py",
        "source_s1_2_quality_sanity_validator_test_2",
    )
    report = {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "active_success_count": 1,
        "mutation_applied_prompt_count": 0,
        "output_changed_count": 0,
        "output_unchanged_count": 1,
        "total_mutation_applied_records": 1,
        "total_active_records": 1,
        "measured_runtime_reduction": False,
        "quality_sanity_passed": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "prompt_results": [
            _sample_prompt_result(
                mutation_applied_count=0,
                active_routing_count=0,
                runtime_behavior_changed_count=0,
                old_new_differ_count=0,
                blocker_reasons=[],
            )
        ],
    }
    result = module.validate_report(report)
    assert result["validation_passed"] is False
    assert any("blocker_reasons" in err for err in result["errors"])
