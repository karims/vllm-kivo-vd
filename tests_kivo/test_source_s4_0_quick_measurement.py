from __future__ import annotations

import itertools
import os
from pathlib import Path
from types import SimpleNamespace

from scripts.kivo_vd import run_source_s4_0_quick_measurement as runner
from scripts.kivo_vd import (
    validate_source_s4_0_quick_measurement as validator,
)


def _fake_generation(args: SimpleNamespace) -> dict[str, object]:
    policy = os.environ.get("KIVO_SOURCE_POLICY")
    if policy == runner.RECENT_WINDOW_MODE:
        text = f"recent::{args.prompt}"
    elif policy == runner.SKETCH_MODE:
        text = f"sketch::{args.prompt}"
    else:
        text = f"baseline::{args.prompt}"
    return {
        "status": "succeeded",
        "output_text": text,
        "prompt_token_length": len(str(args.prompt).split()),
        "generated_token_count": 4,
        "error": None,
    }


def _fake_record_loader(path: str | Path) -> list[dict[str, object]]:
    name = Path(path).name
    measured_runs = len(runner.DEFAULT_PROMPTS) * 3
    if runner.RECENT_WINDOW_MODE in name:
        record = {
            "schema_version": runner.S3_2B_SCHEMA,
            "mutation_attempted": True,
            "mutation_applied": True,
            "active_routing": True,
            "runtime_behavior_changed": False,
            "mutation_blocker_reason": None,
            "original_visible_block_count": 3,
            "selected_block_count": 2,
            "excluded_block_count": 1,
            "theoretical_attention_visible_block_reduction": 1,
            "theoretical_attention_visible_block_reduction_ratio": 0.25,
        }
        if "_prompt_" in name:
            return [record]
        return [record] * measured_runs
    if runner.SKETCH_MODE in name:
        plan = {
            "schema_version": runner.S3_3C_PLAN_SCHEMA,
            "sketch_computed": True,
            "candidate_block_count": 4,
            "selected_block_count": 2,
            "excluded_block_count": 2,
            "sketch_plan_blocker_reason": None,
        }
        metadata = {
            "schema_version": runner.S3_3C_METADATA_SCHEMA,
            "sketch_plan_used": True,
            "mutation_attempted": True,
            "mutation_applied": True,
            "active_routing": True,
            "runtime_behavior_changed": False,
            "mutation_blocker_reason": None,
            "selected_block_count": 2,
            "excluded_block_count": 2,
            "aliased_block_count": 2,
        }
        if "_prompt_" in name:
            return [plan, metadata]
        return [plan, metadata] * measured_runs
    return []


def _fake_timer_factory():
    ticks = itertools.count(0.0, 1.0)
    return lambda: next(ticks)


def _build_args() -> SimpleNamespace:
    return runner._parse_args(
        [
            "--model",
            "gpt2",
            "--max-tokens",
            "32",
            "--repeats",
            "3",
            "--warmup",
            "1",
        ]
    )


def test_parse_args_includes_quick_measurement_fields():
    args = runner._parse_args([])
    assert args.repeats == 3
    assert args.warmup == 1
    assert args.keep_recent_blocks == 1
    assert args.sketch_dim == 8


def test_build_report_computes_mode_and_event_summary_with_fakes():
    report = runner.build_report(
        _build_args(),
        generation_fn=_fake_generation,
        record_loader=_fake_record_loader,
        timer_fn=_fake_timer_factory(),
    )
    assert report["passed"] is True
    assert report["total_prompts"] == 4
    assert report["repeats"] == 3
    assert report["warmup"] == 1
    assert report["baseline_success_count"] == 12
    assert report["recent_window_success_count"] == 12
    assert report["sketch_active_success_count"] == 12
    assert report["output_changed_count"] == 24
    assert report["total_raw_events"] == 36
    assert report["total_s3_2b_events"] == 12
    assert report["total_s3_3c_sketch_plan_events"] == 12
    assert report["total_s3_3c_metadata_alias_events"] == 12
    assert report["measured_runtime_reduction"] is False
    assert report["memory_claim_allowed"] is False
    assert report["quality_claim_allowed"] is False
    assert report["selected_attention_claim_allowed"] is False
    assert report["performance_claim_allowed"] is False
    assert report["baseline_to_recent_window_latency_ratio"] == 1.0
    assert report["baseline_to_sketch_active_latency_ratio"] == 1.0
    assert report["recent_window_to_sketch_active_latency_ratio"] == 1.0
    assert report["per_mode"][runner.BASELINE_MODE]["success_count"] == 12
    assert report["per_mode"][runner.RECENT_WINDOW_MODE]["success_count"] == 12
    assert report["per_mode"][runner.SKETCH_MODE]["success_count"] == 12
    assert report["per_mode"][runner.RECENT_WINDOW_MODE]["event_summary"][
        "total_s3_2b_events"
    ] == 12
    assert report["per_mode"][runner.SKETCH_MODE]["event_summary"][
        "total_s3_3c_sketch_plan_events"
    ] == 12
    assert report["per_mode"][runner.SKETCH_MODE]["event_summary"][
        "total_s3_3c_metadata_alias_events"
    ] == 12
    assert report["mode_event_jsonl"][runner.BASELINE_MODE] is None
    assert report["mode_event_jsonl"][runner.RECENT_WINDOW_MODE]
    assert report["mode_event_jsonl"][runner.SKETCH_MODE]


def test_validate_report_passes_for_good_report():
    report = runner.build_report(
        _build_args(),
        generation_fn=_fake_generation,
        record_loader=_fake_record_loader,
        timer_fn=_fake_timer_factory(),
    )
    raw_events = [
        event
        for run in report["run_records"]
        for event in run["event_records"]
    ]
    result = validator.validate_report(report, raw_events)
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 36
    assert result["total_s3_2b_events"] == 12
    assert result["total_s3_3c_sketch_plan_events"] == 12
    assert result["total_s3_3c_metadata_alias_events"] == 12


def test_validate_report_rejects_claim_fields_and_missing_events():
    report = runner.build_report(
        _build_args(),
        generation_fn=_fake_generation,
        record_loader=_fake_record_loader,
        timer_fn=_fake_timer_factory(),
    )
    bad_claim = dict(report)
    bad_claim["performance_claim_allowed"] = True
    bad_claim["measured_runtime_reduction"] = True
    result = validator.validate_report(bad_claim, [])
    assert result["validation_passed"] is False
    assert any("performance_claim_allowed must be false" in err for err in result["errors"])
    assert any("measured_runtime_reduction must be false" in err for err in result["errors"])
