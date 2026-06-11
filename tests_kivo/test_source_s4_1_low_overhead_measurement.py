from __future__ import annotations

import itertools
import os
from pathlib import Path
from types import SimpleNamespace

from scripts.kivo_vd import run_source_s4_1_low_overhead_measurement as runner
from scripts.kivo_vd import (
    validate_source_s4_1_low_overhead_measurement as validator,
)
from vllm.v1.worker import kivo_runtime_counters as counters


def _fake_generation(args: SimpleNamespace) -> dict[str, object]:
    mode = os.environ.get("KIVO_SOURCE_POLICY")
    if mode == runner.RECENT_WINDOW_POLICY:
        text = f"recent::{args.prompt}"
    elif mode == runner.SKETCH_ACTIVE_POLICY:
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
    if runner.RECENT_WINDOW_VERBOSE_MODE in name:
        return [
            {
                "schema_version": runner.S3_2B_SCHEMA,
                "mutation_attempted": True,
                "mutation_applied": True,
                "active_routing": True,
                "runtime_behavior_changed": False,
                "mutation_blocker_reason": None,
                "original_visible_block_count": 3,
                "visible_block_count_estimate": 3,
                "selected_block_count": 2,
                "excluded_block_count": 1,
                "theoretical_attention_visible_block_reduction_ratio": 0.25,
            }
        ]
    if runner.SKETCH_ACTIVE_VERBOSE_MODE in name:
        return [
            {
                "schema_version": runner.S3_3C_PLAN_SCHEMA,
                "sketch_computed": True,
                "candidate_block_count": 4,
                "selected_block_count": 2,
                "excluded_block_count": 2,
                "sketch_plan_blocker_reason": None,
            },
            {
                "schema_version": runner.S3_3C_METADATA_SCHEMA,
                "sketch_plan_used": True,
                "mutation_attempted": True,
                "mutation_applied": True,
                "active_routing": True,
                "runtime_behavior_changed": False,
                "mutation_blocker_reason": None,
                "visible_block_count_estimate": 3,
                "selected_block_count": 2,
                "excluded_block_count": 2,
                "aliased_block_count": 2,
            },
        ]
    return []


def _fake_counter_loader() -> dict[str, dict[str, object]]:
    record_mode = os.environ.get("KIVO_SOURCE_RECORD_MODE")
    policy = os.environ.get("KIVO_SOURCE_POLICY")
    if record_mode != "counters":
        return {}
    if policy == runner.RECENT_WINDOW_POLICY:
        return {
            runner.S3_2B_SCHEMA: {
                "event_count": 1,
                "recent_window_event_count": 1,
                "mutation_attempted_count": 1,
                "mutation_applied_count": 1,
                "active_routing_count": 1,
                "runtime_behavior_changed_count": 0,
                "blocker_count": 0,
                "blocker_reason_counts": {},
                "max_visible_block_count": 3,
                "max_original_visible_block_count": 3,
                "max_selected_block_count": 2,
                "max_excluded_block_count": 1,
                "max_unselected_block_count": 1,
                "max_theoretical_attention_visible_block_reduction_ratio": 0.25,
            }
        }
    if policy == runner.SKETCH_ACTIVE_POLICY:
        return {
            runner.S3_3C_PLAN_SCHEMA: {
                "event_count": 1,
                "sketch_computed_count": 1,
                "sketch_blocked_count": 0,
                "blocker_count": 0,
                "blocker_reason_counts": {},
                "max_candidate_block_count": 4,
                "max_selected_block_count": 2,
                "max_excluded_block_count": 2,
                "max_unselected_block_count": 2,
            },
            runner.S3_3C_METADATA_SCHEMA: {
                "event_count": 1,
                "metadata_alias_count": 1,
                "sketch_plan_used_count": 1,
                "mutation_attempted_count": 1,
                "mutation_applied_count": 1,
                "active_routing_count": 1,
                "runtime_behavior_changed_count": 0,
                "blocker_count": 0,
                "blocker_reason_counts": {},
                "max_visible_block_count": 3,
                "max_original_visible_block_count": 3,
                "max_selected_block_count": 2,
                "max_excluded_block_count": 2,
                "max_unselected_block_count": 2,
                "max_aliased_block_count": 2,
                "max_theoretical_attention_visible_block_reduction_ratio": 0.25,
            },
        }
    return {}


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
            "1",
            "--warmup",
            "1",
        ]
    )


def test_parse_args_includes_low_overhead_fields():
    args = runner._parse_args([])
    assert args.repeats == 3
    assert args.warmup == 1
    assert args.keep_recent_blocks == 1
    assert args.sketch_seed == 123
    assert args.force_inproc_engine_core is True


def test_runtime_counters_only_record_in_counters_mode(monkeypatch):
    counters.get_and_reset_counters()
    monkeypatch.setenv("KIVO_SOURCE_RECORD_MODE", "events")
    counters.record_counter_event(
        {
            "schema_version": runner.S3_2B_SCHEMA,
            "event_count": 1,
        }
    )
    assert counters.snapshot_counters() == {}

    monkeypatch.setenv("KIVO_SOURCE_RECORD_MODE", "counters")
    counters.record_counter_event(
        {
            "schema_version": runner.S3_2B_SCHEMA,
            "mutation_attempted": True,
            "mutation_applied": True,
            "active_routing": True,
            "runtime_behavior_changed": False,
            "selected_block_count": 2,
            "excluded_block_count": 1,
            "visible_block_count_estimate": 3,
            "mutation_blocker_reason": None,
        }
    )
    snapshot = counters.get_and_reset_counters()
    assert runner.S3_2B_SCHEMA in snapshot
    assert snapshot[runner.S3_2B_SCHEMA]["mutation_applied_count"] == 1


def test_build_report_computes_verbose_and_counter_summaries_with_fakes():
    report = runner.build_report(
        _build_args(),
        generation_fn=_fake_generation,
        record_loader=_fake_record_loader,
        counter_loader=_fake_counter_loader,
        timer_fn=_fake_timer_factory(),
    )
    assert report["passed"] is True
    assert report["total_prompts"] == 4
    assert report["repeats"] == 1
    assert report["warmup"] == 1
    assert report["baseline_success_count"] == 4
    assert report["recent_window_verbose_success_count"] == 4
    assert report["recent_window_counters_success_count"] == 4
    assert report["sketch_active_verbose_success_count"] == 4
    assert report["sketch_active_counters_success_count"] == 4
    assert report["verbose_event_record_count"] == 12
    assert report["counter_event_count"] == 12
    assert report["recent_window_verbose_vs_baseline_latency_ratio"] == 1.0
    assert report["recent_window_counters_vs_baseline_latency_ratio"] == 1.0
    assert report["recent_window_counters_vs_verbose_latency_ratio"] == 1.0
    assert report["sketch_active_verbose_vs_baseline_latency_ratio"] == 1.0
    assert report["sketch_active_counters_vs_baseline_latency_ratio"] == 1.0
    assert report["sketch_active_counters_vs_verbose_latency_ratio"] == 1.0
    assert report["measured_runtime_reduction"] is False
    assert report["memory_claim_allowed"] is False
    assert report["quality_claim_allowed"] is False
    assert report["selected_attention_claim_allowed"] is False
    assert report["performance_claim_allowed"] is False
    assert report["per_mode"][runner.BASELINE_MODE]["success_count"] == 4
    assert report["per_mode"][runner.RECENT_WINDOW_VERBOSE_MODE][
        "verbose_event_record_count"
    ] == 4
    assert report["per_mode"][runner.RECENT_WINDOW_COUNTERS_MODE][
        "counter_event_count"
    ] == 4
    assert report["per_mode"][runner.SKETCH_ACTIVE_VERBOSE_MODE][
        "verbose_event_record_count"
    ] == 8
    assert report["per_mode"][runner.SKETCH_ACTIVE_COUNTERS_MODE][
        "counter_event_count"
    ] == 8
    assert report["per_mode"][runner.SKETCH_ACTIVE_COUNTERS_MODE][
        "counter_summary"
    ]["sketch_computed_count"] == 4
    assert report["per_mode"][runner.RECENT_WINDOW_COUNTERS_MODE][
        "counter_summary"
    ]["mutation_applied_count"] == 4


def test_validate_report_passes_for_good_report():
    report = runner.build_report(
        _build_args(),
        generation_fn=_fake_generation,
        record_loader=_fake_record_loader,
        counter_loader=_fake_counter_loader,
        timer_fn=_fake_timer_factory(),
    )
    result = validator.validate_report(report)
    assert result["validation_passed"] is True
    assert result["baseline_success_count"] == 4
    assert result["recent_window_counters_success_count"] == 4
    assert result["sketch_active_counters_success_count"] == 4


def test_validate_report_rejects_claim_fields():
    report = runner.build_report(
        _build_args(),
        generation_fn=_fake_generation,
        record_loader=_fake_record_loader,
        counter_loader=_fake_counter_loader,
        timer_fn=_fake_timer_factory(),
    )
    bad_report = dict(report)
    bad_report["performance_claim_allowed"] = True
    bad_report["selected_attention_claim_allowed"] = True
    result = validator.validate_report(bad_report)
    assert result["validation_passed"] is False
    assert any("performance_claim_allowed must be false" in err for err in result["errors"])
    assert any(
        "selected_attention_claim_allowed must be false" in err
        for err in result["errors"]
    )
