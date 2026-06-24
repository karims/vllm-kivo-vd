# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "run_sk3_5_countsketch_live_quality.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_sk3_5_countsketch_live_quality",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_countsketch_backend_selected_from_cli_env() -> None:
    module = _load_module()
    args = module.parse_args(
        [
            "--output",
            "/tmp/out.json",
            "--sketch-backend",
            "countsketch",
        ]
    )
    counter_file, trace_file = module._paths_for_mode_prompt(
        args=args,
        mode=module.DEFAULT_MODE,
        prompt_name="factual_recall",
    )

    env = module._mode_env(
        module.DEFAULT_MODE,
        args=args,
        counter_file=counter_file,
        trace_file=trace_file,
    )

    assert env["KIVO_KV_SKETCH_BACKEND"] == "countsketch"


def test_sketch_build_only_countsketch_does_not_enable_freeing() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "/tmp/out.json"])
    counter_file, trace_file = module._paths_for_mode_prompt(
        args=args,
        mode=module.SKETCH_BUILD_ONLY_MODE,
        prompt_name="factual_recall",
    )

    env = module._mode_env(
        module.SKETCH_BUILD_ONLY_MODE,
        args=args,
        counter_file=counter_file,
        trace_file=trace_file,
    )

    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION"] == "plan_only"
    assert "KIVO_KV_FREE_TO_POOL_ENABLE" not in env


def test_countsketch_span_default_uses_conservative_defaults() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "/tmp/out.json"])
    counter_file, trace_file = module._paths_for_mode_prompt(
        args=args,
        mode=module.DEFAULT_MODE,
        prompt_name="factual_recall",
    )

    env = module._mode_env(
        module.DEFAULT_MODE,
        args=args,
        counter_file=counter_file,
        trace_file=trace_file,
    )

    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS"] == "16"
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_TOPK"] == "8"
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_SPAN_RADIUS"] == "1"
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS"] == "32"


def test_retention_ratio_calculation() -> None:
    module = _load_module()

    assert module.retention_ratio_from_counts(16, 32) == 0.5
    assert module.retention_ratio_from_counts(1, 0) is None


def test_contiguous_span_and_gap_diagnostics() -> None:
    module = _load_module()

    contiguous = module.span_diagnostics([10, 11, 12])
    sparse = module.span_diagnostics([10, 11, 20, 21])

    assert contiguous["contiguous_span_count"] == 1
    assert contiguous["max_gap_between_kept_blocks"] == 0
    assert contiguous["kept_blocks_contiguous"] is True
    assert sparse["contiguous_span_count"] == 2
    assert sparse["max_gap_between_kept_blocks"] == 8
    assert sparse["kept_blocks_contiguous"] is False


def test_expected_term_checker() -> None:
    module = _load_module()

    report = module.expected_term_report(
        "Maya Chen moved the project to Toronto in 2020.",
        ["maya chen", "toronto", "2020"],
    )

    assert report["contains_expected_terms"] is True
    assert report["expected_terms_missing"] == []


def test_degeneration_detector_catches_repeated_numeric_junk() -> None:
    module = _load_module()

    repeated = module.degeneration_report("word word word word word word word")
    numeric = module.degeneration_report("1234567890 1234567890")

    assert repeated["degeneration_detected"] is True
    assert numeric["degeneration_detected"] is True


def test_output_schema_includes_required_sections() -> None:
    module = _load_module()
    prompt_result = {
        "success": True,
        "expected_terms": ["maya chen"],
        "contains_expected_terms": True,
        "degeneration_detected": False,
        "latency_seconds": 1.0,
        "counter_summary": {
            "sketch_backend": "countsketch",
            "last_policy": module.COUNTSKETCH_POLICY,
            "scoring_source": module.SCORING_SOURCE,
            "sketch_build_attempted": 4,
            "sketch_build_succeeded": 4,
            "sketch_build_failed": 0,
            "sketched_blocks_total": 4,
            "sketch_bytes_total": 1024,
            "freed_after_sketch_blocks_total": 2,
            "blocks_freed_total": 2,
            "invariants_clean": True,
        },
        "trace_summary": {
            "visible_before_count_max": 10,
            "visible_after_count_min": 6,
            "visible_before_count_last": 10,
            "visible_after_count_last": 6,
            "candidate_demote_count_total": 4,
            "retention_ratio_avg": 0.6,
            "retention_ratio_min": 0.6,
            "retention_ratio_last": 0.6,
        },
    }

    summary = module._aggregate_mode(
        module.DEFAULT_MODE,
        [prompt_result],
        module.parse_args(["--output", "/tmp/out.json"]),
    )

    assert summary["backend_pass"] is True
    assert summary["block_savings_pass"] is True
    assert summary["scoring_source"] == module.SCORING_SOURCE
    assert summary["estimated_sketch_overhead_bytes"] == 1024
    assert summary["quality_pass_count"] == 1
