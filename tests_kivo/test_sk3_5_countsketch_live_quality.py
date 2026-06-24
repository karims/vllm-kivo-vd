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


def _prompt_result(
    module,
    *,
    prompt_name: str,
    success: bool = True,
    contains_expected_terms: bool = True,
    degeneration_detected: bool = False,
    sketch_backend: str | None = "countsketch",
    sketch_build_attempted: int = 4,
    sketch_build_succeeded: int = 4,
    sketch_build_failed: int = 0,
    freed_after_sketch_blocks_total: int = 2,
    blocks_freed_total: int = 2,
    invariants_clean: bool = True,
    prompt_token_count: int = 1024,
    estimated_block_count: int = 64,
    insufficient_block_pressure: bool = False,
) -> dict:
    return {
        "prompt_name": prompt_name,
        "success": success,
        "output_text": "Step one\nStep two",
        "output_length": 16,
        "prompt_token_count": prompt_token_count,
        "estimated_block_count": estimated_block_count,
        "expected_terms": ["term"],
        "expected_terms_found": ["term"] if contains_expected_terms else [],
        "expected_terms_missing": [] if contains_expected_terms else ["term"],
        "contains_expected_terms": contains_expected_terms,
        "degeneration_detected": degeneration_detected,
        "latency_seconds": 1.0,
        "insufficient_block_pressure": insufficient_block_pressure,
        "counter_summary": {
            "sketch_backend": sketch_backend,
            "last_policy": module.COUNTSKETCH_POLICY,
            "scoring_source": module.SCORING_SOURCE,
            "sketch_build_attempted": sketch_build_attempted,
            "sketch_build_succeeded": sketch_build_succeeded,
            "sketch_build_failed": sketch_build_failed,
            "sketched_blocks_total": sketch_build_succeeded,
            "sketch_bytes_total": 1024,
            "freed_after_sketch_blocks_total": freed_after_sketch_blocks_total,
            "blocks_freed_total": blocks_freed_total,
            "invariants_clean": invariants_clean,
        },
        "trace_summary": {
            "visible_before_count_max": 44,
            "visible_after_count_min": 24,
            "visible_before_count_last": 44,
            "visible_after_count_last": 24,
            "candidate_demote_count_total": 20,
            "retention_ratio_avg": 24 / 44,
            "retention_ratio_min": 24 / 44,
            "retention_ratio_last": 24 / 44,
        },
    }


def test_countsketch_modes_force_countsketch_backend_env() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "/tmp/out.json"])
    counter_file, trace_file = module._paths_for_mode_prompt(
        args=args,
        mode=module.DEFAULT_MODE,
        prompt_name="factual_recall_long",
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
        prompt_name="factual_recall_long",
    )

    env = module._mode_env(
        module.SKETCH_BUILD_ONLY_MODE,
        args=args,
        counter_file=counter_file,
        trace_file=trace_file,
    )

    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION"] == "plan_only"
    assert "KIVO_KV_FREE_TO_POOL_ENABLE" not in env


def test_trace_output_is_a_file_path_not_directory(tmp_path: Path) -> None:
    module = _load_module()
    args = module.parse_args(
        [
            "--output",
            str(tmp_path / "report.json"),
            "--trace-output",
            str(tmp_path / "trace.jsonl"),
        ]
    )

    aggregate = module._aggregate_trace_output_path(args)
    counter_file, trace_file = module._paths_for_mode_prompt(
        args=args,
        mode=module.DEFAULT_MODE,
        prompt_name="factual_recall_long",
    )

    assert aggregate.name == "trace.jsonl"
    assert not aggregate.exists()
    assert trace_file.endswith(".trace.jsonl")
    assert aggregate != Path(trace_file)


def test_baseline_gate_fails_when_baseline_is_empty() -> None:
    module = _load_module()
    gate = module._baseline_gate(
        [
            _prompt_result(
                module,
                prompt_name="factual_recall_long",
                success=False,
                contains_expected_terms=False,
                degeneration_detected=True,
                sketch_backend=None,
                sketch_build_attempted=0,
                sketch_build_succeeded=0,
                freed_after_sketch_blocks_total=0,
                blocks_freed_total=0,
            ),
            _prompt_result(module, prompt_name="code_context_long"),
            _prompt_result(module, prompt_name="summarization_long"),
            _prompt_result(module, prompt_name="instruction_following_long"),
        ]
    )

    assert gate["passed"] is False
    assert gate["reason"] == "baseline_failed"


def test_baseline_gate_passes_with_expected_terms() -> None:
    module = _load_module()
    gate = module._baseline_gate(
        [
            _prompt_result(module, prompt_name="factual_recall_long"),
            _prompt_result(module, prompt_name="code_context_long"),
            _prompt_result(module, prompt_name="summarization_long"),
            _prompt_result(module, prompt_name="instruction_following_long"),
        ]
    )

    assert gate["passed"] is True
    assert gate["reason"] is None


def test_countsketch_not_exercised_warning_when_backend_missing() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "/tmp/out.json"])
    summary = module._aggregate_mode(
        module.DEFAULT_MODE,
        [
            _prompt_result(
                module,
                prompt_name="factual_recall_long",
                sketch_backend=None,
                sketch_build_attempted=0,
                sketch_build_succeeded=0,
                freed_after_sketch_blocks_total=0,
                blocks_freed_total=0,
            )
        ],
        args,
    )

    assert summary["backend_pass"] is False
    assert "countsketch_backend_not_observed" in summary["warnings"]
    assert summary["reason"] == "countsketch_not_exercised"


def test_insufficient_block_pressure_warning() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "/tmp/out.json"])
    summary = module._aggregate_mode(
        module.DEFAULT_MODE,
        [
            _prompt_result(
                module,
                prompt_name="factual_recall_long",
                prompt_token_count=128,
                estimated_block_count=4,
                insufficient_block_pressure=True,
            )
        ],
        args,
    )

    assert summary["insufficient_block_pressure"] is True
    assert "insufficient_block_pressure" in summary["warnings"]


def test_verdict_invalid_when_baseline_fails() -> None:
    module = _load_module()
    verdict = module._verdict(
        {
            module.DEFAULT_MODE: {
                "mode": module.DEFAULT_MODE,
                "backend_pass": True,
                "invariants_pass": True,
                "degeneration_count": 0,
                "expected_terms_pass_count": 2,
                "insufficient_block_pressure": False,
                "freed_after_sketch_blocks_total": 4,
                "block_savings_pass": True,
                "retention_ratio_avg": 0.5,
            }
        },
        baseline_gate={"passed": False, "reason": "baseline_failed"},
    )

    assert verdict["verdict_valid"] is False
    assert verdict["reason"] == "baseline_failed"


def test_mode_viable_only_when_backend_savings_quality_and_invariants_pass() -> None:
    module = _load_module()
    mode_summaries = {
        module.SKETCH_BUILD_ONLY_MODE: {
            "mode": module.SKETCH_BUILD_ONLY_MODE,
            "backend_pass": True,
            "invariants_pass": True,
            "degeneration_count": 0,
            "expected_terms_pass_count": 2,
            "insufficient_block_pressure": False,
            "freed_after_sketch_blocks_total": 0,
            "block_savings_pass": False,
            "retention_ratio_avg": 1.0,
            "quality_pass_count": 4,
        },
        module.DEFAULT_MODE: {
            "mode": module.DEFAULT_MODE,
            "backend_pass": True,
            "invariants_pass": True,
            "degeneration_count": 0,
            "expected_terms_pass_count": 2,
            "insufficient_block_pressure": False,
            "freed_after_sketch_blocks_total": 8,
            "block_savings_pass": True,
            "retention_ratio_avg": 0.5,
            "quality_pass_count": 4,
        },
    }

    verdict = module._verdict(
        mode_summaries,
        baseline_gate={"passed": True, "reason": None},
    )

    assert verdict["verdict_valid"] is True
    assert verdict["whether_countsketch_live_baseline_is_viable"] is True
    assert verdict["recommended_next_mode"] == module.DEFAULT_MODE


def test_output_schema_includes_prompt_token_and_block_counts() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "/tmp/out.json"])
    summary = module._aggregate_mode(
        module.DEFAULT_MODE,
        [
            _prompt_result(
                module,
                prompt_name="factual_recall_long",
                prompt_token_count=1200,
                estimated_block_count=75,
            )
        ],
        args,
    )

    assert summary["prompt_token_count_max"] == 1200
    assert summary["estimated_block_count_max"] == 75
