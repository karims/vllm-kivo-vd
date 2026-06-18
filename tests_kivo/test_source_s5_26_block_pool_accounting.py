from __future__ import annotations

import argparse
from pathlib import Path

from scripts.kivo_vd.run_source_s5_26_block_pool_accounting_probe import (
    _build_compact_summary,
    _build_llm_kwargs,
    _supports_enable_prefix_caching,
    build_prompts,
    build_summary,
    parse_args as parse_run_args,
)
from scripts.kivo_vd.validate_source_s5_26_block_pool_accounting_probe import (
    parse_args as parse_validate_args,
    validate_summary,
)
from tests_kivo.test_source_s5_25_free_to_pool import _summary_base


class _FakeLLMWithPrefixCaching:
    def __init__(self, *, enable_prefix_caching: bool = True, **kwargs):
        del enable_prefix_caching, kwargs


class _FakeLLMWithoutPrefixCaching:
    def __init__(self, **kwargs):
        del kwargs


def test_build_prompts_repeated_mode_reuses_same_prompt():
    prompts = build_prompts(repeats=5, num_prompts=3, prompt_mode="repeated")
    assert len(prompts) == 3
    assert prompts[0] == prompts[1] == prompts[2]


def test_build_prompts_varied_mode_changes_text():
    prompts = build_prompts(repeats=5, num_prompts=3, prompt_mode="varied")
    assert len(prompts) == 3
    assert len(set(prompts)) == 3
    assert "suffix" in prompts[0].lower()


def test_build_prompts_varied_length_mode_changes_lengths():
    prompts = build_prompts(repeats=16, num_prompts=4, prompt_mode="varied-length")
    lengths = [len(prompt) for prompt in prompts]
    assert len(prompts) == 4
    assert len(set(lengths)) > 1


def test_supports_enable_prefix_caching_detection():
    assert _supports_enable_prefix_caching(_FakeLLMWithPrefixCaching) is True
    assert _supports_enable_prefix_caching(_FakeLLMWithoutPrefixCaching) is False


def test_build_llm_kwargs_disables_prefix_caching_only_when_supported():
    args = argparse.Namespace(
        model="m",
        dtype="auto",
        seed=0,
        gpu_memory_utilization=0.1,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        device="auto",
        disable_prefix_caching=True,
    )
    kwargs_supported, disabled_supported = _build_llm_kwargs(
        args, _FakeLLMWithPrefixCaching
    )
    kwargs_unsupported, disabled_unsupported = _build_llm_kwargs(
        args, _FakeLLMWithoutPrefixCaching
    )
    assert kwargs_supported["enable_prefix_caching"] is False
    assert disabled_supported is True
    assert "enable_prefix_caching" not in kwargs_unsupported
    assert disabled_unsupported is False


def test_build_summary_reports_compact_fields_and_totals():
    args = argparse.Namespace(
        prompt_mode="varied",
        prompt_repeats=8,
        num_prompts=2,
        disable_prefix_caching=True,
    )
    summary = build_summary(
        args=args,
        generation_success=True,
        prompt_char_lengths=[100, 120],
        prompt_token_lengths=[20, 24],
        parent_counters={
            "worker_envelopes_built": 1,
            "scheduler_envelopes_received": 1,
            "core_commands_attempted": 1,
            "manager_mark_demoted_attempted": 1,
            "ownership_remove_succeeded": 2,
            "req_to_blocks_removed": 8,
            "ownership_removed_blocks_total": 8,
            "free_to_pool_succeeded": 2,
            "free_to_pool_calls": 2,
            "free_to_pool_blocks": 8,
            "ownership_remove_invariant_failed": 0,
            "block_pool_free_accounting_observed": 1,
            "block_pool_num_free_blocks_before": 100,
            "block_pool_num_free_blocks_after": 108,
            "block_pool_num_free_blocks_delta": 8,
            "last_freed_block_ids_sample": (10, 11),
        },
        exported_counters=None,
        counter_export_file_found=False,
        counter_export_pid=None,
        prefix_caching_disabled=False,
        prefix_caching_disable_supported=True,
        cuda_before={"cuda_memory_snapshot_observed": False},
        cuda_after={"cuda_memory_snapshot_observed": False},
        wall_time_seconds=1.25,
    )
    assert summary["avg_prompt_tokens"] == 22.0
    assert summary["req_to_blocks_removed_total"] == 8
    assert summary["free_to_pool_blocks_total"] == 8
    assert summary["free_to_pool_calls_total"] == 2
    assert summary["summary"]["free_to_pool_observed"] is True
    assert summary["summary"]["block_pool_accounting_observed"] is True
    assert summary["summary"]["wall_time_seconds"] == 1.25
    assert "worker_envelopes_built" in summary["cumulative_counters"]
    assert "last_freed_block_ids_sample" in summary["last_snapshot_counters"]


def test_compact_summary_uses_total_or_last():
    compact = _build_compact_summary(
        {
            "generation_success": True,
            "prompt_count": 2,
            "avg_prompt_tokens": 10.5,
            "transport_observed": True,
            "ownership_remove_observed": True,
            "free_to_pool_succeeded": 0,
            "free_to_pool_calls": 0,
            "block_pool_free_accounting_observed": 0,
            "req_to_blocks_removed_total": 4,
            "free_to_pool_blocks_total": None,
            "free_to_pool_blocks_last": 2,
            "free_to_pool_calls_total": 0,
            "ownership_remove_invariant_failed": 0,
            "wall_time_seconds": 0.5,
        }
    )
    assert compact["free_to_pool_blocks_total_or_last"] == 2


def test_validator_passes_when_block_pool_accounting_increases():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 1,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [10, 11],
            "last_remaining_block_ids_sample": [12],
            "block_pool_free_accounting_observed": 1,
            "block_pool_free_accounting_increased": 1,
            "block_pool_free_accounting_rejected": 0,
            "block_pool_free_accounting_blocker_reasons": {},
            "block_pool_free_capacity_before": 100,
            "block_pool_free_capacity_after": 102,
            "block_pool_free_capacity_delta": 2,
            "block_pool_num_free_blocks_before": 100,
            "block_pool_num_free_blocks_after": 102,
            "block_pool_num_free_blocks_delta": 2,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True
    assert result["block_pool_free_accounting_observed"] is True


def test_validator_passes_when_accounting_unavailable_with_blocker():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 1,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [10, 11],
            "last_remaining_block_ids_sample": [12],
            "block_pool_free_accounting_observed": 0,
            "block_pool_free_accounting_increased": 0,
            "block_pool_free_accounting_rejected": 1,
            "block_pool_free_accounting_blocker_reasons": {
                "block_pool_free_count_api_unavailable": 1
            },
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True


def test_validator_fails_on_negative_delta():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 1,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [10, 11],
            "last_remaining_block_ids_sample": [12],
            "block_pool_free_accounting_observed": 1,
            "block_pool_free_accounting_increased": 0,
            "block_pool_free_accounting_rejected": 1,
            "block_pool_free_capacity_before": 100,
            "block_pool_free_capacity_after": 99,
            "block_pool_free_capacity_delta": -1,
            "block_pool_num_free_blocks_before": 100,
            "block_pool_num_free_blocks_after": 99,
            "block_pool_num_free_blocks_delta": -1,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_validator_keeps_memory_claim_disallowed():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 1,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [10, 11],
            "last_remaining_block_ids_sample": [12],
            "block_pool_free_accounting_observed": 1,
            "block_pool_free_accounting_increased": 1,
            "block_pool_free_accounting_rejected": 0,
            "block_pool_free_capacity_before": 100,
            "block_pool_free_capacity_after": 102,
            "block_pool_free_capacity_delta": 2,
            "block_pool_num_free_blocks_before": 100,
            "block_pool_num_free_blocks_after": 102,
            "block_pool_num_free_blocks_delta": 2,
            "memory_claim_allowed": False,
            "cuda_memory_snapshot_observed": True,
            "cuda_memory_allocated_before": 1000,
            "cuda_memory_allocated_after": 1000,
            "cuda_memory_reserved_before": 2000,
            "cuda_memory_reserved_after": 2000,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True


def test_validator_surfaces_reuse_probe_success():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 1,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [10, 11],
            "last_remaining_block_ids_sample": [12],
            "block_pool_free_accounting_observed": 1,
            "block_pool_free_accounting_increased": 1,
            "block_pool_free_accounting_rejected": 0,
            "block_pool_free_capacity_before": 100,
            "block_pool_free_capacity_after": 102,
            "block_pool_free_capacity_delta": 2,
            "block_pool_num_free_blocks_before": 100,
            "block_pool_num_free_blocks_after": 102,
            "block_pool_num_free_blocks_delta": 2,
            "reuse_probe_enabled": True,
            "reuse_probe_success": True,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True
    assert result["reuse_probe_success"] is True


def test_validator_fails_on_generation_failure():
    summary = _summary_base()
    summary.update(
        {
            "generation_success": False,
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 1,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [10, 11],
            "last_remaining_block_ids_sample": [12],
            "block_pool_free_accounting_observed": 1,
            "block_pool_free_accounting_increased": 1,
            "block_pool_free_accounting_rejected": 0,
            "block_pool_free_capacity_before": 100,
            "block_pool_free_capacity_after": 102,
            "block_pool_free_capacity_delta": 2,
            "block_pool_num_free_blocks_before": 100,
            "block_pool_num_free_blocks_after": 102,
            "block_pool_num_free_blocks_delta": 2,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_validator_fails_on_double_free_risk():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 0,
            "free_to_pool_rejected": 1,
            "free_to_pool_blocks": 0,
            "free_to_pool_calls": 0,
            "free_to_pool_double_free_prevented": 1,
            "block_pool_free_accounting_observed": 0,
            "block_pool_free_accounting_increased": 0,
            "block_pool_free_accounting_rejected": 1,
            "block_pool_free_accounting_blocker_reasons": {
                "block_pool_free_count_api_unavailable": 1
            },
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_cli_help_includes_expected_args():
    run_args = parse_run_args(
        [
            "--output",
            str(Path("/tmp/out.json")),
            "--prompt-mode",
            "varied",
            "--disable-prefix-caching",
        ]
    )
    validate_args = parse_validate_args(["--input", str(Path("/tmp/out.json"))])
    assert run_args.output.endswith("out.json")
    assert run_args.prompt_mode == "varied"
    assert run_args.disable_prefix_caching is True
    assert validate_args.input.endswith("out.json")
