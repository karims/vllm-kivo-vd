from __future__ import annotations

from pathlib import Path

from scripts.kivo_vd.run_source_s5_26_block_pool_accounting_probe import (
    parse_args as parse_run_args,
)
from scripts.kivo_vd.validate_source_s5_26_block_pool_accounting_probe import (
    parse_args as parse_validate_args,
    validate_summary,
)
from tests_kivo.test_source_s5_25_free_to_pool import _summary_base


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
    run_args = parse_run_args(["--output", str(Path("/tmp/out.json"))])
    validate_args = parse_validate_args(["--input", str(Path("/tmp/out.json"))])
    assert run_args.output.endswith("out.json")
    assert validate_args.input.endswith("out.json")
