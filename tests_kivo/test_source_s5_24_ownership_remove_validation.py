from __future__ import annotations

from pathlib import Path

from scripts.kivo_vd.run_source_s5_24_ownership_remove_validation import (
    parse_args as parse_run_args,
)
from scripts.kivo_vd.validate_source_s5_24_ownership_remove_validation import (
    parse_args as parse_validate_args,
    validate_summary,
)
from tests_kivo.test_source_s5_23_ownership_remove import (
    _manager,
    _remove_config,
)
from vllm.v1.core.kivo_demotion_counters import (
    get_kivo_demotion_counters_snapshot,
    reset_kivo_demotion_counters,
)


def _summary_base() -> dict[str, object]:
    return {
        "generation_success": True,
        "prompt_count": 1,
        "transport_observed": True,
        "counter_export_file_found": True,
        "free_to_pool_calls": 0,
        "memory_claim_allowed": False,
        "free_to_pool_claim_allowed": False,
        "ownership_removal_claim_allowed": False,
        "performance_claim_allowed": False,
        "quality_claim_allowed": False,
        "live_kv_free_enabled": False,
        "counters": {},
    }


def test_validator_passes_disabled_noop_case():
    summary = _summary_base()
    summary.update(
        {
            "transport_observed": False,
            "counter_export_file_found": False,
            "ownership_remove_succeeded": 0,
            "req_to_blocks_removed": 0,
            "ownership_remaining_blocks_last": 0,
            "ownership_remove_invariant_failed": 0,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True
    assert result["ownership_remove_observed"] is False


def test_validator_passes_successful_removal_case():
    summary = _summary_base()
    summary.update(
        {
            "demoted_blocks_marked": 4,
            "ownership_remove_attempted": 2,
            "ownership_remove_succeeded": 2,
            "req_to_blocks_removed": 4,
            "ownership_remaining_blocks_last": 1,
            "ownership_remove_invariant_failed": 0,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True
    assert result["ownership_remove_observed"] is True


def test_validator_fails_if_no_prompt_removes():
    summary = _summary_base()
    summary.update(
        {
            "ownership_remove_attempted": 0,
            "ownership_remove_succeeded": 0,
            "req_to_blocks_removed": 1,
            "demoted_blocks_marked": 0,
            "ownership_remaining_blocks_last": 1,
            "ownership_remove_invariant_failed": 0,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_validator_fails_if_remaining_blocks_empty():
    summary = _summary_base()
    summary.update(
        {
            "demoted_blocks_marked": 2,
            "ownership_remove_attempted": 1,
            "ownership_remove_succeeded": 1,
            "req_to_blocks_removed": 2,
            "ownership_remaining_blocks_last": 0,
            "ownership_remove_invariant_failed": 0,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_validator_fails_if_free_to_pool_positive():
    summary = _summary_base()
    summary.update(
        {
            "demoted_blocks_marked": 2,
            "ownership_remove_attempted": 1,
            "ownership_remove_succeeded": 1,
            "req_to_blocks_removed": 2,
            "ownership_remaining_blocks_last": 1,
            "ownership_remove_invariant_failed": 0,
            "free_to_pool_calls": 1,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_validator_fails_if_performance_claim_allowed():
    summary = _summary_base()
    summary.update(
        {
            "performance_claim_allowed": True,
            "ownership_remove_succeeded": 0,
            "req_to_blocks_removed": 0,
            "ownership_remaining_blocks_last": 0,
            "ownership_remove_invariant_failed": 0,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_removed_ids_absent_after_helper_and_bookkeeping_cleared(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    manager = _manager()
    manager.kivo_req_to_demoted_block_ids["req0"] = {10, 11}

    result = manager.remove_kivo_marked_demoted_blocks_if_safe(
        "req0",
        config=_remove_config(),
    )
    snapshot = get_kivo_demotion_counters_snapshot()

    assert result.accepted is True
    assert 10 not in manager.get_request_block_ids_for_kivo("req0")
    assert 11 not in manager.get_request_block_ids_for_kivo("req0")
    assert manager.get_kivo_demoted_block_ids("req0") == ()
    assert snapshot["ownership_remove_invariant_checked"] == 1
    assert snapshot["ownership_remove_invariant_failed"] == 0
    assert snapshot["ownership_demoted_bookkeeping_cleared"] == 1
    assert snapshot["ownership_removed_absent_after"] == 1
    assert snapshot["ownership_remaining_nonempty"] == 1


def test_rejects_would_remove_all_and_records_failure_counter(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    manager = _manager()
    manager.kivo_req_to_demoted_block_ids["req0"] = {10, 11, 12, 13}

    result = manager.remove_kivo_marked_demoted_blocks_if_safe(
        "req0",
        config=_remove_config(),
    )
    snapshot = get_kivo_demotion_counters_snapshot()

    assert result.accepted is False
    assert result.blocker_reasons["would_remove_all_owned_blocks"] == 1
    assert snapshot["ownership_remove_rejected"] == 1


def test_cli_help_includes_expected_args():
    run_args = parse_run_args(["--output", str(Path("/tmp/out.json"))])
    validate_args = parse_validate_args(["--input", str(Path("/tmp/out.json"))])
    assert run_args.output.endswith("out.json")
    assert validate_args.input.endswith("out.json")
