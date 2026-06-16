from __future__ import annotations

from pathlib import Path

from scripts.kivo_vd.run_source_s5_18_transport_counters import (
    build_summary,
    parse_args as parse_run_args,
)
from scripts.kivo_vd.validate_source_s5_18_transport_counters import (
    parse_args as parse_validate_args,
    validate_summary,
)
from vllm.v1.core.kivo_demotion_counters import (
    add_kivo_demotion_blocker_reasons,
    get_kivo_demotion_counters_snapshot,
    increment_kivo_demotion_counter,
    reset_kivo_demotion_counters,
)


def test_counters_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", raising=False)
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("worker_envelopes_built")
    assert get_kivo_demotion_counters_snapshot()["worker_envelopes_built"] == 0


def test_reset_snapshot_works(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("worker_envelopes_built")
    reset_kivo_demotion_counters()
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["worker_envelopes_built"] == 0
    assert snapshot["blocker_reasons"] == {}


def test_worker_envelope_built_increments_when_enabled(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("worker_envelopes_built")
    assert get_kivo_demotion_counters_snapshot()["worker_envelopes_built"] == 1


def test_scheduler_received_increments_when_enabled(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("scheduler_envelopes_received", 2)
    assert (
        get_kivo_demotion_counters_snapshot()["scheduler_envelopes_received"] == 2
    )


def test_core_command_accepted_and_rejected_increments(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("core_commands_accepted")
    increment_kivo_demotion_counter("core_commands_rejected", 2)
    add_kivo_demotion_blocker_reasons({"rejected_for_test": 2})
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["core_commands_accepted"] == 1
    assert snapshot["core_commands_rejected"] == 2
    assert snapshot["blocker_reasons"]["rejected_for_test"] == 2


def test_manager_mark_demoted_increments(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("manager_mark_demoted_attempted")
    increment_kivo_demotion_counter("manager_mark_demoted_succeeded")
    increment_kivo_demotion_counter("demoted_blocks_marked", 3)
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["manager_mark_demoted_attempted"] == 1
    assert snapshot["manager_mark_demoted_succeeded"] == 1
    assert snapshot["demoted_blocks_marked"] == 3


def test_free_to_pool_and_req_to_blocks_counters_stay_zero(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["free_to_pool_calls"] == 0
    assert snapshot["req_to_blocks_removed"] == 0


def test_run_script_build_summary_has_required_fields():
    summary = build_summary(
        generation_success=True,
        prompt_count=2,
        counters={"scheduler_envelopes_received": 0},
    )
    assert summary["generation_success"] is True
    assert summary["prompt_count"] == 2
    assert "counters" in summary
    assert summary["memory_claim_allowed"] is False
    assert summary["free_to_pool_claim_allowed"] is False


def test_validator_passes_noop_transport_case():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 2,
            "counters": {
                "scheduler_envelopes_received": 0,
                "core_commands_attempted": 0,
                "manager_mark_demoted_attempted": 0,
                "req_to_blocks_removed": 0,
                "free_to_pool_calls": 0,
            },
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
        }
    )
    assert result["validation_passed"] is True
    assert result["transport_observed"] is False
    assert result["reason"] == "no_demotable_blocks_or_policy_did_not_emit"


def test_validator_marks_transport_observed_when_scheduler_or_core_seen():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 2,
            "counters": {
                "scheduler_envelopes_received": 1,
                "core_commands_attempted": 1,
                "manager_mark_demoted_attempted": 1,
                "req_to_blocks_removed": 0,
                "free_to_pool_calls": 0,
            },
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
        }
    )
    assert result["validation_passed"] is True
    assert result["transport_observed"] is True


def test_validator_rejects_memory_or_free_claims():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 2,
            "counters": {
                "scheduler_envelopes_received": 0,
                "core_commands_attempted": 0,
                "manager_mark_demoted_attempted": 0,
                "req_to_blocks_removed": 0,
                "free_to_pool_calls": 0,
            },
            "memory_claim_allowed": True,
            "free_to_pool_claim_allowed": False,
        }
    )
    assert result["validation_passed"] is False


def test_cli_help_includes_expected_args():
    run_args = parse_run_args(["--output", str(Path("/tmp/out.json"))])
    validate_args = parse_validate_args(["--input", str(Path("/tmp/out.json"))])
    assert run_args.output.endswith("out.json")
    assert validate_args.input.endswith("out.json")
