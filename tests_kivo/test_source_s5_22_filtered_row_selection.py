from __future__ import annotations

from vllm.v1.core.kivo_demotion_counters import (
    get_kivo_demotion_counters_snapshot,
    reset_kivo_demotion_counters,
)
from vllm.v1.worker.kivo_runtime_block_table_apply import (
    _plan_runtime_filtered_row,
    maybe_build_kivo_demotion_command_after_runtime_apply,
    reset_kivo_demotion_command_dedupe_state_for_tests,
)


def setup_function():
    reset_kivo_demotion_command_dedupe_state_for_tests()


def test_recent_only_keep_recent_1_keeps_latest_block():
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="recent_only",
        keep_recent_blocks=1,
        max_full_blocks=64,
    )
    assert plan.visible_after_block_ids == (13,)
    assert plan.candidate_demote_block_ids == (10, 11, 12)


def test_recent_only_keep_recent_2_keeps_latest_two_blocks():
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="recent_only",
        keep_recent_blocks=2,
        max_full_blocks=64,
    )
    assert plan.visible_after_block_ids == (12, 13)
    assert plan.candidate_demote_block_ids == (10, 11)


def test_row_with_trailing_zero_padding_fails_closed_explicitly(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 0, 0),
        policy="recent_only",
        keep_recent_blocks=1,
        max_full_blocks=64,
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert plan.filtered_row_changed is False
    assert plan.blocker_reasons["padding_zero_ambiguous"] == 2
    assert snapshot["padding_zero_ambiguous"] == 2


def test_block_id_zero_ambiguity_is_explicit():
    plan = _plan_runtime_filtered_row(
        original_row=(0, 11, 12),
        policy="recent_only",
        keep_recent_blocks=1,
        max_full_blocks=64,
    )
    assert plan.blocker_reasons["padding_zero_ambiguous"] == 1


def test_noop_row_reports_reason_and_not_changed(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    plan = _plan_runtime_filtered_row(
        original_row=(12,),
        policy="recent_only",
        keep_recent_blocks=1,
        max_full_blocks=64,
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert plan.filtered_row_changed is False
    assert plan.noop_reason == "filtered_row_noop_no_blocks_above_budget"
    assert snapshot["filtered_row_apply_noop"] == 1


def test_changed_row_increments_filtered_row_changed_count(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="recent_only",
        keep_recent_blocks=1,
        max_full_blocks=64,
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["filtered_row_changed_count"] == 1


def test_command_export_receives_nonempty_candidate_demote_ids_after_changed_row(
    monkeypatch,
):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="recent_only",
        keep_recent_blocks=1,
        max_full_blocks=64,
    )
    export = maybe_build_kivo_demotion_command_after_runtime_apply(
        request_id="req0",
        visible_before_block_ids=plan.visible_before_block_ids,
        visible_after_block_ids=plan.visible_after_block_ids,
        candidate_demote_block_ids=plan.candidate_demote_block_ids,
        protected_block_ids=plan.protected_block_ids,
        apply_summary_present=True,
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        filtered_row_changed=plan.filtered_row_changed,
        keep_recent_blocks=1,
        policy="recent_only",
    )
    assert export.command is not None
    assert export.candidate_demote_block_ids == (10, 11, 12)


def test_no_req_to_blocks_removal_and_no_free_to_pool(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="recent_only",
        keep_recent_blocks=1,
        max_full_blocks=64,
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["req_to_blocks_removed"] == 0
    assert snapshot["free_to_pool_calls"] == 0
