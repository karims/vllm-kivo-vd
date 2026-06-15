from __future__ import annotations

from vllm.v1.core.kivo_live_ownership_apply import (
    KivoLiveOwnershipApplyConfig,
    build_kivo_live_ownership_apply_decision,
)


def _config(
    *,
    enabled: bool = True,
    action: str = "plan_paired_apply",
    require_block_table_applied: bool = True,
    require_slot_mapping_refresh: bool = True,
) -> KivoLiveOwnershipApplyConfig:
    return KivoLiveOwnershipApplyConfig(
        enabled=enabled,
        action=action,
        policy="recent_only",
        keep_recent_blocks=4,
        max_full_blocks=64,
        require_block_table_applied=require_block_table_applied,
        require_slot_mapping_refresh=require_slot_mapping_refresh,
    )


def test_disabled_config_returns_fail_closed_noop_decision():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=False,
        slot_mapping_refresh_guaranteed=False,
        ownership_mapping_available=False,
        config=_config(enabled=False),
    )
    assert decision.enabled is False
    assert decision.safe_to_mutate_ownership is False
    assert decision.blocker_reasons["disabled"] == 1


def test_plan_paired_apply_does_not_mutate_even_when_pairing_is_eligible():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(action="plan_paired_apply"),
    )
    assert decision.ownership_mutation_block_ids == (10, 11)
    assert decision.safe_to_mutate_ownership is False


def test_candidate_demote_ids_absent_after_filtered_row_are_eligible():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.ownership_mutation_block_ids == (10, 11)
    assert "candidate_demote_still_visible_after" not in decision.blocker_reasons


def test_candidate_demote_ids_still_visible_after_filtered_row_fail_closed():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(11, 12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["candidate_demote_still_visible_after"] == 1


def test_protected_ids_missing_after_filtered_row_fail_closed():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(13,),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["protected_ids_missing_after"] == 1


def test_slot_mapping_refresh_not_guaranteed_fails_closed():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=False,
        ownership_mapping_available=True,
        config=_config(require_slot_mapping_refresh=True),
    )
    assert decision.blocker_reasons["slot_mapping_refresh_not_guaranteed"] == 1


def test_missing_request_id_fails_closed():
    decision = build_kivo_live_ownership_apply_decision(
        request_id=None,
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["missing_request_id"] == 1


def test_ownership_mutation_ids_exclude_protected_recent_ids():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(11, 12, 13),
        candidate_demote_block_ids=(10,),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.ownership_mutation_block_ids == (10,)
    assert "ownership_mutation_overlaps_protected" not in decision.blocker_reasons


def test_duplicate_ids_fail_closed():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 10, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 10),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["duplicate_visible_before_ids"] == 1
    assert decision.blocker_reasons["duplicate_candidate_demote_ids"] == 1


def test_empty_filtered_row_fails_closed():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11),
        visible_after_block_ids=(),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["empty_visible_after_blocks"] == 1


def test_apply_action_reports_explicit_local_blocker():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(action="apply_block_table_then_mark_ownership"),
    )
    assert decision.safe_to_mutate_ownership is False
    assert decision.blocker_reasons["ownership_mutation_not_enabled_locally"] == 1


def test_missing_ownership_mapping_reports_explicit_blocker():
    decision = build_kivo_live_ownership_apply_decision(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=False,
        config=_config(),
    )
    assert decision.blocker_reasons["ownership_mapping_unavailable"] == 1
