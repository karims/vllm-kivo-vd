from __future__ import annotations

from vllm.v1.core.kivo_ownership_bridge import (
    KivoOwnershipBridgeConfig,
    build_kivo_ownership_bridge_decision,
)


def _config(
    *,
    enabled: bool = True,
    action: str = "plan_only",
    require_block_table_applied: bool = True,
    require_slot_mapping_refresh: bool = True,
) -> KivoOwnershipBridgeConfig:
    return KivoOwnershipBridgeConfig(
        enabled=enabled,
        action=action,
        require_block_table_applied=require_block_table_applied,
        require_slot_mapping_refresh=require_slot_mapping_refresh,
    )


def test_disabled_bridge_returns_fail_closed_noop():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=False,
        slot_mapping_refresh_guaranteed=False,
        ownership_mapping_available=False,
        config=_config(enabled=False),
    )
    assert decision.enabled is False
    assert decision.safe_to_mark_demoted is False
    assert decision.safe_to_free is False
    assert decision.blocker_reasons["disabled"] == 1


def test_plan_only_never_mutates_even_when_eligible():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(action="plan_only"),
    )
    assert decision.ownership_after_block_ids == (12, 13)
    assert decision.safe_to_mark_demoted is False
    assert decision.safe_to_free is False


def test_demote_absent_after_and_present_before_is_eligible_for_mark_only():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(action="mark_demoted_if_safe"),
    )
    assert decision.ownership_after_block_ids == (12, 13)
    assert decision.safe_to_mark_demoted is False
    assert decision.blocker_reasons["ownership_mark_demoted_not_implemented"] == 1


def test_demote_id_still_visible_after_fails_closed():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(11, 12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["demote_ids_still_visible_after"] == 1


def test_demote_id_missing_from_ownership_before_fails_closed():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        ownership_before_block_ids=(10, 12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["demote_ids_missing_from_ownership_before"] == 1


def test_visible_after_not_subset_of_ownership_before_fails_closed():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(12, 13, 99),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["visible_after_not_subset_of_ownership_before"] == 1


def test_missing_request_id_fails_closed():
    decision = build_kivo_ownership_bridge_decision(
        request_id=None,
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["missing_request_id"] == 1


def test_missing_block_table_applied_proof_fails_closed():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=False,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(require_block_table_applied=True),
    )
    assert decision.blocker_reasons["block_table_apply_required"] == 1


def test_missing_slot_mapping_refresh_proof_fails_closed():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=False,
        ownership_mapping_available=True,
        config=_config(require_slot_mapping_refresh=True),
    )
    assert decision.blocker_reasons["slot_mapping_refresh_not_guaranteed"] == 1


def test_safe_to_free_remains_false_in_s5_11():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.safe_to_free is False


def test_demote_overlaps_protected_fails_closed():
    decision = build_kivo_ownership_bridge_decision(
        request_id="req0",
        demote_block_ids=(10, 13),
        visible_after_block_ids=(11, 12, 13),
        ownership_before_block_ids=(10, 11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        ownership_mapping_available=True,
        config=_config(),
    )
    assert decision.blocker_reasons["demote_overlaps_protected"] == 1
