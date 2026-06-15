from __future__ import annotations

import torch

from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.kivo_kv_sync_apply import (
    KivoKVSyncApplyConfig,
    apply_block_table_only_if_safe,
    build_kivo_kv_sync_apply_decision,
)


def _make_block_table() -> BlockTable:
    return BlockTable(
        block_size=16,
        max_num_reqs=3,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )


def test_disabled_config_returns_noop_fail_closed_decision():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [1, 2, 3, 4],
        [2, 3, 4],
        [1],
        config=KivoKVSyncApplyConfig(False, "plan_only", True),
    )
    assert decision.enabled is False
    assert decision.safe_to_apply is False
    assert decision.blocker_reasons == {"disabled": 1}


def test_valid_keep_demote_split_builds_filtered_row_preserving_order():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [10, 11, 12, 13],
        [11, 13],
        [10, 12],
        config=KivoKVSyncApplyConfig(True, "plan_only", False),
    )
    assert decision.filtered_block_ids == (11, 13)


def test_demote_overlaps_protected_fails_closed():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [1, 2, 3, 4],
        [2, 3, 4],
        [1, 4],
        protected_block_ids=[4],
        config=KivoKVSyncApplyConfig(True, "plan_only", False),
    )
    assert decision.blocker_reasons["demote_overlaps_protected"] == 1


def test_demote_overlaps_keep_fails_closed():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [1, 2, 3, 4],
        [2, 3, 4],
        [1, 4],
        config=KivoKVSyncApplyConfig(True, "plan_only", False),
    )
    assert decision.blocker_reasons["demote_overlaps_keep"] == 1


def test_empty_filtered_row_fails_closed():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [1, 2, 3],
        [],
        [1, 2, 3],
        config=KivoKVSyncApplyConfig(True, "plan_only", False),
    )
    assert decision.blocker_reasons["empty_filtered_view"] == 1


def test_unknown_keep_ids_fail_closed():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [1, 2, 3],
        [2, 9],
        [1, 3],
        config=KivoKVSyncApplyConfig(True, "plan_only", False),
    )
    assert decision.blocker_reasons["keep_ids_not_in_original"] == 1


def test_plan_only_never_mutates():
    table = _make_block_table()
    table.add_row([10, 11, 12, 13], 0)
    before = table.get_row_block_ids(0)
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        before,
        [11, 13],
        [10, 12],
        config=KivoKVSyncApplyConfig(True, "plan_only", False),
    )
    assert apply_block_table_only_if_safe(table, 0, decision) is False
    assert table.get_row_block_ids(0) == before


def test_apply_block_table_only_can_apply_locally_when_safe():
    table = _make_block_table()
    table.add_row([10, 11, 12, 13], 0)
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [10, 11, 12, 13],
        [11, 13],
        [10, 12],
        config=KivoKVSyncApplyConfig(True, "apply_block_table_only", False),
        slot_mapping_refresh_available=False,
    )
    assert decision.block_table_safe_to_apply is True
    assert decision.safe_to_apply is True
    assert apply_block_table_only_if_safe(table, 0, decision) is True
    assert table.get_row_block_ids(0) == (11, 13)


def test_apply_block_table_and_mark_ownership_fails_closed():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [10, 11, 12, 13],
        [11, 13],
        [10, 12],
        config=KivoKVSyncApplyConfig(
            True, "apply_block_table_and_mark_ownership", False
        ),
    )
    assert decision.ownership_safe_to_apply is False
    assert decision.safe_to_apply is False
    assert decision.blocker_reasons["ownership_apply_not_enabled_locally"] == 1


def test_require_slot_mapping_refresh_blocks_apply_when_unavailable():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [10, 11, 12, 13],
        [11, 13],
        [10, 12],
        config=KivoKVSyncApplyConfig(True, "apply_block_table_only", True),
        slot_mapping_refresh_available=False,
    )
    assert decision.safe_to_apply is False
    assert decision.blocker_reasons["slot_mapping_refresh_unavailable"] == 1


def test_decision_summary_contains_blocker_reasons():
    decision = build_kivo_kv_sync_apply_decision(
        "req",
        [1, 2, 3],
        [2, 9],
        [1, 3],
        config=KivoKVSyncApplyConfig(True, "plan_only", True),
        slot_mapping_refresh_available=False,
    )
    assert "keep_ids_not_in_original" in decision.blocker_reasons
    assert "slot_mapping_refresh_unavailable" in decision.blocker_reasons
