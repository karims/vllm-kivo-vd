from __future__ import annotations

import torch

from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.kivo_block_table_sync import (
    KivoBlockTableSyncConfig,
    apply_filtered_block_row_if_safe,
    build_block_table_sync_plan,
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


def test_default_disabled_behavior_unchanged():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3, 4],
        [2, 4],
        config=KivoBlockTableSyncConfig(False, "off", True),
    )
    assert plan.enabled is False
    assert plan.filtered_block_ids == (1, 2, 3, 4)


def test_pure_apply_helper_returns_filtered_row_preserving_order():
    result = apply_filtered_block_row_if_safe([9, 11, 13, 15], [11, 15])
    assert result.ok is True
    assert result.filtered_block_ids == (11, 15)


def test_unknown_block_id_fails_closed():
    result = apply_filtered_block_row_if_safe([1, 2, 3], [2, 9])
    assert result.ok is False
    assert result.blocker_reasons["filtered_ids_not_in_original"] == 1


def test_duplicate_ids_fail_closed():
    result = apply_filtered_block_row_if_safe([1, 2, 2, 3], [2, 2, 3])
    assert result.ok is False
    assert result.blocker_reasons["duplicate_original_ids"] == 1
    assert result.blocker_reasons["duplicate_filtered_ids"] == 1


def test_empty_row_fails_closed():
    result = apply_filtered_block_row_if_safe([1, 2, 3], [])
    assert result.ok is False
    assert result.blocker_reasons["empty_filtered_row"] == 1


def test_block_table_row_replacement_modifies_only_target_row():
    table = _make_block_table()
    table.add_row([10, 11, 12, 13], 0)
    table.add_row([20, 21, 22], 1)

    ok = table.replace_row_block_ids_if_safe(0, [11, 13])

    assert ok is True
    assert table.get_row_block_ids(0) == (11, 13)
    assert table.get_row_block_ids(1) == (20, 21, 22)


def test_row_replacement_clears_trailing_entries_correctly():
    table = _make_block_table()
    table.add_row([5, 6, 7, 8], 0)

    ok = table.replace_row_block_ids_if_safe(0, [6, 8])

    assert ok is True
    assert table.get_row_block_ids(0) == (6, 8)
    assert table.block_table.np[0, 2] == 0
    assert table.block_table.np[0, 3] == 0


def test_unrelated_rows_unchanged():
    table = _make_block_table()
    table.add_row([1, 2, 3], 0)
    table.add_row([4, 5, 6], 1)
    before_other_row = table.get_row_block_ids(1)

    table.replace_row_block_ids_if_safe(0, [2, 3])

    assert table.get_row_block_ids(1) == before_other_row


def test_invalid_filtered_view_does_not_replace_row():
    table = _make_block_table()
    table.add_row([1, 2, 3], 0)
    before = table.get_row_block_ids(0)

    ok = table.replace_row_block_ids_if_safe(0, [2, 9])

    assert ok is False
    assert table.get_row_block_ids(0) == before


def test_apply_action_remains_gated():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3, 4],
        [2, 3, 4],
        protected_block_ids=[4],
        config=KivoBlockTableSyncConfig(True, "apply_filtered_view_if_safe", True),
    )
    assert plan.safe_to_apply is False
    assert plan.blocker_reasons["apply_not_enabled_locally"] == 1
