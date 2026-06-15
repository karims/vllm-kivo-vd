from __future__ import annotations

import torch

from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.kivo_block_table_sync import (
    KivoBlockTableSyncConfig,
    build_block_table_sync_plan,
)


def test_disabled_config_returns_noop_plan():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3],
        [2, 3],
        config=KivoBlockTableSyncConfig(False, "off", True),
    )
    assert plan.enabled is False
    assert plan.filtered_block_ids == (1, 2, 3)
    assert plan.removed_block_ids == ()


def test_filtered_view_preserves_order():
    plan = build_block_table_sync_plan(
        "req",
        [10, 11, 12, 13],
        [13, 11],
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )
    assert plan.filtered_block_ids == (11, 13)
    assert plan.preserves_order is True


def test_removed_ids_computed_correctly():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3, 4],
        [2, 4],
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )
    assert plan.removed_block_ids == (1, 3)


def test_unknown_keep_id_fails_closed():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3],
        [2, 9],
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )
    assert plan.blocker_reasons["keep_ids_not_in_original"] == 1


def test_empty_filtered_view_fails_closed():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3],
        [],
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )
    assert plan.blocker_reasons["empty_filtered_view"] == 1


def test_protected_id_missing_from_filtered_fails_closed():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3, 4],
        [2, 3],
        protected_block_ids=[4],
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )
    assert plan.blocker_reasons["protected_ids_missing_from_filtered"] == 1


def test_duplicate_ids_handled_conservatively():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 2, 3],
        [2, 2, 3],
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )
    assert plan.blocker_reasons["duplicate_original_ids"] == 1
    assert plan.blocker_reasons["duplicate_keep_ids"] == 1


def test_plan_filtered_view_does_not_mutate_inputs():
    original = [1, 2, 3, 4]
    keep = [2, 4]
    protected = [4]
    before_original = original.copy()
    before_keep = keep.copy()
    before_protected = protected.copy()

    build_block_table_sync_plan(
        "req",
        original,
        keep,
        protected_block_ids=protected,
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )

    assert original == before_original
    assert keep == before_keep
    assert protected == before_protected


def test_apply_filtered_view_if_safe_fails_closed_locally():
    plan = build_block_table_sync_plan(
        "req",
        [1, 2, 3, 4],
        [2, 3, 4],
        protected_block_ids=[4],
        config=KivoBlockTableSyncConfig(True, "apply_filtered_view_if_safe", True),
    )
    assert plan.safe_to_apply is False
    assert plan.blocker_reasons["apply_not_enabled_locally"] == 1


def test_integration_light_block_table_row_ordering():
    table = BlockTable(
        block_size=16,
        max_num_reqs=2,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    table.add_row([9, 11, 13, 15], 0)
    original_row = table.get_row_block_ids(0)
    plan = build_block_table_sync_plan(
        "req",
        original_row,
        [15, 11],
        protected_block_ids=[15],
        config=KivoBlockTableSyncConfig(True, "plan_filtered_view", True),
    )
    assert original_row == (9, 11, 13, 15)
    assert plan.filtered_block_ids == (11, 15)
    assert len(plan.filtered_block_ids) == 2
