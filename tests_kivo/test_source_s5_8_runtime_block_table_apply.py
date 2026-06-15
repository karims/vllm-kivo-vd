from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.core.kivo_kv_block_score_store import (
    KivoKVBlockScore,
    clear_block_scores,
    update_block_scores,
)
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.kivo_runtime_block_table_apply import (
    KivoRuntimeBlockTableApplyConfig,
    build_runtime_block_table_apply_summary,
)


@dataclass
class FakeInputBatch:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    block_table: MultiGroupBlockTable

    def get_req_index(self, req_id: str) -> int | None:
        return self.req_id_to_index.get(req_id)

    def get_req_block_row_ids(
        self, req_id: str, kv_cache_gid: int = 0
    ) -> tuple[int, ...] | None:
        req_index = self.req_id_to_index.get(req_id)
        if req_index is None:
            return None
        return self.block_table[kv_cache_gid].get_row_block_ids(req_index)


def _make_input_batch() -> FakeInputBatch:
    block_table = MultiGroupBlockTable(
        max_num_reqs=4,
        max_model_len=128,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks=[8],
        cp_kv_cache_interleave_size=1,
    )
    block_table.add_row(([10, 11, 12, 13],), 0)
    block_table.add_row(([20, 21, 22, 23],), 1)
    return FakeInputBatch(
        req_ids=["req0", "req1"],
        req_id_to_index={"req0": 0, "req1": 1},
        block_table=block_table,
    )


def test_disabled_runtime_apply_returns_noop_summary():
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        config=KivoRuntimeBlockTableApplyConfig(
            False, "off", "recent_only", 4, 64, True
        ),
    )
    assert summary.enabled is False
    assert summary.attempted_row_count == 0
    assert batch.block_table[0].get_row_block_ids(0) == (10, 11, 12, 13)


def test_enabled_recent_only_can_filter_fake_row_before_slot_mapping():
    clear_block_scores()
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 2, 2, True
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0) == (12, 13)


def test_filtered_row_preserves_order():
    clear_block_scores()
    batch = _make_input_batch()
    build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 2, 2, True
        ),
    )
    assert batch.block_table[0].get_row_block_ids(0) == (12, 13)


def test_empty_filtered_row_fails_closed():
    clear_block_scores()
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 0, 0, True
        ),
    )
    assert summary.blocked_row_count == 1
    assert summary.blocker_reasons["empty_filtered_view"] == 1


def test_missing_request_row_mapping_fails_closed():
    clear_block_scores()
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["missing"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 2, 2, True
        ),
    )
    assert summary.blocked_row_count == 1
    assert summary.blocker_reasons["missing_request_row_mapping"] == 1


def test_block_table_only_apply_does_not_call_ownership_or_free_path():
    clear_block_scores()
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 2, 2, True
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0) == (12, 13)


def test_countsketch_online_keeps_recent_plus_high_score_blocks():
    clear_block_scores()
    update_block_scores(
        [
            KivoKVBlockScore(block_id=10, score=0.1, source="test"),
            KivoKVBlockScore(block_id=11, score=0.9, source="test"),
            KivoKVBlockScore(block_id=12, score=0.2, source="test"),
            KivoKVBlockScore(block_id=13, score=0.8, source="test"),
        ]
    )
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "countsketch_online", 1, 2, True
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0) == (11, 13)


def test_missing_countsketch_scores_are_protected_or_fail_closed():
    clear_block_scores()
    update_block_scores([KivoKVBlockScore(block_id=10, score=0.1, source="test")])
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "countsketch_online", 1, 2, True
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0) == (11, 12, 13)


def test_summary_reports_attempted_applied_blocked_counts():
    clear_block_scores()
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0", "missing"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 2, 2, True
        ),
    )
    assert summary.attempted_row_count == 2
    assert summary.applied_row_count == 1
    assert summary.blocked_row_count == 1


def test_default_behavior_unchanged_when_disabled():
    clear_block_scores()
    batch = _make_input_batch()
    before = batch.block_table[0].get_row_block_ids(0)
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=False,
        config=KivoRuntimeBlockTableApplyConfig(
            False, "off", "recent_only", 4, 64, True
        ),
    )
    assert summary.enabled is False
    assert batch.block_table[0].get_row_block_ids(0) == before
