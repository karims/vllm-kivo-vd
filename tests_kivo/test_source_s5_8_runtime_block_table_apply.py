from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import pytest

from vllm.v1.core.kivo_kv_block_score_store import (
    KivoKVBlockScore,
    clear_block_scores,
    update_block_scores,
)
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.kivo_kv_sketch_runtime import (
    KivoKVBlockSketchRecord,
    KivoKVSketchRuntime,
    KivoKVSketchRuntimeConfig,
    KivoKVSketchStore,
    RandomProjectionKVSketchBackend,
)
from vllm.v1.worker.kivo_runtime_block_table_apply import (
    KivoRuntimeBlockTableApplyConfig,
    _plan_runtime_filtered_row,
    build_runtime_block_table_apply_summary,
    maybe_build_kivo_demotion_command_after_runtime_apply,
    reset_kivo_demotion_command_dedupe_state_for_tests,
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


def _make_sketch_runtime(
    *, sketch_dim: int = 4, seed: int = 7, max_blocks: int = 16
) -> KivoKVSketchRuntime:
    return KivoKVSketchRuntime(
        config=KivoKVSketchRuntimeConfig(
            enabled=True,
            backend="random_projection",
            sketch_dim=sketch_dim,
            seed=seed,
            max_blocks=max_blocks,
        ),
        backend=RandomProjectionKVSketchBackend(
            sketch_dim=sketch_dim,
            seed=seed,
        ),
        store=KivoKVSketchStore(max_blocks=max_blocks),
    )


def _make_kv_cache(num_blocks: int = 8) -> torch.Tensor:
    return torch.arange(
        num_blocks * 2 * 4 * 2 * 4,
        dtype=torch.float32,
    ).reshape(num_blocks, 2, 4, 2, 4)


@pytest.fixture(autouse=True)
def _reset_dedupe_state() -> None:
    reset_kivo_demotion_command_dedupe_state_for_tests()


def _store_fake_sketch_score(
    runtime: KivoKVSketchRuntime,
    *,
    block_id: int,
    score: float,
) -> None:
    runtime.store.update(
        KivoKVBlockSketchRecord(
            block_id=block_id,
            backend="random_projection",
            sketch_dim=1,
            sketch=torch.tensor([score], dtype=torch.float32),
            source_shape=(1,),
            source_numel=1,
            source_dtype="torch.float32",
            source_device="cpu",
            shape_summary=(1,),
            created_counter=1,
            updated_counter=1,
        )
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
    assert summary.paired_plan_attempted_row_count == 0
    assert summary.runtime_demotion_mark_attempted_request_count == 0


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


def test_sketch_topk_always_keeps_recent_blocks():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=10.0)
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_topk", 2, 3, True, 1
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0) == (10, 12, 13)


def test_sketch_topk_keeps_top_scored_old_blocks():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=0.1)
    _store_fake_sketch_score(runtime, block_id=11, score=9.0)
    _store_fake_sketch_score(runtime, block_id=12, score=0.2)
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_topk", 1, 2, True, 1
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0) == (11, 13)


def test_sketch_topk_builds_scores_before_selection_from_kv_cache():
    runtime = _make_sketch_runtime()
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=_make_kv_cache(32),
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_topk", 1, 2, True, 1
        ),
    )
    kept = batch.block_table[0].get_row_block_ids(0)
    older_kept = tuple(block_id for block_id in kept if block_id not in (13,))

    assert summary.applied_row_count == 1
    assert kept[-1:] == (13,)
    assert len(older_kept) == 1
    assert runtime.store.get(10) is not None
    assert runtime.store.get(11) is not None
    assert runtime.store.get(12) is not None


def test_sketch_topk_missing_scores_falls_back_to_recent_only():
    runtime = _make_sketch_runtime()
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_topk", 2, 4, True, 2
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0) == (12, 13)


def test_prefix_recent_keeps_first_p_and_last_k() -> None:
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13, 14, 15),
        policy="prefix_recent",
        keep_recent_blocks=2,
        keep_prefix_blocks=2,
        max_full_blocks=4,
    )
    assert plan.visible_after_block_ids == (10, 11, 14, 15)
    assert plan.candidate_demote_block_ids == (12, 13)


def test_prefix_recent_preserves_original_order() -> None:
    plan = _plan_runtime_filtered_row(
        original_row=(100, 101, 102, 103, 104, 105),
        policy="prefix_recent",
        keep_recent_blocks=2,
        keep_prefix_blocks=2,
        max_full_blocks=4,
    )
    assert plan.visible_after_block_ids == (100, 101, 104, 105)


def test_prefix_recent_respects_max_full_blocks() -> None:
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13, 14, 15),
        policy="prefix_recent",
        keep_recent_blocks=3,
        keep_prefix_blocks=3,
        max_full_blocks=4,
    )
    assert plan.visible_after_block_ids == (10, 11, 12, 13)


def test_prefix_recent_trace_includes_prefix_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace_path = tmp_path / "prefix_recent_trace.jsonl"
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_RETAINED_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_MAX_BLOCKS", "64")
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        config=KivoRuntimeBlockTableApplyConfig(
            True,
            "apply_block_table_only",
            "prefix_recent",
            2,
            4,
            True,
            0,
            0,
            2,
        ),
    )
    assert summary.applied_row_count == 1
    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["policy"] == "prefix_recent"
    assert record["visible_before_count"] == 4
    assert record["visible_after_count"] == 4
    assert record["recent_keep_count"] == 2
    assert record["retention_ratio"] == 1.0


def test_sketch_topk_missing_kv_cache_falls_back_safely():
    runtime = _make_sketch_runtime()
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_topk",
        keep_recent_blocks=2,
        max_full_blocks=4,
        sketch_topk_blocks=2,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.visible_after_block_ids == (12, 13)
    assert plan.candidate_demote_block_ids == (10, 11)


def test_trace_disabled_has_no_behavior_change(monkeypatch):
    monkeypatch.delenv(
        "KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_RETAINED_BLOCKS", raising=False
    )
    trace_file = Path("/tmp/kivo_runtime_trace_disabled.jsonl")
    if trace_file.exists():
        trace_file.unlink()
    runtime = _make_sketch_runtime()
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_topk",
        keep_recent_blocks=1,
        max_full_blocks=2,
        sketch_topk_blocks=1,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.visible_after_block_ids == (13,)
    assert not trace_file.exists()


def test_trace_enabled_writes_retention_fields(monkeypatch, tmp_path):
    trace_file = tmp_path / "retention_trace.jsonl"
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_RETAINED_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_FILE", str(trace_file))
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_MAX_BLOCKS", "64")
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=5.0)
    batch = _make_input_batch()
    build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_topk", 1, 2, True, 1
        ),
    )
    rows = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    record = rows[-1]
    assert record["visible_before_block_ids"] == [10, 11, 12, 13]
    assert record["recent_keep_block_ids"] == [13]
    assert record["sketch_topk_keep_block_ids"] == [10]
    assert record["candidate_demote_block_ids"] == [11, 12]
    assert record["final_block_order"] == [10, 13]


def test_trace_max_blocks_truncates_safely(monkeypatch, tmp_path):
    trace_file = tmp_path / "retention_trace.jsonl"
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_RETAINED_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_FILE", str(trace_file))
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_MAX_BLOCKS", "2")
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=5.0)
    batch = _make_input_batch()
    build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_topk", 1, 2, True, 1
        ),
    )
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["visible_before_count"] == 4
    assert record["visible_before_block_ids"] == [10, 11]


def test_trace_preserves_relative_order_of_kept_blocks():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=5.0)
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_topk",
        keep_recent_blocks=1,
        max_full_blocks=2,
        sketch_topk_blocks=1,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.visible_after_block_ids == (10, 13)
    assert plan.sketch_topk_keep_block_ids == (10,)


def test_trace_includes_scored_and_missing_blocks():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=5.0)
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_topk",
        keep_recent_blocks=1,
        max_full_blocks=2,
        sketch_topk_blocks=1,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.old_block_score_pairs == ((10, 5.0),)
    assert plan.missing_score_block_ids == (11, 12)


def test_sketch_topk_candidate_demote_excludes_recent_and_sketch_kept_old():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=5.0)
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_topk",
        keep_recent_blocks=1,
        max_full_blocks=2,
        sketch_topk_blocks=1,
        kv_sketch_runtime=runtime,
    )
    assert plan.visible_after_block_ids == (10, 13)
    assert plan.protected_block_ids == (13,)
    assert plan.candidate_demote_block_ids == (11, 12)


def test_sketch_span_topk_keeps_recent_blocks():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=11, score=9.0)
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_span_topk", 1, 4, True, 1, 1
        ),
    )
    assert summary.applied_row_count == 1
    assert batch.block_table[0].get_row_block_ids(0)[-1:] == (13,)


def test_sketch_span_topk_keeps_anchor_and_neighbors():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=11, score=9.0)
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_span_topk",
        keep_recent_blocks=1,
        max_full_blocks=4,
        sketch_topk_blocks=1,
        sketch_span_radius=1,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.sketch_span_anchor_block_ids == (11,)
    assert plan.sketch_span_keep_block_ids == (10, 11, 12)
    assert plan.visible_after_block_ids == (10, 11, 12, 13)


def test_sketch_span_topk_preserves_original_order():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=11, score=9.0)
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_span_topk",
        keep_recent_blocks=1,
        max_full_blocks=3,
        sketch_topk_blocks=1,
        sketch_span_radius=1,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.visible_after_block_ids in {(10, 11, 13), (11, 12, 13)}
    assert plan.visible_after_block_ids == tuple(
        block_id
        for block_id in (10, 11, 12, 13)
        if block_id in set(plan.visible_after_block_ids)
    )


def test_sketch_span_topk_respects_max_full_blocks_cap():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=11, score=9.0)
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_span_topk",
        keep_recent_blocks=1,
        max_full_blocks=2,
        sketch_topk_blocks=1,
        sketch_span_radius=2,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.visible_after_block_ids == (11, 13)
    assert plan.sketch_span_keep_block_ids == (11,)


def test_sketch_span_topk_missing_scores_falls_back_safely():
    runtime = _make_sketch_runtime()
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_span_topk",
        keep_recent_blocks=2,
        max_full_blocks=4,
        sketch_topk_blocks=1,
        sketch_span_radius=1,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.visible_after_block_ids == (12, 13)
    assert plan.sketch_span_anchor_block_ids == ()


def test_sketch_span_topk_trace_includes_span_fields(monkeypatch, tmp_path):
    trace_file = tmp_path / "span_trace.jsonl"
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_RETAINED_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_FILE", str(trace_file))
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=11, score=9.0)
    build_runtime_block_table_apply_summary(
        _make_input_batch(),
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=runtime,
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "sketch_span_topk", 1, 4, True, 1, 1
        ),
    )
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["sketch_span_anchor_block_ids"] == [11]
    assert record["sketch_span_keep_block_ids"] == [10, 11, 12]
    assert record["retention_ratio"] == 1.0
    assert record["contiguous_span_count"] == 1
    assert record["max_gap_between_kept_blocks"] == 0


def test_sketch_span_topk_derived_diagnostics_are_computed():
    runtime = _make_sketch_runtime()
    _store_fake_sketch_score(runtime, block_id=10, score=9.0)
    _store_fake_sketch_score(runtime, block_id=12, score=8.0)
    plan = _plan_runtime_filtered_row(
        original_row=(10, 11, 12, 13),
        policy="sketch_span_topk",
        keep_recent_blocks=1,
        max_full_blocks=3,
        sketch_topk_blocks=2,
        sketch_span_radius=0,
        kv_sketch_runtime=runtime,
        kv_cache_tensor=None,
    )
    assert plan.retention_ratio_numerator == 3
    assert plan.retention_ratio_denominator == 4
    assert plan.contiguous_span_count == 2
    assert plan.max_gap_between_kept_blocks == 1


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


def test_live_paired_plan_reports_explicit_blocker_without_mutating_ownership(
    monkeypatch,
):
    monkeypatch.setenv("KIVO_KV_LIVE_APPLY_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_LIVE_APPLY_ACTION", "plan_paired_apply")
    monkeypatch.setenv("KIVO_KV_LIVE_APPLY_REQUIRE_BLOCK_TABLE_APPLIED", "1")
    monkeypatch.setenv("KIVO_KV_LIVE_APPLY_REQUIRE_SLOT_MAPPING_REFRESH", "1")
    monkeypatch.setenv("KIVO_KV_LIVE_APPLY_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_LIVE_APPLY_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_LIVE_APPLY_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_BRIDGE_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_BRIDGE_ACTION", "mark_demoted_if_safe")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_BRIDGE_REQUIRE_BLOCK_TABLE_APPLIED", "1")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_BRIDGE_REQUIRE_SLOT_MAPPING_REFRESH", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_DEMOTION_MARK_ENABLE", "1")
    monkeypatch.setenv(
        "KIVO_KV_RUNTIME_DEMOTION_MARK_ACTION",
        "mark_demoted_after_block_table_apply",
    )
    monkeypatch.setenv("KIVO_KV_RUNTIME_DEMOTION_MARK_REQUIRE_BLOCK_TABLE_APPLIED", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_DEMOTION_MARK_REQUIRE_SLOT_MAPPING_REFRESH", "1")
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
    assert summary.paired_plan_attempted_row_count == 1
    assert summary.paired_plan_safe_row_count == 0
    assert summary.paired_plan_blocked_row_count == 1
    assert summary.paired_plan_blocker_reasons["ownership_mapping_unavailable"] == 1
    assert (
        summary.paired_plan_blocker_reasons[
            "worker_path_lacks_core_kv_manager_reference"
        ]
        == 1
    )
    assert summary.runtime_demotion_mark_attempted_request_count == 1
    assert summary.runtime_demotion_mark_marked_request_count == 0
    assert summary.runtime_demotion_mark_blocked_request_count == 1
    assert summary.runtime_demotion_mark_blocker_reasons["kv_cache_manager_unavailable"] == 1


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


def test_sketch_disabled_keeps_candidate_demote_blocks_unchanged():
    export = maybe_build_kivo_demotion_command_after_runtime_apply(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        apply_summary_present=True,
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        filtered_row_changed=True,
        keep_recent_blocks=2,
        policy="recent_only",
        kv_sketch_runtime=None,
        kv_cache_tensor=None,
    )
    assert export.command is not None
    assert export.command.candidate_demote_block_ids == (10, 11)
    assert export.sketch_build_attempted == 0


def test_sketch_enabled_all_candidate_blocks_succeed():
    runtime = _make_sketch_runtime()
    export = maybe_build_kivo_demotion_command_after_runtime_apply(
        request_id="req0",
        visible_before_block_ids=(1, 2, 3, 4),
        visible_after_block_ids=(3, 4),
        candidate_demote_block_ids=(1, 2),
        protected_block_ids=(3, 4),
        apply_summary_present=True,
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        filtered_row_changed=True,
        keep_recent_blocks=2,
        policy="recent_only",
        kv_sketch_runtime=runtime,
        kv_cache_tensor=_make_kv_cache(8),
    )
    assert export.command is not None
    assert export.command.candidate_demote_block_ids == (1, 2)
    assert export.sketch_build_attempted == 2
    assert export.sketch_build_succeeded == 2
    assert export.sketch_build_failed == 0
    assert runtime.store.get(1) is not None
    assert runtime.store.get(2) is not None


def test_sketch_enabled_partial_failure_excludes_failed_block_ids():
    runtime = _make_sketch_runtime()
    export = maybe_build_kivo_demotion_command_after_runtime_apply(
        request_id="req0",
        visible_before_block_ids=(1, 2, 3, 4),
        visible_after_block_ids=(3, 4),
        candidate_demote_block_ids=(1, 99),
        protected_block_ids=(3, 4),
        apply_summary_present=True,
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        filtered_row_changed=True,
        keep_recent_blocks=2,
        policy="recent_only",
        kv_sketch_runtime=runtime,
        kv_cache_tensor=_make_kv_cache(8),
    )
    assert export.command is not None
    assert export.command.candidate_demote_block_ids == (1,)
    assert export.sketch_build_attempted == 2
    assert export.sketch_build_succeeded == 1
    assert export.sketch_build_failed == 1
    assert export.sketch_missing_prevented_demotion == 1
    assert runtime.store.get(1) is not None
    assert runtime.store.get(99) is None


def test_sketch_enabled_all_fail_exports_no_unsafe_demotion():
    runtime = _make_sketch_runtime()
    export = maybe_build_kivo_demotion_command_after_runtime_apply(
        request_id="req0",
        visible_before_block_ids=(1, 2, 3, 4),
        visible_after_block_ids=(3, 4),
        candidate_demote_block_ids=(90, 91),
        protected_block_ids=(3, 4),
        apply_summary_present=True,
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        filtered_row_changed=True,
        keep_recent_blocks=2,
        policy="recent_only",
        kv_sketch_runtime=runtime,
        kv_cache_tensor=_make_kv_cache(8),
    )
    assert export.command is None
    assert export.sketch_build_attempted == 2
    assert export.sketch_build_succeeded == 0
    assert export.sketch_build_failed == 2
    assert export.sketch_missing_prevented_demotion == 2
    assert export.blocker_reasons["empty_candidate_demote_ids"] == 1


def test_sketch_enabled_unsupported_kv_cache_shape_fails_closed():
    runtime = _make_sketch_runtime()
    export = maybe_build_kivo_demotion_command_after_runtime_apply(
        request_id="req0",
        visible_before_block_ids=(1, 2, 3, 4),
        visible_after_block_ids=(3, 4),
        candidate_demote_block_ids=(1, 2),
        protected_block_ids=(3, 4),
        apply_summary_present=True,
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        filtered_row_changed=True,
        keep_recent_blocks=2,
        policy="recent_only",
        kv_sketch_runtime=runtime,
        kv_cache_tensor=torch.ones(2, 2, dtype=torch.float32),
    )
    assert export.command is None
    assert export.sketch_build_succeeded == 0
    assert export.sketch_build_failed == 2
    assert export.blocker_reasons["empty_candidate_demote_ids"] == 1


def test_summary_reports_sketch_gating_stats(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_TRANSPORT_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_DEMOTION_TRANSPORT_ACTION", "export_only")
    batch = _make_input_batch()
    summary = build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=_make_sketch_runtime(),
        kv_cache_tensor=_make_kv_cache(32),
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 2, 2, True
        ),
    )
    assert summary.demotion_transport_exported_count == 1
    assert summary.sketch_build_attempted == 2
    assert summary.sketch_build_succeeded == 2
    assert summary.sketch_build_failed == 0
    assert summary.sketched_blocks_total == 2
    assert summary.sketch_missing_prevented_demotion == 0
    assert summary.sketch_backend == "random_projection"
    assert summary.sketch_bytes_total > 0


def test_counter_export_file_includes_sketch_fields(monkeypatch, tmp_path):
    export_path = tmp_path / "kivo_counters.json"
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE", str(export_path))
    monkeypatch.setenv("KIVO_KV_DEMOTION_TRANSPORT_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_DEMOTION_TRANSPORT_ACTION", "export_only")

    batch = _make_input_batch()
    build_runtime_block_table_apply_summary(
        batch,
        req_ids=["req0"],
        slot_mapping_refresh_available=True,
        kv_sketch_runtime=_make_sketch_runtime(),
        kv_cache_tensor=_make_kv_cache(32),
        config=KivoRuntimeBlockTableApplyConfig(
            True, "apply_block_table_only", "recent_only", 2, 2, True
        ),
    )

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    counters = payload["counters"]
    assert counters["sketch_build_attempted"] == 2
    assert counters["sketch_build_succeeded"] == 2
    assert counters["sketch_build_failed"] == 0
    assert counters["sketch_missing_prevented_demotion"] == 0
    assert counters["sketched_blocks_total"] == 2
    assert counters["sketch_bytes_total"] > 0
    assert counters["sketch_backend"] == "random_projection"
