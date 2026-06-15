from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


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


@dataclass
class FakeRunner:
    input_batch: FakeInputBatch
    _last_kivo_runtime_block_table_apply_summary: object | None = None


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
    return FakeInputBatch(
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        block_table=block_table,
    )


def test_disabled_hook_does_nothing(monkeypatch):
    monkeypatch.delenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", raising=False)
    batch = _make_input_batch()
    runner = FakeRunner(input_batch=batch)

    summary = GPUModelRunner._maybe_apply_kivo_runtime_block_table_before_slot_mapping(
        runner, 1
    )

    assert summary.enabled is False
    assert batch.block_table[0].get_row_block_ids(0) == (10, 11, 12, 13)


def test_hook_is_actually_placed_before_compute_slot_mapping():
    source = Path(
        "/Users/ksnaik/StudioProjects/vllm-kivo-vd/vllm/v1/worker/gpu_model_runner.py"
    ).read_text()
    prepare_start = source.index("def _prepare_inputs(")
    build_start = source.index("def _build_attention_metadata(")
    prepare_source = source[prepare_start:build_start]
    hook_pos = prepare_source.index(
        "self._maybe_apply_kivo_runtime_block_table_before_slot_mapping(num_reqs)"
    )
    compute_pos = prepare_source.index(
        "self.input_batch.block_table.compute_slot_mapping("
    )
    assert hook_pos < compute_pos


def test_slot_mapping_computed_after_row_replacement_in_mocked_order(monkeypatch):
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION", "apply_block_table_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_REQUIRE_SLOT_MAPPING_REFRESH", "1")
    batch = _make_input_batch()
    runner = FakeRunner(input_batch=batch)
    observed: dict[str, tuple[int, ...]] = {}

    def fake_compute_slot_mapping(*args, **kwargs):
        del args, kwargs
        observed["row"] = batch.block_table[0].get_row_block_ids(0)

    batch.block_table.compute_slot_mapping = fake_compute_slot_mapping  # type: ignore[method-assign]

    GPUModelRunner._maybe_apply_kivo_runtime_block_table_before_slot_mapping(runner, 1)
    batch.block_table.compute_slot_mapping()

    assert batch.block_table[0].get_row_block_ids(0) == (12, 13)
    assert observed["row"] == (12, 13)


def test_missing_request_row_mapping_fails_closed(monkeypatch):
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION", "apply_block_table_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_REQUIRE_SLOT_MAPPING_REFRESH", "1")
    batch = _make_input_batch()
    batch.req_ids = ["missing"]
    runner = FakeRunner(input_batch=batch)

    summary = GPUModelRunner._maybe_apply_kivo_runtime_block_table_before_slot_mapping(
        runner, 1
    )

    assert summary.blocked_row_count == 1
    assert summary.blocker_reasons["missing_request_row_mapping"] == 1


def test_invalid_filtered_row_fails_closed(monkeypatch):
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION", "apply_block_table_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_REQUIRE_SLOT_MAPPING_REFRESH", "1")
    batch = _make_input_batch()
    batch.block_table.add_row(([10, 10, 12, 13],), 0)
    runner = FakeRunner(input_batch=batch)

    summary = GPUModelRunner._maybe_apply_kivo_runtime_block_table_before_slot_mapping(
        runner, 1
    )

    assert summary.blocked_row_count == 1
    assert summary.blocker_reasons["duplicate_original_ids"] == 1


def test_no_ownership_free_path_is_called(monkeypatch):
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION", "apply_block_table_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_REQUIRE_SLOT_MAPPING_REFRESH", "1")
    batch = _make_input_batch()
    runner = FakeRunner(input_batch=batch)

    summary = GPUModelRunner._maybe_apply_kivo_runtime_block_table_before_slot_mapping(
        runner, 1
    )

    assert summary.applied_row_count == 1
    assert not hasattr(runner, "block_pool")


def test_summary_reports_attempted_applied_blocked_rows(monkeypatch):
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION", "apply_block_table_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RUNTIME_BLOCK_TABLE_REQUIRE_SLOT_MAPPING_REFRESH", "1")
    batch = _make_input_batch()
    batch.req_ids = ["req0", "missing"]
    runner = FakeRunner(input_batch=batch)

    summary = GPUModelRunner._maybe_apply_kivo_runtime_block_table_before_slot_mapping(
        runner, 2
    )

    assert summary.attempted_row_count == 2
    assert summary.applied_row_count == 1
    assert summary.blocked_row_count == 1


def test_default_prepare_inputs_path_remains_unchanged_when_disabled(monkeypatch):
    monkeypatch.delenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", raising=False)
    batch = _make_input_batch()
    before = batch.block_table[0].get_row_block_ids(0)
    runner = FakeRunner(input_batch=batch)

    summary = GPUModelRunner._maybe_apply_kivo_runtime_block_table_before_slot_mapping(
        runner, 1
    )

    assert summary.enabled is False
    assert batch.block_table[0].get_row_block_ids(0) == before
