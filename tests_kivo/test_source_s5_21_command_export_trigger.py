from __future__ import annotations

from vllm.v1.core.kivo_demotion_counters import (
    get_kivo_demotion_counters_snapshot,
    reset_kivo_demotion_counters,
)
from vllm.v1.worker.kivo_runtime_block_table_apply import (
    maybe_build_kivo_demotion_command_after_runtime_apply,
)


def _call_helper(**overrides):
    kwargs = {
        "request_id": "req0",
        "visible_before_block_ids": (10, 11, 12, 13),
        "visible_after_block_ids": (12, 13),
        "candidate_demote_block_ids": (10, 11),
        "protected_block_ids": (12, 13),
        "apply_summary_present": True,
        "block_table_applied": True,
        "slot_mapping_refresh_guaranteed": True,
        "filtered_row_changed": True,
        "keep_recent_blocks": 1,
        "policy": "recent_only",
    }
    kwargs.update(overrides)
    return maybe_build_kivo_demotion_command_after_runtime_apply(**kwargs)


def test_successful_block_table_apply_enters_command_export_path(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    _call_helper()
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["demotion_command_export_path_entered"] == 1


def test_no_apply_summary_increments_skipped_no_apply_summary(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    result = _call_helper(apply_summary_present=False)
    snapshot = get_kivo_demotion_counters_snapshot()
    assert result.command is None
    assert snapshot["demotion_command_export_skipped_no_apply_summary"] == 1


def test_unsuccessful_apply_increments_skipped_apply_not_successful(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    result = _call_helper(block_table_applied=False)
    snapshot = get_kivo_demotion_counters_snapshot()
    assert result.command is None
    assert (
        snapshot["demotion_command_export_skipped_apply_not_successful"] == 1
    )


def test_missing_request_id_increments_skipped_no_request_id(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    result = _call_helper(request_id=None)
    snapshot = get_kivo_demotion_counters_snapshot()
    assert result.command is None
    assert snapshot["demotion_command_export_skipped_no_request_id"] == 1


def test_unchanged_row_no_candidate_demote_ids_increments_skip(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    result = _call_helper(
        visible_after_block_ids=(10, 11, 12, 13),
        candidate_demote_block_ids=(),
        filtered_row_changed=False,
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert result.command is None
    assert snapshot["demotion_command_export_skipped_no_candidate_demote_ids"] == 1


def test_changed_row_with_demote_ids_builds_command(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    result = _call_helper()
    snapshot = get_kivo_demotion_counters_snapshot()
    assert result.command is not None
    assert snapshot["demotion_command_export_attempted"] == 1
    assert snapshot["demotion_command_export_succeeded"] == 1


def test_built_command_increments_worker_envelope_counter(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    result = _call_helper()
    snapshot = get_kivo_demotion_counters_snapshot()
    assert result.command is not None
    assert snapshot["worker_envelopes_built"] == 1


def test_no_req_to_blocks_removal_and_no_free_to_pool(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    _call_helper()
    snapshot = get_kivo_demotion_counters_snapshot()
    assert snapshot["req_to_blocks_removed"] == 0
    assert snapshot["free_to_pool_calls"] == 0
