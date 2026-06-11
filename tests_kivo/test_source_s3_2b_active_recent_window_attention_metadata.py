from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.kivo_vd import (
    run_source_s3_2b_active_recent_window_attention_metadata as runner,
)
from scripts.kivo_vd import (
    validate_source_s3_2b_active_recent_window_attention_metadata as validator,
)


def _good_event(*, mutation_applied: bool = True) -> dict[str, object]:
    return {
        "schema_version": validator.SCHEMA,
        "policy_name": validator.POLICY_NAME,
        "hook_point": "_build_attention_metadata",
        "active_filter_mode": validator.ACTIVE_FILTER_MODE,
        "selection_policy_name": validator.SELECTION_POLICY_NAME,
        "original_seq_len": 48,
        "modified_seq_len": 16 if mutation_applied else None,
        "original_visible_block_count": 3,
        "selected_block_count": 1,
        "excluded_block_count": 2,
        "selected_block_ids_sample": [7],
        "excluded_block_ids_sample": [5, 6],
        "keep_recent_blocks": 1,
        "selected_token_length": 16 if mutation_applied else None,
        "theoretical_attention_visible_block_reduction": 2,
        "theoretical_attention_visible_block_reduction_ratio": 2.0 / 3.0,
        "mutation_attempted": True,
        "mutation_applied": mutation_applied,
        "mutation_blocker_reason": (
            None if mutation_applied else "visible_blocks_not_above_budget"
        ),
        "active_routing": mutation_applied,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
    }


def _good_report() -> dict[str, object]:
    return {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "active_success_count": 1,
        "mutation_attempted_event_count": 1,
        "mutation_applied_event_count": 1,
        "active_routing_event_count": 1,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_2b_active_recent_window_passed": True,
        "prompt_results": [
            {
                "baseline_status": "succeeded",
                "active_status": "succeeded",
            }
        ],
    }


def test_parse_args_includes_recent_window_mode():
    args = runner._parse_args(
        [
            "--active-filter-mode",
            runner.ACTIVE_FILTER_MODE,
            "--keep-recent-blocks",
            "1",
        ]
    )
    assert args.active_filter_mode == runner.ACTIVE_FILTER_MODE
    assert args.keep_recent_blocks == 1


def test_summarize_records_tracks_recent_window_reduction():
    summary = runner.summarize_records([_good_event()])
    assert summary["records_written"] == 1
    assert summary["mutation_attempted_event_count"] == 1
    assert summary["mutation_applied_event_count"] == 1
    assert summary["max_original_visible_block_count"] == 3


def test_validate_recent_window_passes_and_filters_mixed_schemas():
    result = validator.validate_recent_window(
        _good_report(),
        [
            {"schema_version": "kivo_source_s3_2a_active_selected_attention_metadata_v1"},
            _good_event(),
        ],
    )
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 2
    assert result["total_s3_2b_events"] == 1
    assert result["ignored_non_s3_events"] == 1


def test_validate_recent_window_rejects_missing_seq_len_compaction():
    event = _good_event()
    event["modified_seq_len"] = 48
    result = validator.validate_recent_window(_good_report(), [event])
    assert result["validation_passed"] is False
    assert any("modified_seq_len < original_seq_len" in error for error in result["errors"])


def test_recent_window_helper_clones_and_compacts(monkeypatch):
    torch = pytest.importorskip("torch")
    from vllm.v1.worker import kivo_attention_metadata_observer as observer

    block_table = torch.tensor([[5, 6, 7, 9]], dtype=torch.int32)
    seq_lens = torch.tensor([48], dtype=torch.int32)
    metadata = SimpleNamespace(
        block_table_tensor=block_table,
        slot_mapping=torch.tensor(
            list(range(5 * 16, 8 * 16)),
            dtype=torch.int64,
        ),
        query_start_loc=torch.tensor([0, 48], dtype=torch.int32),
        seq_lens=seq_lens,
        positions=torch.arange(48, dtype=torch.int64),
        max_query_len=48,
        max_seq_len=48,
    )
    kv_cache_spec = SimpleNamespace(block_size=16)
    monkeypatch.setenv(
        "KIVO_SOURCE_ACTIVE_FILTER_MODE",
        observer._RECENT_WINDOW_FILTER_MODE,
    )
    monkeypatch.setenv("KIVO_SOURCE_KEEP_RECENT_BLOCKS", "1")

    record = observer._build_active_recent_window_record(
        hook_point="_build_attention_metadata",
        kv_cache_group_id=0,
        common_attn_metadata=metadata,
        kv_cache_spec=kv_cache_spec,
        env_debug={
            "kivo_source_enable_seen": True,
            "kivo_source_policy_seen": observer._RECENT_WINDOW_POLICY_NAME,
            "observe_path_present": True,
        },
    )

    assert record["mutation_applied"] is True
    assert record["modified_seq_len"] == 16
    assert record["selected_block_ids_sample"] == [7]
    assert torch.equal(block_table, torch.tensor([[5, 6, 7, 9]], dtype=torch.int32))
    assert metadata.block_table_tensor is not block_table
    assert metadata.block_table_tensor[0, 0].item() == 7
    assert metadata.seq_lens is not seq_lens
    assert metadata.seq_lens[0].item() == 16
