from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.kivo_vd import (
    run_source_s3_2a_active_selected_attention_metadata as runner,
)
from scripts.kivo_vd import (
    validate_source_s3_2a_active_selected_attention_metadata as validator,
)


def _good_event(*, mutation_applied: bool = True) -> dict[str, object]:
    return {
        "schema_version": validator.SCHEMA,
        "policy_name": validator.POLICY_NAME,
        "hook_point": "_build_attention_metadata",
        "selection_policy_name": validator.SELECTION_POLICY_NAME,
        "active_filter_mode": validator.ACTIVE_FILTER_MODE,
        "visible_block_count_estimate": 3,
        "visible_block_ids_sample": [5, 6, 7],
        "selected_block_count": 2,
        "selected_block_ids_sample": [6, 7],
        "excluded_block_count": 1,
        "excluded_block_ids_sample": [5],
        "aliased_block_count": 1 if mutation_applied else 0,
        "alias_target_block_id": 7 if mutation_applied else None,
        "alias_pairs_sample": (
            [{"excluded_block_id": 5, "alias_target_block_id": 7}]
            if mutation_applied
            else []
        ),
        "mutation_attempted": True,
        "mutation_applied": mutation_applied,
        "mutation_blocker_reason": (
            None if mutation_applied else "no excluded blocks selected for aliasing"
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
        "output_changed_count": 0,
        "mutation_attempted_event_count": 1,
        "mutation_applied_event_count": 1,
        "active_routing_event_count": 1,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_2a_active_metadata_passed": True,
        "prompt_results": [
            {
                "baseline_status": "succeeded",
                "active_status": "succeeded",
            }
        ],
    }


def test_parse_args_includes_active_filter_mode():
    args = runner._parse_args(
        [
            "--active-filter-mode",
            runner.ACTIVE_FILTER_MODE,
            "--budget-ratio",
            "0.5",
        ]
    )
    assert args.active_filter_mode == runner.ACTIVE_FILTER_MODE


def test_summarize_records_tracks_active_mutation():
    summary = runner.summarize_records([_good_event()])
    assert summary["records_written"] == 1
    assert summary["mutation_attempted_event_count"] == 1
    assert summary["mutation_applied_event_count"] == 1
    assert summary["active_routing_event_count"] == 1
    assert summary["max_aliased_block_count"] == 1


def test_validate_active_metadata_passes_and_filters_mixed_schemas():
    result = validator.validate_active_metadata(
        _good_report(),
        [
            {"schema_version": "kivo_source_s3_1b_other"},
            _good_event(),
        ],
    )
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 2
    assert result["total_s3_2a_events"] == 1
    assert result["ignored_non_s3_events"] == 1


def test_validate_active_metadata_rejects_invalid_applied_alias():
    event = _good_event()
    event["alias_target_block_id"] = -1
    event["mutation_blocker_reason"] = "unsafe blocker"
    result = validator.validate_active_metadata(_good_report(), [event])
    assert result["validation_passed"] is False
    assert any("invalid alias target" in error for error in result["errors"])
    assert any("blocker after mutation_applied" in error for error in result["errors"])


def test_active_helper_clones_and_aliases_only_valid_entries(monkeypatch):
    torch = pytest.importorskip("torch")
    from vllm.v1.worker import kivo_attention_metadata_observer as observer

    block_table = torch.tensor([[5, 6, 7, 0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        block_table_tensor=block_table,
        slot_mapping=torch.tensor(
            list(range(5 * 16, 8 * 16)),
            dtype=torch.int64,
        ),
        query_start_loc=torch.tensor([0, 48], dtype=torch.int32),
        seq_lens=torch.tensor([48], dtype=torch.int32),
        positions=torch.arange(48, dtype=torch.int64),
        max_query_len=48,
        max_seq_len=48,
    )
    kv_cache_spec = SimpleNamespace(block_size=16)
    monkeypatch.setenv("KIVO_SOURCE_BUDGET_RATIO", "0.5")
    monkeypatch.setenv("KIVO_SOURCE_KEEP_RECENT_BLOCKS", "1")
    monkeypatch.setenv("KIVO_SOURCE_COVERAGE_WEIGHT", "0.6")
    monkeypatch.setenv("KIVO_SOURCE_RECENCY_WEIGHT", "0.4")
    monkeypatch.setenv(
        "KIVO_SOURCE_ACTIVE_FILTER_MODE",
        observer._ACTIVE_FILTER_MODE,
    )

    record = observer._build_active_selected_attention_record(
        hook_point="_build_attention_metadata",
        kv_cache_group_id=0,
        common_attn_metadata=metadata,
        kv_cache_spec=kv_cache_spec,
        env_debug={
            "kivo_source_enable_seen": True,
            "kivo_source_policy_seen": observer._ACTIVE_POLICY_NAME,
            "observe_path_present": True,
        },
    )

    assert record["mutation_applied"] is True
    assert record["active_routing"] is True
    assert record["aliased_block_count"] == 1
    assert torch.equal(block_table, torch.tensor([[5, 6, 7, 0]], dtype=torch.int32))
    assert metadata.block_table_tensor is not block_table
    assert metadata.block_table_tensor[0, 3].item() == 0
    assert metadata.block_table_tensor[0, :3].tolist().count(
        record["alias_target_block_id"]
    ) >= 2
