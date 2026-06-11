from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.kivo_vd import (
    run_source_s3_3c_active_sketch_kv_metadata_alias as runner,
)
from scripts.kivo_vd import (
    validate_source_s3_3c_active_sketch_kv_metadata_alias as validator,
)


def _good_plan_event() -> dict[str, object]:
    return {
        "schema_version": validator.PLAN_SCHEMA,
        "policy_name": "active_sketch_kv_metadata_alias",
        "sketch_computed": True,
        "selected_attention_claim_allowed": False,
        "measured_runtime_reduction": False,
        "performance_claim_allowed": False,
    }


def _good_metadata_event(*, applied: bool = True) -> dict[str, object]:
    return {
        "schema_version": validator.METADATA_SCHEMA,
        "policy_name": "active_sketch_kv_metadata_alias",
        "sketch_plan_used": True,
        "mutation_attempted": True,
        "mutation_applied": applied,
        "active_routing": applied,
        "alias_pairs_sample": (
            [{"excluded_block_id": 5, "alias_target_block_id": 7}]
            if applied
            else []
        ),
        "alias_target_block_id": 7 if applied else None,
        "selected_attention_claim_allowed": False,
        "measured_runtime_reduction": False,
        "performance_claim_allowed": False,
    }


def _good_report() -> dict[str, object]:
    return {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "active_success_count": 1,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "prompt_results": [
            {
                "baseline_status": "succeeded",
                "active_status": "succeeded",
            }
        ],
    }


def test_parse_args_includes_active_filter_mode():
    args = runner._parse_args(["--active-filter-mode", runner.ACTIVE_FILTER_MODE])
    assert args.active_filter_mode == runner.ACTIVE_FILTER_MODE


def test_summarize_records_splits_plan_and_metadata_events():
    summary = runner.summarize_records(
        [{"schema_version": "other"}, _good_plan_event(), _good_metadata_event()]
    )
    assert summary["total_s3_3c_sketch_plan_events"] == 1
    assert summary["total_s3_3c_metadata_alias_events"] == 1
    assert summary["mutation_applied_event_count"] == 1
    assert summary["ignored_non_s3_events"] == 1


def test_validate_active_sketch_alias_passes_for_good_events():
    result = validator.validate_active_sketch_alias(
        _good_report(),
        [_good_plan_event(), _good_metadata_event()],
    )
    assert result["validation_passed"] is True
    assert result["total_s3_3c_sketch_plan_events"] == 1
    assert result["total_s3_3c_metadata_alias_events"] == 1


def test_validate_active_sketch_alias_rejects_applied_without_plan_use():
    event = _good_metadata_event()
    event["sketch_plan_used"] = False
    result = validator.validate_active_sketch_alias(
        _good_report(),
        [_good_plan_event(), event],
    )
    assert result["validation_passed"] is False
    assert any("requires sketch_plan_used" in error for error in result["errors"])


def test_validate_active_sketch_alias_rejects_invalid_alias_target():
    event = _good_metadata_event()
    event["alias_target_block_id"] = -1
    result = validator.validate_active_sketch_alias(
        _good_report(),
        [_good_plan_event(), event],
    )
    assert result["validation_passed"] is False
    assert any("invalid alias target" in error for error in result["errors"])


def test_active_sketch_helper_uses_latest_plan(monkeypatch):
    torch = pytest.importorskip("torch")
    from vllm.v1.worker import kivo_attention_metadata_observer as metadata_observer
    from vllm.v1.worker import kivo_attention_tensor_observer as tensor_observer

    kv_cache = torch.arange(
        8 * 2 * 4 * 2 * 4,
        dtype=torch.bfloat16,
    ).reshape(8, 2, 4, 2, 4)
    slot_mapping = torch.tensor([17], dtype=torch.int64)
    layer = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
    )
    plan_record = tensor_observer.build_active_sketch_plan_record(
        hook_point="unified_attention_with_output",
        layer_name="transformer.h.1.attn",
        attn_layer=layer,
        query=None,
        key=None,
        value=None,
        kv_cache=kv_cache,
        attn_metadata=None,
        slot_mapping=slot_mapping,
    )
    tensor_observer._store_latest_sketch_plan(plan_record)

    block_table = torch.tensor([[2, 3, 4, 0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        block_table_tensor=block_table,
        slot_mapping=torch.tensor(list(range(2 * 4, 5 * 4)), dtype=torch.int64),
        query_start_loc=torch.tensor([0, 12], dtype=torch.int32),
        seq_lens=torch.tensor([12], dtype=torch.int32),
        positions=torch.arange(12, dtype=torch.int64),
        max_query_len=12,
        max_seq_len=12,
    )
    monkeypatch.setenv(
        "KIVO_SOURCE_ACTIVE_FILTER_MODE",
        metadata_observer._ACTIVE_SKETCH_FILTER_MODE,
    )
    record = metadata_observer._build_active_sketch_metadata_alias_record(
        hook_point="_build_attention_metadata",
        kv_cache_group_id=0,
        common_attn_metadata=metadata,
        kv_cache_spec=SimpleNamespace(block_size=4),
        env_debug={
            "kivo_source_enable_seen": True,
            "kivo_source_policy_seen": metadata_observer._ACTIVE_SKETCH_POLICY_NAME,
            "observe_path_present": True,
        },
    )
    assert record["sketch_plan_used"] is True
    assert record["mutation_applied"] is True
    assert record["alias_pairs_sample"]
    assert metadata.block_table_tensor is not block_table

