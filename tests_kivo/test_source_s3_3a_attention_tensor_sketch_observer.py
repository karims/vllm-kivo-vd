from __future__ import annotations

from types import SimpleNamespace

from scripts.kivo_vd import (
    run_source_s3_3a_attention_tensor_sketch_observer as runner,
)
from scripts.kivo_vd import (
    validate_source_s3_3a_attention_tensor_sketch_observer as validator,
)
from vllm.v1.worker import kivo_attention_tensor_observer as observer


class FakeTensor:
    def __init__(self, shape, dtype="torch.float16", device="cuda:0"):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total


def _good_event() -> dict[str, object]:
    return {
        "schema_version": validator.SCHEMA,
        "policy_name": validator.POLICY,
        "hook_point": "unified_attention_with_output",
        "query_present": True,
        "key_present": True,
        "value_present": True,
        "kv_cache_present": True,
        "can_build_query_sketch": True,
        "can_build_key_sketch": True,
        "can_build_value_sketch": True,
        "can_build_kv_block_sketch": True,
        "recommended_sketch_source": "kv_cache",
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
    }


def _good_report() -> dict[str, object]:
    return {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "observer_success_count": 1,
        "output_changed_count": 0,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "prompt_results": [
            {
                "baseline_status": "succeeded",
                "observer_status": "succeeded",
                "output_changed": False,
            }
        ],
    }


def test_build_record_observes_qkv_cache_without_values():
    metadata = SimpleNamespace(
        block_table=FakeTensor((1, 4), dtype="torch.int32"),
    )
    layer = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
    )
    record = observer.build_attention_tensor_record(
        hook_point=observer.HOOK_POINT,
        layer_name="transformer.h.2.attn",
        attn_layer=layer,
        query=FakeTensor((8, 12, 64)),
        key=FakeTensor((8, 12, 64)),
        value=FakeTensor((8, 12, 64)),
        kv_cache=FakeTensor((64, 2, 16, 12, 64)),
        attn_metadata=metadata,
        slot_mapping=FakeTensor((8,), dtype="torch.int64"),
    )
    assert record["layer_index"] == 2
    assert record["query_shape"] == [8, 12, 64]
    assert record["kv_cache_shape"] == [64, 2, 16, 12, 64]
    assert record["can_build_kv_block_sketch"] is True
    assert record["recommended_sketch_source"] == "kv_cache"
    assert "query_sample" not in record


def test_summarize_records_filters_mixed_schemas():
    summary = runner.summarize_records([
        {"schema_version": "other"},
        _good_event(),
    ])
    assert summary["records_written"] == 1
    assert summary["raw_records_written"] == 2
    assert summary["ignored_non_s3_events"] == 1
    assert summary["query_observed_event_count"] == 1


def test_validator_passes_and_filters_mixed_schemas():
    result = validator.validate_observer(
        _good_report(),
        [{"schema_version": "other"}, _good_event()],
    )
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 2
    assert result["total_s3_3a_events"] == 1
    assert result["ignored_non_s3_events"] == 1


def test_validator_rejects_no_tensor_visibility():
    event = _good_event()
    for field in [
        "query_present",
        "key_present",
        "value_present",
        "kv_cache_present",
    ]:
        event[field] = False
    result = validator.validate_observer(_good_report(), [event])
    assert result["validation_passed"] is False
    assert any("tensor source" in error for error in result["errors"])


def test_validator_rejects_runtime_claims():
    event = _good_event()
    event["mutation_applied"] = True
    event["active_routing"] = True
    result = validator.validate_observer(_good_report(), [event])
    assert result["validation_passed"] is False
    assert any("mutation_applied must be false" in e for e in result["errors"])
    assert any("active_routing must be false" in e for e in result["errors"])
