from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.kivo_vd import run_source_s3_3b_shadow_kv_block_sketch as runner
from scripts.kivo_vd import (
    validate_source_s3_3b_shadow_kv_block_sketch as validator,
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


def _good_report() -> dict[str, object]:
    return {
        "total_prompts": 1,
        "baseline_success_count": 1,
        "shadow_success_count": 1,
        "output_changed_count": 0,
        "sketch_computed_event_count": 1,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "prompt_results": [
            {
                "baseline_status": "succeeded",
                "shadow_status": "succeeded",
                "output_changed": False,
            }
        ],
    }


def _good_event() -> dict[str, object]:
    return {
        "schema_version": validator.SCHEMA,
        "policy_name": validator.POLICY,
        "hook_point": "unified_attention_with_output",
        "sketch_source": "kv_cache",
        "sketch_method": "random_projection_l2",
        "candidate_block_count": 2,
        "selected_block_count": 1,
        "excluded_block_count": 1,
        "block_sketch_sample": [
            {
                "block_id": 7,
                "k_l2_norm": 1.0,
                "v_l2_norm": 1.5,
                "score": 2.5,
                "k_sketch_sample": [0.1] * 8,
                "v_sketch_sample": [0.2] * 8,
            }
        ],
        "sketch_computed": True,
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
    }


def test_runner_parse_args_includes_sketch_controls():
    args = runner._parse_args(
        ["--sketch-dim", "8", "--max-sketch-blocks", "4", "--budget-ratio", "0.5"]
    )
    assert args.sketch_dim == 8
    assert args.max_sketch_blocks == 4
    assert args.budget_ratio == 0.5


def test_summarize_records_counts_computed_and_blocked():
    summary = runner.summarize_records(
        [
            {"schema_version": "other"},
            _good_event(),
            {**_good_event(), "sketch_computed": False, "block_sketch_sample": []},
        ]
    )
    assert summary["records_written"] == 2
    assert summary["ignored_non_s3_events"] == 1
    assert summary["sketch_computed_event_count"] == 1
    assert summary["sketch_blocked_event_count"] == 1


def test_shadow_sketch_record_computes_real_block_sketch(monkeypatch):
    torch = pytest.importorskip("torch")
    kv_cache = torch.arange(
        8 * 2 * 4 * 2 * 4,
        dtype=torch.bfloat16,
    ).reshape(8, 2, 4, 2, 4)
    slot_mapping = torch.tensor([17], dtype=torch.int64)
    layer = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
    )
    monkeypatch.setenv("KIVO_SOURCE_SKETCH_DIM", "4")
    monkeypatch.setenv("KIVO_SOURCE_MAX_SKETCH_BLOCKS", "3")
    monkeypatch.setenv("KIVO_SOURCE_BUDGET_RATIO", "0.5")
    monkeypatch.setenv("KIVO_SOURCE_SKETCH_SEED", "123")
    record = observer.build_shadow_kv_block_sketch_record(
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
    assert record["sketch_computed"] is True
    assert record["current_slot_id"] == 17
    assert record["current_physical_block_id"] == 4
    assert record["candidate_block_count"] == 3
    assert record["selected_block_count"] >= 1
    assert len(record["block_sketch_sample"]) == 3
    assert len(record["block_sketch_sample"][0]["k_sketch_sample"]) == 4


def test_shadow_sketch_record_blocks_on_bad_layout():
    record = observer.build_shadow_kv_block_sketch_record(
        hook_point="unified_attention_with_output",
        layer_name="transformer.h.1.attn",
        attn_layer=SimpleNamespace(attn_backend=None),
        query=None,
        key=None,
        value=None,
        kv_cache=FakeTensor((4, 4, 4, 4)),
        attn_metadata=None,
        slot_mapping=FakeTensor((1,), dtype="torch.int64"),
    )
    assert record["sketch_computed"] is False
    assert record["sketch_blocker_reason"] == "kv_cache ndim is not 5"


def test_validator_passes_and_filters_mixed_schemas():
    result = validator.validate_shadow_kv_block_sketch(
        _good_report(),
        [{"schema_version": "other"}, _good_event()],
    )
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 2
    assert result["total_s3_3b_events"] == 1
    assert result["ignored_non_s3_events"] == 1


def test_validator_rejects_missing_sketches():
    event = _good_event()
    event["block_sketch_sample"] = []
    result = validator.validate_shadow_kv_block_sketch(_good_report(), [event])
    assert result["validation_passed"] is False
    assert any("block_sketch_sample" in error for error in result["errors"])


def test_validator_rejects_runtime_claims():
    event = _good_event()
    event["active_routing"] = True
    result = validator.validate_shadow_kv_block_sketch(_good_report(), [event])
    assert result["validation_passed"] is False
    assert any("active_routing must be false" in error for error in result["errors"])
