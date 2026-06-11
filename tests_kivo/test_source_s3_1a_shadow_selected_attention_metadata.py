from __future__ import annotations

from scripts.kivo_vd import (
    run_source_s3_1a_shadow_selected_attention_metadata as runner,
)
from scripts.kivo_vd import (
    validate_source_s3_1a_shadow_selected_attention_metadata as validator,
)


def _good_event(
    *,
    visible_block_count_estimate: int = 3,
    selected_block_count: int = 2,
    excluded_block_count: int = 1,
    keep_recent_blocks: int = 1,
) -> dict[str, object]:
    ratio = (
        float(excluded_block_count) / float(visible_block_count_estimate)
        if visible_block_count_estimate > 0
        else 0.0
    )
    return {
        "schema_version": validator.SCHEMA,
        "policy_name": "shadow_selected_attention_metadata",
        "hook_point": "_build_attention_metadata",
        "sequence_id": 1,
        "pid": 123,
        "block_size": 16,
        "block_table_tensor_present": True,
        "block_table_tensor_shape": [1, 3],
        "block_table_tensor_dtype": "torch.int32",
        "block_table_tensor_device": "cuda:0",
        "slot_mapping_present": True,
        "slot_mapping_shape": [48],
        "slot_mapping_dtype": "torch.int64",
        "slot_mapping_device": "cuda:0",
        "query_start_loc_shape": [2],
        "seq_lens_shape": [1],
        "positions_shape": [48],
        "max_query_len": 8,
        "max_seq_len": 32,
        "visible_block_count_estimate": visible_block_count_estimate,
        "visible_block_ids_sample": [0, 1, 2][:visible_block_count_estimate],
        "selected_block_count": selected_block_count,
        "selected_block_ids_sample": [1, 2][:selected_block_count],
        "excluded_block_count": excluded_block_count,
        "excluded_block_ids_sample": [0][:excluded_block_count],
        "theoretical_attention_visible_block_reduction": excluded_block_count,
        "theoretical_attention_visible_block_reduction_ratio": ratio,
        "selection_policy_name": "deterministic_placeholder_block_score",
        "budget_ratio": 0.5,
        "keep_recent_blocks": keep_recent_blocks,
        "mutation_attempted": False,
        "mutation_applied": False,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "caveats": [],
    }


def _good_report() -> dict[str, object]:
    prompt_results = [
        {
            "prompt_index": 0,
            "prompt": "a",
            "baseline_status": "succeeded",
            "shadow_status": "succeeded",
            "baseline_output": "x",
            "shadow_output": "x",
            "baseline_error": None,
            "shadow_error": None,
            "output_changed": False,
            "records_written": 1,
            "excluded_event_count": 0,
            "max_visible_block_count": 1,
            "max_selected_block_count": 1,
            "max_excluded_block_count": 0,
            "max_theoretical_attention_visible_block_reduction": 0,
            "max_theoretical_attention_visible_block_reduction_ratio": 0.0,
        },
        {
            "prompt_index": 1,
            "prompt": "b",
            "baseline_status": "succeeded",
            "shadow_status": "succeeded",
            "baseline_output": "y",
            "shadow_output": "y",
            "baseline_error": None,
            "shadow_error": None,
            "output_changed": False,
            "records_written": 1,
            "excluded_event_count": 1,
            "max_visible_block_count": 2,
            "max_selected_block_count": 1,
            "max_excluded_block_count": 1,
            "max_theoretical_attention_visible_block_reduction": 1,
            "max_theoretical_attention_visible_block_reduction_ratio": 0.5,
        },
        {
            "prompt_index": 2,
            "prompt": "c",
            "baseline_status": "succeeded",
            "shadow_status": "succeeded",
            "baseline_output": "z",
            "shadow_output": "z",
            "baseline_error": None,
            "shadow_error": None,
            "output_changed": False,
            "records_written": 1,
            "excluded_event_count": 1,
            "max_visible_block_count": 3,
            "max_selected_block_count": 2,
            "max_excluded_block_count": 1,
            "max_theoretical_attention_visible_block_reduction": 1,
            "max_theoretical_attention_visible_block_reduction_ratio": (
                1.0 / 3.0
            ),
        },
    ]
    return {
        "model": "gpt2",
        "max_tokens": 8,
        "gpu_memory_utilization": 0.1,
        "max_model_len": 256,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
        "budget_ratio": 0.5,
        "keep_recent_blocks": 1,
        "total_prompts": 3,
        "baseline_success_count": 3,
        "shadow_success_count": 3,
        "output_changed_count": 0,
        "total_events": 3,
        "shadow_plan_prompt_count": 3,
        "excluded_block_prompt_count": 2,
        "max_visible_block_count": 3,
        "max_selected_block_count": 2,
        "max_excluded_block_count": 1,
        "max_theoretical_attention_visible_block_reduction": 1,
        "max_theoretical_attention_visible_block_reduction_ratio": 0.5,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_1a_shadow_plan_passed": True,
        "prompt_results": prompt_results,
        "events_jsonl": "events.jsonl",
    }


def test_parse_args_includes_shadow_plan_flags():
    args = runner._parse_args(
        [
            "--events-jsonl",
            "foo.jsonl",
            "--budget-ratio",
            "0.75",
            "--keep-recent-blocks",
            "2",
        ]
    )
    assert args.events_jsonl == "foo.jsonl"
    assert args.budget_ratio == 0.75
    assert args.keep_recent_blocks == 2


def test_summarize_records_tracks_exclusion_stats():
    summary = runner.summarize_records(
        [
            _good_event(visible_block_count_estimate=1, selected_block_count=1, excluded_block_count=0),
            _good_event(visible_block_count_estimate=3, selected_block_count=2, excluded_block_count=1),
        ]
    )
    assert summary["records_written"] == 2
    assert summary["excluded_event_count"] == 1
    assert summary["max_visible_block_count"] == 3
    assert summary["max_selected_block_count"] == 2
    assert summary["max_excluded_block_count"] == 1


def test_validate_shadow_plan_passes_for_mixed_prompt_behavior():
    report = _good_report()
    events = [
        _good_event(
            visible_block_count_estimate=1,
            selected_block_count=1,
            excluded_block_count=0,
            keep_recent_blocks=1,
        ),
        _good_event(
            visible_block_count_estimate=2,
            selected_block_count=1,
            excluded_block_count=1,
            keep_recent_blocks=1,
        ),
        _good_event(
            visible_block_count_estimate=3,
            selected_block_count=2,
            excluded_block_count=1,
            keep_recent_blocks=1,
        ),
    ]
    result = validator.validate_shadow_plan(report, events)
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 3
    assert result["total_s3_1a_events"] == 3
    assert result["ignored_non_s3_events"] == 0


def test_validate_shadow_plan_filters_mixed_schema_events():
    report = _good_report()
    events = [
        {
            "schema_version": "kivo_source_s3_0b_attention_metadata_observer_v1",
            "policy_name": "observe_attention_metadata",
        },
        _good_event(),
    ]
    result = validator.validate_shadow_plan(report, events)
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 2
    assert result["total_s3_1a_events"] == 1
    assert result["ignored_non_s3_events"] == 1


def test_validate_shadow_plan_rejects_when_no_shadow_events_exist():
    report = _good_report()
    result = validator.validate_shadow_plan(
        report,
        [
            {
                "schema_version": "kivo_source_s3_0b_attention_metadata_observer_v1",
                "policy_name": "observe_attention_metadata",
            }
        ],
    )
    assert result["validation_passed"] is False
    assert any("total_s3_1a_events must be > 0" in e for e in result["errors"])


def test_validate_shadow_plan_rejects_missing_exclusion_for_eligible_events():
    report = _good_report()
    events = [
        _good_event(
            visible_block_count_estimate=3,
            selected_block_count=3,
            excluded_block_count=0,
            keep_recent_blocks=1,
        )
    ]
    result = validator.validate_shadow_plan(report, events)
    assert result["validation_passed"] is False
    assert any("at least one event must exclude blocks" in e for e in result["errors"])


def test_validate_shadow_plan_rejects_runtime_claims():
    report = _good_report()
    report["performance_claim_allowed"] = True
    events = [_good_event()]
    events[0]["runtime_behavior_changed"] = True
    result = validator.validate_shadow_plan(report, events)
    assert result["validation_passed"] is False
    assert any("performance_claim_allowed must be false" in e for e in result["errors"])
    assert any(
        "must not claim runtime behavior change" in e for e in result["errors"]
    )


def test_validator_cli_help_mentions_events_jsonl(capsys):
    try:
        validator._parse_args(["--help"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "--events-jsonl" in captured.out
