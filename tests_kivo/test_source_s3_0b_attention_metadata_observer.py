from __future__ import annotations

import json
from pathlib import Path

from scripts.kivo_vd import (
    run_source_s3_0b_attention_metadata_observer as runner,
)
from scripts.kivo_vd import (
    validate_source_s3_0b_attention_metadata_observer as validator,
)


def _good_event() -> dict[str, object]:
    return {
        "schema_version": validator.SCHEMA,
        "policy_name": "observe_attention_metadata",
        "hook_point": "build_attn_metadata",
        "mutation_attempted": False,
        "mutation_applied": False,
        "selected_block_count": None,
        "active_routing": False,
        "runtime_behavior_changed": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "block_table_tensor_present": True,
        "block_table_tensor_shape": [3, 4],
        "block_table_tensor_dtype": "torch.int32",
        "block_table_tensor_device": "cpu",
        "slot_mapping_present": True,
        "slot_mapping_shape": [48],
        "slot_mapping_dtype": "torch.int64",
        "slot_mapping_device": "cpu",
        "query_start_loc_shape": [4],
        "seq_lens_shape": [3],
        "positions_shape": [48],
        "max_query_len": 8,
        "max_seq_len": 16,
        "visible_block_count_estimate": 3,
        "visible_block_count_estimate_caveat": None,
        "block_table_tensor_sample": [0, 1, 2, 3],
        "slot_mapping_sample": [0, 1, 2, 3],
        "caveats": [],
    }


def _good_report() -> dict[str, object]:
    prompt_results = [
        {
            "prompt_index": 0,
            "prompt": "a",
            "baseline_status": "succeeded",
            "observer_status": "succeeded",
            "baseline_output": "x",
            "observer_output": "x",
            "baseline_error": None,
            "observer_error": None,
            "output_changed": False,
            "records_written": 1,
            "max_block_table_rows": 3,
            "max_block_table_cols": 4,
            "max_slot_mapping_len": 48,
        },
        {
            "prompt_index": 1,
            "prompt": "b",
            "baseline_status": "succeeded",
            "observer_status": "succeeded",
            "baseline_output": "y",
            "observer_output": "y",
            "baseline_error": None,
            "observer_error": None,
            "output_changed": False,
            "records_written": 1,
            "max_block_table_rows": 3,
            "max_block_table_cols": 4,
            "max_slot_mapping_len": 48,
        },
        {
            "prompt_index": 2,
            "prompt": "c",
            "baseline_status": "succeeded",
            "observer_status": "succeeded",
            "baseline_output": "z",
            "observer_output": "z",
            "baseline_error": None,
            "observer_error": None,
            "output_changed": False,
            "records_written": 2,
            "max_block_table_rows": 4,
            "max_block_table_cols": 5,
            "max_slot_mapping_len": 64,
        },
    ]
    return {
        "model": "gpt2",
        "max_tokens": 8,
        "gpu_memory_utilization": 0.1,
        "max_model_len": 256,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
        "total_prompts": 3,
        "baseline_success_count": 3,
        "observer_success_count": 3,
        "output_changed_count": 0,
        "total_events": 4,
        "metadata_observed_prompt_count": 3,
        "max_block_table_rows": 4,
        "max_block_table_cols": 5,
        "max_slot_mapping_len": 64,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_0b_observer_passed": True,
        "prompt_results": prompt_results,
        "events_jsonl": "events.jsonl",
    }


def test_parse_args_includes_observer_paths():
    args = runner._parse_args(
        [
            "--events-jsonl",
            "foo.jsonl",
            "--output-json",
            "bar.json",
            "--output-md",
            "baz.md",
        ]
    )
    assert args.events_jsonl == "foo.jsonl"
    assert args.output_json == "bar.json"
    assert args.output_md == "baz.md"


def test_summarize_records_counts_shapes():
    summary = runner.summarize_records(
        [
            {
                "block_table_tensor_shape": [3, 4],
                "slot_mapping_shape": [48],
                "mutation_attempted": False,
                "mutation_applied": False,
                "active_routing": False,
                "runtime_behavior_changed": False,
            },
            {
                "block_table_tensor_shape": [4, 5],
                "slot_mapping_shape": [64],
                "mutation_attempted": False,
                "mutation_applied": False,
                "active_routing": False,
                "runtime_behavior_changed": False,
            },
        ]
    )
    assert summary["records_written"] == 2
    assert summary["max_block_table_rows"] == 4
    assert summary["max_block_table_cols"] == 5
    assert summary["max_slot_mapping_len"] == 64


def test_validate_observer_passes_for_good_report_and_events():
    report = _good_report()
    events = [_good_event(), _good_event()]
    result = validator.validate_observer(report, events)
    assert result["validation_passed"] is True
    assert result["total_prompts"] == 3
    assert result["total_raw_events"] == 2
    assert result["total_s3_0b_events"] == 2
    assert result["ignored_non_s3_events"] == 0


def test_validate_observer_filters_mixed_schema_events():
    report = _good_report()
    mixed_events = [
        {
            "schema_version": "kivo_source_s1_block_table_v1",
            "policy_name": "mask_last_valid_slot",
            "mutation_attempted": False,
            "mutation_applied": False,
        },
        _good_event(),
        {
            "schema_version": "kivo_source_s1_block_table_v1",
            "policy_name": "mask_last_valid_slot",
            "mutation_attempted": False,
            "mutation_applied": False,
        },
        _good_event(),
    ]
    result = validator.validate_observer(report, mixed_events)
    assert result["validation_passed"] is True
    assert result["total_raw_events"] == 4
    assert result["total_s3_0b_events"] == 2
    assert result["ignored_non_s3_events"] == 2


def test_validate_observer_rejects_claims_and_runtime_changes():
    report = _good_report()
    report["measured_runtime_reduction"] = True
    events = [_good_event()]
    events[0]["active_routing"] = True
    events[0]["runtime_behavior_changed"] = True
    result = validator.validate_observer(report, events)
    assert result["validation_passed"] is False
    assert any("measured_runtime_reduction must be false" in e for e in result["errors"])
    assert any("must not claim active routing" in e for e in result["errors"])
    assert any(
        "must not claim runtime behavior change" in e for e in result["errors"]
    )


def test_validate_observer_rejects_missing_or_empty_events(tmp_path: Path):
    report = _good_report()
    result = validator.validate_observer(report, [])
    assert result["validation_passed"] is False
    assert any("total_raw_events must be > 0" in e for e in result["errors"])


def test_validate_observer_rejects_mixed_events_with_no_s3_records():
    report = _good_report()
    result = validator.validate_observer(
        report,
        [
            {
                "schema_version": "kivo_source_s1_block_table_v1",
                "policy_name": "mask_last_valid_slot",
                "mutation_attempted": False,
                "mutation_applied": False,
            },
            {
                "schema_version": "kivo_source_s1_block_table_v1",
                "policy_name": "mask_last_valid_slot",
                "mutation_attempted": False,
                "mutation_applied": False,
            },
        ],
    )
    assert result["validation_passed"] is False
    assert any("total_s3_0b_events must be > 0" in e for e in result["errors"])


def test_validator_cli_help_mentions_events_jsonl(capsys):
    try:
        validator._parse_args(["--help"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "--events-jsonl" in captured.out
