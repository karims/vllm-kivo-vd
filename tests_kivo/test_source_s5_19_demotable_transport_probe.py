from __future__ import annotations

from pathlib import Path

from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (
    build_long_context_prompt,
    build_prompts,
    build_summary,
    parse_args as parse_run_args,
)
from scripts.kivo_vd.validate_source_s5_19_demotable_transport_probe import (
    parse_args as parse_validate_args,
    validate_summary,
)


def test_long_prompt_builder_produces_longer_prompt_than_tiny_probe():
    prompt = build_long_context_prompt(repeats=50)
    assert "Question:" in prompt
    assert len(prompt) > len("Kivo transport counter probe prompt one.")


def test_json_summary_schema_contains_required_fields():
    summary = build_summary(
        generation_success=True,
        prompt_count=1,
        prompt_char_lengths=[1234],
        prompt_token_lengths=[256],
        parent_counters={},
        exported_counters=None,
        counter_export_file_found=False,
        counter_export_pid=None,
    )
    for field in [
        "phase",
        "generation_success",
        "prompt_count",
        "prompt_char_lengths",
        "prompt_token_lengths",
        "counters",
        "transport_observed",
        "worker_envelope_observed",
        "scheduler_envelope_observed",
        "core_command_observed",
        "manager_mark_demoted_observed",
        "req_to_blocks_removed",
        "free_to_pool_calls",
        "memory_claim_allowed",
        "free_to_pool_claim_allowed",
    ]:
        assert field in summary


def test_validator_passes_safety_only_case_with_zero_envelopes():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "transport_observed": False,
            "counter_export_file_found": False,
            "req_to_blocks_removed": 0,
            "free_to_pool_calls": 0,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
            "counters": {},
        }
    )
    assert result["validation_passed"] is True
    assert result["transport_observed"] is False
    assert result["reason"] == "no_demotable_blocks_or_runtime_policy_did_not_emit"


def test_validator_passes_observed_transport_case():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "transport_observed": True,
            "counter_export_file_found": True,
            "req_to_blocks_removed": 0,
            "free_to_pool_calls": 0,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
            "counters": {"worker_envelopes_built": 1},
        }
    )
    assert result["validation_passed"] is True
    assert result["transport_observed"] is True


def test_validator_fails_generation_failure():
    result = validate_summary(
        {
            "generation_success": False,
            "prompt_count": 1,
            "transport_observed": False,
            "counter_export_file_found": False,
            "req_to_blocks_removed": 0,
            "free_to_pool_calls": 0,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
            "counters": {},
        }
    )
    assert result["validation_passed"] is False


def test_validator_fails_if_free_to_pool_calls_positive():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "transport_observed": False,
            "counter_export_file_found": False,
            "req_to_blocks_removed": 0,
            "free_to_pool_calls": 1,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
            "counters": {},
        }
    )
    assert result["validation_passed"] is False


def test_validator_fails_if_req_to_blocks_removed_positive():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "transport_observed": False,
            "counter_export_file_found": False,
            "req_to_blocks_removed": 1,
            "free_to_pool_calls": 0,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
            "counters": {},
        }
    )
    assert result["validation_passed"] is False


def test_cli_and_prompt_builders_work():
    run_args = parse_run_args(["--output", str(Path("/tmp/out.json"))])
    validate_args = parse_validate_args(["--input", str(Path("/tmp/out.json"))])
    prompts = build_prompts(repeats=10, num_prompts=2)
    assert run_args.output.endswith("out.json")
    assert validate_args.input.endswith("out.json")
    assert len(prompts) == 2
