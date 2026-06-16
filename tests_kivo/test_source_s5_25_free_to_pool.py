from __future__ import annotations

from pathlib import Path

from scripts.kivo_vd.run_source_s5_25_free_to_pool_probe import (
    parse_args as parse_run_args,
)
from scripts.kivo_vd.validate_source_s5_25_free_to_pool_probe import (
    parse_args as parse_validate_args,
    validate_summary,
)
from tests_kivo.test_source_s5_23_ownership_remove import (
    _command,
    _core_config,
    _manager,
)
from vllm.v1.core.kivo_demotion_command import KivoFreeToPoolConfig
from vllm.v1.core.kivo_demotion_counters import (
    get_kivo_demotion_counters_snapshot,
    reset_kivo_demotion_counters,
)


def _free_config(
    *, enabled: bool = True, action: str = "free_removed_demoted_only"
) -> KivoFreeToPoolConfig:
    return KivoFreeToPoolConfig(enabled=enabled, action=action)


def _summary_base() -> dict[str, object]:
    return {
        "generation_success": True,
        "prompt_count": 1,
        "transport_observed": True,
        "counter_export_file_found": True,
        "memory_claim_allowed": False,
        "quality_claim_allowed": False,
        "performance_claim_allowed": False,
        "free_to_pool_claim_allowed": False,
        "ownership_removal_claim_allowed": False,
        "live_kv_free_enabled": True,
        "ownership_remove_succeeded": 1,
        "req_to_blocks_removed": 2,
        "ownership_remaining_blocks_last": 1,
        "ownership_remove_invariant_failed": 0,
        "demoted_blocks_marked": 2,
        "ownership_remove_attempted": 1,
        "counters": {},
    }


def test_free_disabled_does_nothing(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    manager = _manager()
    manager.kivo_req_to_removed_demoted_blocks["req0"] = [
        manager.req_to_blocks["req0"][0]
    ]
    result = manager.free_kivo_removed_demoted_blocks_to_pool_if_safe(
        "req0",
        config=_free_config(enabled=False),
    )
    assert result.succeeded is False
    assert result.block_pool_free_called is False


def test_free_enabled_but_no_removed_blocks_fails_closed():
    manager = _manager()
    result = manager.free_kivo_removed_demoted_blocks_to_pool_if_safe(
        "req0",
        config=_free_config(),
    )
    assert result.succeeded is False
    assert result.blocker_reasons["no_removed_demoted_blocks"] == 1


def test_free_enabled_frees_only_removed_blocks(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_REMOVE_ENABLE", "1")
    monkeypatch.setenv(
        "KIVO_KV_OWNERSHIP_REMOVE_ACTION", "remove_marked_demoted_only"
    )
    monkeypatch.setenv("KIVO_KV_FREE_TO_POOL_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_FREE_TO_POOL_ACTION", "free_removed_demoted_only")
    reset_kivo_demotion_counters()
    manager = _manager()

    result = manager.apply_kivo_demotion_command(_command(), config=_core_config())
    snapshot = get_kivo_demotion_counters_snapshot()

    assert result.accepted is True
    assert manager.block_pool.freed_calls == [[10, 11]]
    assert snapshot["free_to_pool_attempted"] == 1
    assert snapshot["free_to_pool_succeeded"] == 1
    assert snapshot["free_to_pool_calls"] == 1
    assert snapshot["free_to_pool_blocks"] == 2


def test_block_still_owned_is_never_freed():
    manager = _manager()
    manager.kivo_req_to_removed_demoted_blocks["req0"] = [manager.req_to_blocks["req0"][0]]
    result = manager.free_kivo_removed_demoted_blocks_to_pool_if_safe(
        "req0",
        config=_free_config(),
    )
    assert result.succeeded is False
    assert result.blocker_reasons["removed_block_still_owned_by_request"] == 1


def test_block_owned_by_other_request_is_never_freed():
    manager = _manager()
    removed_block = manager.req_to_blocks["req0"][0]
    manager.req_to_blocks["req0"] = manager.req_to_blocks["req0"][1:]
    manager.req_to_blocks["req1"] = [removed_block]
    manager.kivo_req_to_removed_demoted_blocks["req0"] = [removed_block]
    result = manager.free_kivo_removed_demoted_blocks_to_pool_if_safe(
        "req0",
        config=_free_config(),
    )
    assert result.succeeded is False
    assert result.blocker_reasons["removed_block_still_owned_by_other_request"] == 1


def test_double_free_is_prevented(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    manager = _manager()
    removed_block = manager.req_to_blocks["req0"][0]
    manager.req_to_blocks["req0"] = manager.req_to_blocks["req0"][1:]
    manager.kivo_req_to_removed_demoted_blocks["req0"] = [removed_block]
    first = manager.free_kivo_removed_demoted_blocks_to_pool_if_safe(
        "req0",
        config=_free_config(),
    )
    manager.kivo_req_to_removed_demoted_blocks["req0"] = [removed_block]
    second = manager.free_kivo_removed_demoted_blocks_to_pool_if_safe(
        "req0",
        config=_free_config(),
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert first.succeeded is True
    assert second.succeeded is False
    assert second.double_free_prevented is True
    assert snapshot["free_to_pool_double_free_prevented"] == 1


def test_validator_accepts_success_case():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 1,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [10, 11],
            "last_remaining_block_ids_sample": [12],
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True


def test_validator_accepts_fail_closed_case():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 0,
            "free_to_pool_rejected": 1,
            "free_to_pool_blocks": 0,
            "free_to_pool_calls": 0,
            "free_to_pool_double_free_prevented": 0,
            "last_freed_block_ids_sample": [],
            "last_remaining_block_ids_sample": [12],
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is True


def test_validator_fails_if_call_without_success():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 0,
            "free_to_pool_rejected": 0,
            "free_to_pool_blocks": 2,
            "free_to_pool_calls": 1,
            "free_to_pool_double_free_prevented": 0,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_validator_fails_if_double_free_detected():
    summary = _summary_base()
    summary.update(
        {
            "free_to_pool_attempted": 1,
            "free_to_pool_succeeded": 0,
            "free_to_pool_rejected": 1,
            "free_to_pool_blocks": 0,
            "free_to_pool_calls": 0,
            "free_to_pool_double_free_prevented": 1,
        }
    )
    result = validate_summary(summary)
    assert result["validation_passed"] is False


def test_cli_help_includes_expected_args():
    run_args = parse_run_args(["--output", str(Path("/tmp/out.json"))])
    validate_args = parse_validate_args(["--input", str(Path("/tmp/out.json"))])
    assert run_args.output.endswith("out.json")
    assert validate_args.input.endswith("out.json")
