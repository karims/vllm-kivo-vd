from __future__ import annotations

import json
from pathlib import Path

from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (
    _load_exported_counters,
    build_summary,
)
from scripts.kivo_vd.validate_source_s5_19_demotable_transport_probe import (
    validate_summary,
)
from vllm.v1.core.kivo_demotion_counters import (
    export_kivo_demotion_counters_snapshot_if_enabled,
    increment_kivo_demotion_counter,
    reset_kivo_demotion_counters,
)


def test_no_export_file_when_env_disabled(monkeypatch, tmp_path: Path):
    export_path = tmp_path / "counters.json"
    monkeypatch.delenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", raising=False)
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE", str(export_path))
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("worker_envelopes_built")
    export_kivo_demotion_counters_snapshot_if_enabled(source="test")
    assert not export_path.exists()


def test_export_file_written_when_env_enabled(monkeypatch, tmp_path: Path):
    export_path = tmp_path / "counters.json"
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE", str(export_path))
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("worker_envelopes_built")
    export_kivo_demotion_counters_snapshot_if_enabled(source="test")
    assert export_path.exists()


def test_exported_json_contains_pid_and_counters(monkeypatch, tmp_path: Path):
    export_path = tmp_path / "counters.json"
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE", str(export_path))
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("worker_envelopes_built")
    export_kivo_demotion_counters_snapshot_if_enabled(source="test")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("pid"), int)
    assert payload["source"] == "test"
    assert payload["counters"]["worker_envelopes_built"] == 1


def test_atomic_overwrite_works(monkeypatch, tmp_path: Path):
    export_path = tmp_path / "counters.json"
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE", str(export_path))
    reset_kivo_demotion_counters()
    increment_kivo_demotion_counter("worker_envelopes_built")
    export_kivo_demotion_counters_snapshot_if_enabled(source="one")
    increment_kivo_demotion_counter("worker_envelopes_built")
    export_kivo_demotion_counters_snapshot_if_enabled(source="two")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["source"] == "two"
    assert payload["counters"]["worker_envelopes_built"] == 2
    assert not list(tmp_path.glob("counters.json.tmp.*"))


def test_runner_prefers_exported_counters_over_parent_counters():
    summary = build_summary(
        generation_success=True,
        prompt_count=1,
        prompt_char_lengths=[100],
        prompt_token_lengths=[10],
        parent_counters={"worker_envelopes_built": 0},
        exported_counters={"worker_envelopes_built": 3},
        counter_export_file_found=True,
        counter_export_pid=123,
    )
    assert summary["counters"]["worker_envelopes_built"] == 3
    assert summary["exported_counters"]["worker_envelopes_built"] == 3
    assert summary["parent_counters"]["worker_envelopes_built"] == 0


def test_validator_reports_export_file_found():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "counter_export_file_found": True,
            "exported_counters": {"scheduler_envelopes_received": 1},
            "req_to_blocks_removed": 0,
            "free_to_pool_calls": 0,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
        }
    )
    assert result["validation_passed"] is True
    assert result["counter_export_file_found"] is True
    assert result["transport_observed"] is True


def test_validator_passes_generation_success_with_zero_counters_but_warns():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "counter_export_file_found": False,
            "parent_counters": {},
            "req_to_blocks_removed": 0,
            "free_to_pool_calls": 0,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
        }
    )
    assert result["validation_passed"] is True
    assert result["transport_observed"] is False
    assert "no_cross_process_counter_export_observed" in result["warnings"]


def test_validator_fails_if_free_to_pool_calls_positive():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "counter_export_file_found": True,
            "exported_counters": {},
            "req_to_blocks_removed": 0,
            "free_to_pool_calls": 1,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
        }
    )
    assert result["validation_passed"] is False


def test_validator_fails_if_req_to_blocks_removed_positive():
    result = validate_summary(
        {
            "generation_success": True,
            "prompt_count": 1,
            "counter_export_file_found": True,
            "exported_counters": {},
            "req_to_blocks_removed": 1,
            "free_to_pool_calls": 0,
            "memory_claim_allowed": False,
            "free_to_pool_claim_allowed": False,
        }
    )
    assert result["validation_passed"] is False


def test_load_exported_counters_reads_snapshot(tmp_path: Path):
    export_path = tmp_path / "counters.json"
    export_path.write_text(
        json.dumps({"pid": 321, "counters": {"core_commands_attempted": 2}}),
        encoding="utf-8",
    )
    counters, found, pid = _load_exported_counters(str(export_path))
    assert found is True
    assert pid == 321
    assert counters == {"core_commands_attempted": 2}
