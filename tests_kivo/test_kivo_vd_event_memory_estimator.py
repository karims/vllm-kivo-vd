# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "estimate_kivo_memory_from_events.py"
    )
    spec = importlib.util.spec_from_file_location(
        "estimate_kivo_memory_from_events", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_bytes_per_block_formula() -> None:
    module = _load_module()

    result = module.bytes_per_kv_block(
        num_layers=12,
        num_kv_heads=12,
        head_dim=64,
        block_size=16,
        dtype_bytes=2,
    )

    assert result == 589_824


def test_estimator_parses_events_and_aggregates(tmp_path: Path) -> None:
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "event_type": "after_allocate_slots",
                "request_id": "r1",
            },
            {
                "event_type": "dry_run_routing_decision",
                "event_id": 2,
                "request_id": "r1",
                "source": "waiting",
                "selected_block_count": 6,
                "recent_block_count": 2,
                "skipped_block_count": 4,
                "candidate_budget_blocks": 8,
                "recent_window_blocks": 4,
            },
            {
                "name": "dry_run_routing_decision",
                "request_id": "r1",
                "source": "running",
                "selected_block_count": 8,
                "recent_block_count": 4,
                "skipped_block_count": 2,
            },
        ],
    )

    result = module.estimate_memory(
        events_path=events_path,
        memory_baseline_path=None,
        model="gpt2",
        num_layers=12,
        num_kv_heads=12,
        head_dim=64,
        block_size=16,
        dtype_bytes=2,
    )

    block_bytes = 589_824
    first = result["per_event_estimates"][0]
    assert first["active_blocks"] == 6
    assert first["total_considered_blocks"] == 10
    assert first["full_considered_kv_bytes"] == 10 * block_bytes
    assert first["active_kv_bytes"] == 6 * block_bytes
    assert first["skipped_kv_bytes"] == 4 * block_bytes
    assert first["estimated_reduction_ratio"] == pytest.approx(0.4)

    aggregate = result["aggregate"]
    assert aggregate["total_routing_events"] == 2
    assert aggregate["estimated_routing_events"] == 2
    assert aggregate["average_selected_blocks"] == 7
    assert aggregate["average_recent_blocks"] == 3
    assert aggregate["average_skipped_blocks"] == 3
    assert aggregate["average_estimated_reduction_ratio"] == pytest.approx(0.3)
    assert aggregate["request_ids_seen"] == ["r1"]
    assert aggregate["sources_seen"] == ["running", "waiting"]
    assert result["estimated_only"] is True
    assert result["measured_runtime_reduction"] is False


def test_estimator_can_infer_metadata_from_baseline(tmp_path: Path) -> None:
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    baseline_path = tmp_path / "baseline.json"
    _write_jsonl(
        events_path,
        [{
            "event_type": "dry_run_routing_decision",
            "selected_block_count": 1,
            "skipped_block_count": 1,
        }],
    )
    baseline_path.write_text(
        json.dumps({
            "config": {
                "model": "test-model",
                "model_config": {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 4,
                    "head_dim": 8,
                },
            }
        }),
        encoding="utf-8",
    )

    result = module.estimate_memory(
        events_path=events_path,
        memory_baseline_path=baseline_path,
        model="",
        num_layers=None,
        num_kv_heads=None,
        head_dim=None,
        block_size=16,
        dtype_bytes=2,
    )

    assert result["model_kv_metadata"]["model"] == "test-model"
    assert result["bytes_per_block"] == 2 * 2 * 4 * 8 * 16 * 2


def test_missing_metadata_has_clear_error(tmp_path: Path) -> None:
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(events_path, [])

    with pytest.raises(ValueError, match="missing required KV metadata"):
        module.estimate_memory(
            events_path=events_path,
            memory_baseline_path=None,
            model="gpt2",
            num_layers=None,
            num_kv_heads=None,
            head_dim=None,
            block_size=16,
            dtype_bytes=2,
        )


def test_markdown_contains_estimated_only_caveat(tmp_path: Path) -> None:
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [{
            "event_type": "dry_run_routing_decision",
            "selected_block_count": 1,
            "recent_block_count": 1,
            "skipped_block_count": 3,
        }],
    )
    result = module.estimate_memory(
        events_path=events_path,
        memory_baseline_path=None,
        model="gpt2",
        num_layers=12,
        num_kv_heads=12,
        head_dim=64,
        block_size=16,
        dtype_bytes=2,
    )

    markdown = module.render_markdown(result)

    assert "Estimated-only" in markdown
    assert "not measured runtime memory reduction" in markdown
    assert "Proven Vs Not Proven" in markdown


def test_estimator_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "kivo_vd" / "estimate_kivo_memory_from_events.py"
    )

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--events",
        "--memory-baseline",
        "--num-layers",
        "--num-kv-heads",
        "--head-dim",
        "--block-size",
        "--dtype-bytes",
        "--output-json",
        "--output-md",
    ):
        assert flag in proc.stdout
