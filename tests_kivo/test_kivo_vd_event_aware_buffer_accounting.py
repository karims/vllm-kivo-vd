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
        repo_root / "scripts" / "kivo_vd" / "model_sketch_buffer_accounting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "model_sketch_buffer_accounting", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _overhead() -> dict:
    return {
        "full_kv_bytes": 10_000,
        "rows": [
            {
                "sketch_type": "count_sketch",
                "sketch_dim": 16,
                "theoretical_sketch_bytes": 100,
                "sketch_overhead_ratio_vs_full_kv": 0.01,
            }
        ],
    }


def test_cumulative_skipped_uses_complete_per_event_rows() -> None:
    module = _load_module()
    warnings = []
    estimate = {
        "aggregate": {"estimated_routing_events": 2},
        "per_event_estimates": [
            {"skipped_kv_bytes": 300},
            {"skipped_kv_bytes": 500},
        ],
    }

    cumulative, source = module.calculate_cumulative_skipped_kv(
        estimate, warnings
    )

    assert cumulative == 800
    assert source == "per_event_sum"
    assert not warnings


def test_cumulative_skipped_falls_back_to_average_times_count() -> None:
    module = _load_module()
    warnings = []
    estimate = {
        "aggregate": {
            "estimated_routing_events": 4,
            "average_skipped_kv_bytes": 250,
        },
        "per_event_estimates": [{"skipped_kv_bytes": 250}],
    }

    cumulative, source = module.calculate_cumulative_skipped_kv(
        estimate, warnings
    )

    assert cumulative == 1000
    assert source == "average_times_event_count"
    assert warnings


def test_break_even_and_cumulative_models() -> None:
    module = _load_module()
    event = {
        "average_skipped_kv_bytes": 250,
        "cumulative_skipped_kv_bytes": 1000,
        "bytes_per_block": 60,
    }
    row = module.model_accounting_row(
        {
            "sketch_type": "count_sketch",
            "sketch_dim": 16,
            "theoretical_sketch_bytes": 300,
        },
        event,
        full_kv_pool_bytes=3000,
    )

    assert row["break_even_model"]["break_even_events"] == 2
    assert row["break_even_model"]["break_even_skipped_blocks"] == 5
    assert row["break_even_model"]["break_even_events_classification"] == "fast"
    assert row["cumulative_request_model"][
        "overhead_vs_cumulative_skipped_kv"
    ] == pytest.approx(0.3)
    assert row["cumulative_request_model"]["classification"] == "questionable"
    assert row["cumulative_request_model"][
        "net_cumulative_theoretical_bytes"
    ] == 700


def test_accounting_classification_boundaries() -> None:
    module = _load_module()

    assert module.cumulative_overhead_classification(0.05) == "excellent"
    assert module.cumulative_overhead_classification(0.15) == "acceptable"
    assert module.cumulative_overhead_classification(0.30) == "questionable"
    assert module.cumulative_overhead_classification(0.31) == "poor"
    assert module.break_even_classification(1) == "immediate"
    assert module.break_even_classification(4) == "fast"
    assert module.break_even_classification(16) == "moderate"
    assert module.break_even_classification(17) == "slow"


def test_missing_skipped_data_produces_warning(tmp_path: Path) -> None:
    module = _load_module()
    event_path = tmp_path / "event.json"
    overhead_path = tmp_path / "overhead.json"
    _write(
        event_path,
        {
            "bytes_per_block": 100,
            "aggregate": {
                "estimated_routing_events": 2,
                "average_selected_blocks": 1,
            },
        },
    )
    _write(overhead_path, _overhead())

    report = module.build_report(
        event_estimate_path=event_path,
        sketch_overhead_path=overhead_path,
    )

    assert report["warnings"]
    row = report["accounting_rows"][0]
    assert row["cumulative_request_model"][
        "overhead_vs_cumulative_skipped_kv"
    ] is None
    assert row["break_even_model"]["break_even_events"] is None


def test_markdown_contains_required_caveats(tmp_path: Path) -> None:
    module = _load_module()
    event_path = tmp_path / "event.json"
    overhead_path = tmp_path / "overhead.json"
    _write(
        event_path,
        {
            "bytes_per_block": 100,
            "aggregate": {
                "estimated_routing_events": 2,
                "average_selected_blocks": 2,
                "average_skipped_blocks": 3,
                "average_active_kv_bytes": 200,
                "average_skipped_kv_bytes": 300,
                "average_estimated_reduction_ratio": 0.6,
            },
        },
    )
    _write(overhead_path, _overhead())
    report = module.build_report(
        event_estimate_path=event_path,
        sketch_overhead_path=overhead_path,
    )

    markdown = module.render_markdown(report)

    assert "theoretical only" in markdown.lower()
    assert "Full KV is still allocated" in markdown
    assert "No active routing" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "kivo_vd" / "model_sketch_buffer_accounting.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--event-estimate",
        "--sketch-overhead",
        "--memory-comparison",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
