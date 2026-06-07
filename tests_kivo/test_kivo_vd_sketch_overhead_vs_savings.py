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
        repo_root
        / "scripts"
        / "kivo_vd"
        / "compare_sketch_overhead_to_savings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_sketch_overhead_to_savings", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _event_estimate() -> dict:
    return {
        "bytes_per_block": 100,
        "aggregate": {
            "average_selected_blocks": 4,
            "average_skipped_blocks": 6,
            "average_active_kv_bytes": 400,
            "average_skipped_kv_bytes": 1000,
            "average_estimated_reduction_ratio": 0.6,
            "estimated_routing_events": 8,
        },
    }


def _sketch_overhead() -> dict:
    return {
        "model_kv_metadata": {"model": "gpt2", "num_blocks": 16},
        "full_kv_bytes": 1600,
        "buffer_assumption": "one vector per block",
        "rows": [
            {
                "sketch_type": "count_sketch",
                "sketch_dim": 32,
                "theoretical_sketch_bytes": 100,
                "sketch_overhead_ratio_vs_full_kv": 0.0625,
                "measured_allocated_delta_bytes": None,
            },
            {
                "sketch_type": "srht",
                "sketch_dim": 64,
                "theoretical_sketch_bytes": 400,
                "sketch_overhead_ratio_vs_full_kv": 0.25,
                "measured_allocated_delta_bytes": 512,
            },
        ],
    }


def test_event_and_overhead_payloads_are_parsed(tmp_path: Path) -> None:
    module = _load_module()
    event_path = tmp_path / "event.json"
    overhead_path = tmp_path / "overhead.json"
    _write(event_path, _event_estimate())
    _write(overhead_path, _sketch_overhead())

    result = module.build_comparison(
        event_estimate_path=event_path,
        sketch_overhead_path=overhead_path,
    )

    assert result["event_estimate_summary"]["bytes_per_block"] == 100
    assert result["event_estimate_summary"]["routing_event_count"] == 8
    assert len(result["comparison_rows"]) == 2
    assert result["comparison_rows"][1]["experimental_reference"] is True


def test_overhead_and_net_savings_calculation() -> None:
    module = _load_module()
    row = module.compare_overhead_row(
        {
            "sketch_type": "count_sketch",
            "sketch_dim": 32,
            "theoretical_sketch_bytes": 100,
            "sketch_overhead_ratio_vs_full_kv": 0.05,
        },
        average_skipped_kv_bytes=1000,
    )

    assert row["overhead_vs_avg_skipped_kv_ratio"] == pytest.approx(0.1)
    assert row["net_theoretical_savings_bytes"] == pytest.approx(900)
    assert row["net_theoretical_savings_ratio_vs_skipped"] == pytest.approx(0.9)
    assert row["affordability"] == "acceptable"
    assert row["overhead_affordable"] is True


def test_affordability_classification_boundaries() -> None:
    module = _load_module()

    assert module.affordability_classification(0.05) == "excellent"
    assert module.affordability_classification(0.15) == "acceptable"
    assert module.affordability_classification(0.30) == "questionable"
    assert module.affordability_classification(0.31) == "poor"


def test_markdown_contains_required_caveats(tmp_path: Path) -> None:
    module = _load_module()
    event_path = tmp_path / "event.json"
    overhead_path = tmp_path / "overhead.json"
    _write(event_path, _event_estimate())
    _write(overhead_path, _sketch_overhead())
    result = module.build_comparison(
        event_estimate_path=event_path,
        sketch_overhead_path=overhead_path,
    )

    markdown = module.render_markdown(result)

    assert "theoretical only" in markdown.lower()
    assert "overhead only" in markdown.lower()
    assert "No active routing" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "do not replace full KV" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "compare_sketch_overhead_to_savings.py"
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
