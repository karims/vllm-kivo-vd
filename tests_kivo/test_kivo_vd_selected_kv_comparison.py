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
        / "compare_selected_kv_materialization.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_selected_kv_materialization", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _materialization(preview_only: bool = False) -> dict:
    return {
        "num_events_processed": 2,
        "device": {
            "resolved": "cuda:0",
            "cuda_available": True,
        },
        "aggregate": {
            "average_selected_blocks": 2,
            "average_selected_kv_bytes": 200,
            "total_selected_kv_bytes_materialized": 400,
            "average_copy_time_ms": 2,
            "p50_copy_time_ms": 1.5,
            "p90_copy_time_ms": 2.5,
            "max_copy_time_ms": 2.5,
            "average_materialization_ratio": 0.25,
        },
        "per_event_rows": [
            {
                "full_considered_kv_bytes": 800,
                "selected_ids_preview_only": preview_only,
                "cuda_allocated_delta_bytes": 200,
                "cuda_reserved_delta_bytes": 256,
            },
            {
                "full_considered_kv_bytes": 800,
                "selected_ids_preview_only": preview_only,
                "cuda_allocated_delta_bytes": 200,
                "cuda_reserved_delta_bytes": 256,
            },
        ],
        "per_event_rows_truncated": False,
        "caveats": {"synthetic_kv": True},
    }


def _event_estimate() -> dict:
    return {
        "bytes_per_block": 100,
        "aggregate": {
            "average_selected_blocks": 2,
            "average_skipped_blocks": 6,
            "average_active_kv_bytes": 200,
            "average_skipped_kv_bytes": 600,
            "average_estimated_reduction_ratio": 0.75,
            "estimated_routing_events": 2,
        },
        "per_event_estimates": [
            {"skipped_kv_bytes": 600},
            {"skipped_kv_bytes": 600},
        ],
    }


def _sketch_accounting() -> dict:
    return {
        "recommendations": {
            "preferred_configs": [{
                "sketch_type": "count_sketch",
                "sketch_dim": 16,
            }]
        },
        "accounting_rows": [{
            "sketch_type": "count_sketch",
            "sketch_dim": 16,
            "global_pool_model": {"sketch_pool_bytes": 100},
            "cumulative_request_model": {
                "overhead_vs_cumulative_skipped_kv": 0.08,
                "classification": "acceptable",
            },
            "break_even_model": {
                "break_even_events": 1,
                "break_even_skipped_blocks": 1,
            },
        }],
    }


def test_parses_materialization_and_event_estimate(
    tmp_path: Path,
) -> None:
    module = _load_module()
    materialization_path = tmp_path / "materialization.json"
    event_path = tmp_path / "event.json"
    _write(materialization_path, _materialization())
    _write(event_path, _event_estimate())

    result = module.build_comparison(
        materialization_path=materialization_path,
        event_estimate_path=event_path,
    )

    assert result["materialization_summary"][
        "num_events_processed"
    ] == 2
    assert result["event_estimate_summary"]["bytes_per_block"] == 100
    assert result["event_estimate_summary"][
        "cumulative_skipped_kv_bytes"
    ] == 1200


def test_selected_ratios_and_copy_throughput(tmp_path: Path) -> None:
    module = _load_module()
    materialization_path = tmp_path / "materialization.json"
    event_path = tmp_path / "event.json"
    _write(materialization_path, _materialization())
    _write(event_path, _event_estimate())

    result = module.build_comparison(
        materialization_path=materialization_path,
        event_estimate_path=event_path,
    )
    metrics = result["comparison_metrics"]

    assert metrics["selected_vs_full_considered_ratio"] == 0.25
    assert metrics["selected_vs_skipped_ratio"] == pytest.approx(1 / 3)
    assert metrics[
        "cumulative_selected_vs_cumulative_skipped_ratio"
    ] == pytest.approx(1 / 3)
    assert metrics["rough_copy_throughput_bytes_per_second"] == 100_000


def test_optional_sketch_accounting_is_integrated(
    tmp_path: Path,
) -> None:
    module = _load_module()
    materialization_path = tmp_path / "materialization.json"
    event_path = tmp_path / "event.json"
    sketch_path = tmp_path / "sketch.json"
    _write(materialization_path, _materialization())
    _write(event_path, _event_estimate())
    _write(sketch_path, _sketch_accounting())

    result = module.build_comparison(
        materialization_path=materialization_path,
        event_estimate_path=event_path,
        sketch_accounting_path=sketch_path,
    )
    row = result["comparison_metrics"][
        "selected_materialization_plus_sketch_overhead"
    ][0]

    assert row["sketch_type"] == "count_sketch"
    assert row["average_selected_plus_sketch_bytes"] == 300
    assert row["selected_plus_sketch_vs_average_skipped_ratio"] == 0.5


def test_missing_optional_sketch_accounting_does_not_fail(
    tmp_path: Path,
) -> None:
    module = _load_module()
    materialization_path = tmp_path / "materialization.json"
    event_path = tmp_path / "event.json"
    _write(materialization_path, _materialization())
    _write(event_path, _event_estimate())

    result = module.build_comparison(
        materialization_path=materialization_path,
        event_estimate_path=event_path,
    )

    assert result["sketch_accounting_summary"] is None
    assert result["comparison_metrics"][
        "selected_materialization_plus_sketch_overhead"
    ] == []


def test_preview_only_rows_block_repeated_run_recommendation(
    tmp_path: Path,
) -> None:
    module = _load_module()
    materialization_path = tmp_path / "materialization.json"
    event_path = tmp_path / "event.json"
    _write(materialization_path, _materialization(preview_only=True))
    _write(event_path, _event_estimate())

    result = module.build_comparison(
        materialization_path=materialization_path,
        event_estimate_path=event_path,
    )

    assert result["recommendations"][
        "phase9_2_repeated_run_recommended"
    ] is False
    assert result["recommendations"][
        "preview_only_limits_conclusion"
    ] is True
    assert result["warnings"]


def test_markdown_contains_required_caveats(tmp_path: Path) -> None:
    module = _load_module()
    materialization_path = tmp_path / "materialization.json"
    event_path = tmp_path / "event.json"
    _write(materialization_path, _materialization())
    _write(event_path, _event_estimate())
    result = module.build_comparison(
        materialization_path=materialization_path,
        event_estimate_path=event_path,
    )

    markdown = module.render_markdown(result)

    assert "KV tensors are synthetic" in markdown
    assert "outside the attention path" in markdown
    assert "Full KV is still allocated" in markdown
    assert "No active routing" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "Quality is not measured" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "compare_selected_kv_materialization.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--materialization",
        "--event-estimate",
        "--sketch-accounting",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
