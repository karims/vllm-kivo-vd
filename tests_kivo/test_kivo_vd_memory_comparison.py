# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "compare_memory_baseline_and_estimate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_memory_baseline_and_estimate", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory_payload(
    *,
    kivo_enabled: bool,
    init_allocated: int = 100,
    generation_allocated: int = 20,
    peak_allocated: int = 150,
    peak_reserved: int = 200,
) -> dict:
    before_init = 10
    after_init = before_init + init_allocated
    before_generate = after_init
    after_generate = before_generate + generation_allocated
    names_and_values = [
        ("process_start", 0, 0, 0, 0),
        ("before_llm_init", before_init, 20, 10, 20),
        ("after_llm_init", after_init, 140, peak_allocated, peak_reserved),
        (
            "before_generate",
            before_generate,
            140,
            peak_allocated,
            peak_reserved,
        ),
        (
            "after_generate",
            after_generate,
            160,
            peak_allocated,
            peak_reserved,
        ),
        (
            "after_request_or_cleanup",
            5,
            30,
            peak_allocated,
            peak_reserved,
        ),
    ]
    checkpoints = [
        {
            "name": name,
            "timestamp": float(index),
            "memory_allocated_bytes": allocated,
            "memory_reserved_bytes": reserved,
            "max_memory_allocated_bytes": max_allocated,
            "max_memory_reserved_bytes": max_reserved,
            "free_memory_bytes": 1000,
            "total_memory_bytes": 2000,
        }
        for index, (
            name,
            allocated,
            reserved,
            max_allocated,
            max_reserved,
        ) in enumerate(names_and_values)
    ]
    return {
        "config": {"model": "gpt2"},
        "kivo_enabled": kivo_enabled,
        "memory_checkpoints": checkpoints,
    }


def _estimate_payload() -> dict:
    return {
        "bytes_per_block": 589_824,
        "aggregate": {
            "total_routing_events": 2,
            "estimated_routing_events": 2,
            "average_selected_blocks": 6.0,
            "average_skipped_blocks": 4.0,
            "average_active_kv_bytes": 3_538_944.0,
            "average_skipped_kv_bytes": 2_359_296.0,
            "average_estimated_reduction_ratio": 0.4,
        },
        "estimated_only": True,
        "measured_runtime_reduction": False,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_measured_memory_summary_calculates_deltas() -> None:
    module = _load_module()

    summary = module.summarize_measured_memory(
        _memory_payload(kivo_enabled=False)
    )

    assert summary["model_init_allocated_delta_bytes"] == 100
    assert summary["generation_allocated_delta_bytes"] == 20
    assert summary["peak_allocated_bytes"] == 150
    assert summary["peak_reserved_bytes"] == 200
    assert summary["cleanup_allocated_bytes"] == 5


def test_event_estimate_summary_parses_tiny_payload() -> None:
    module = _load_module()

    summary = module.summarize_event_estimate(_estimate_payload())

    assert summary["bytes_per_block"] == 589_824
    assert summary["average_selected_blocks"] == 6
    assert summary["average_skipped_blocks"] == 4
    assert summary["average_estimated_reduction_ratio"] == 0.4
    assert summary["source_estimated_only"] is True


def test_optional_kivo_comparison_reports_observed_drop(
    tmp_path: Path,
) -> None:
    module = _load_module()
    baseline_path = tmp_path / "baseline.json"
    kivo_path = tmp_path / "kivo.json"
    estimate_path = tmp_path / "estimate.json"
    _write_json(
        baseline_path,
        _memory_payload(kivo_enabled=False, peak_allocated=200),
    )
    _write_json(
        kivo_path,
        _memory_payload(kivo_enabled=True, peak_allocated=180),
    )
    _write_json(estimate_path, _estimate_payload())

    result = module.build_comparison(
        baseline_path=baseline_path,
        kivo_path=kivo_path,
        event_estimate_path=estimate_path,
    )

    comparison = result["baseline_vs_kivo_measured_comparison"]
    assert comparison["kivo_minus_baseline_peak_allocated_bytes"] == -20
    assert comparison["peak_allocated_drop_observed"] is True
    assert result["conclusion"]["measured_runtime_drop_observed"] is True
    assert result["caveats"]["measured_runtime_reduction"] is False


def test_comparison_works_without_kivo_memory(tmp_path: Path) -> None:
    module = _load_module()
    baseline_path = tmp_path / "baseline.json"
    estimate_path = tmp_path / "estimate.json"
    _write_json(baseline_path, _memory_payload(kivo_enabled=False))
    _write_json(estimate_path, _estimate_payload())

    result = module.build_comparison(
        baseline_path=baseline_path,
        kivo_path=None,
        event_estimate_path=estimate_path,
    )

    assert result["measured_memory_summary"]["kivo_dry_run"] is None
    assert result["baseline_vs_kivo_measured_comparison"] is None
    assert result["conclusion"]["measured_runtime_drop_observed"] is False
    assert (
        result["conclusion"]["theoretical_active_kv_reduction_available"]
        is True
    )


def test_markdown_denies_measured_runtime_reduction(tmp_path: Path) -> None:
    module = _load_module()
    baseline_path = tmp_path / "baseline.json"
    estimate_path = tmp_path / "estimate.json"
    _write_json(baseline_path, _memory_payload(kivo_enabled=False))
    _write_json(estimate_path, _estimate_payload())
    result = module.build_comparison(
        baseline_path=baseline_path,
        kivo_path=None,
        event_estimate_path=estimate_path,
    )

    markdown = module.render_markdown(result)

    assert "measured runtime KV memory reduction" in markdown
    assert "`measured_runtime_reduction` is `false`" in markdown
    assert "theoretical event-based active-KV savings" in markdown


def test_comparison_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "compare_memory_baseline_and_estimate.py"
    )

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--baseline-memory",
        "--kivo-memory",
        "--event-estimate",
        "--output-json",
        "--output-md",
    ):
        assert flag in proc.stdout
