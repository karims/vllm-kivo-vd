# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "check_phase8_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase8_readiness", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _artifact_bundle(
    tmp_path: Path,
    *,
    classification: str = "excellent",
    pipeline_success: bool = True,
) -> tuple[Path, Path]:
    overhead_json = tmp_path / "sketch_buffer_overhead.json"
    overhead_md = tmp_path / "sketch_buffer_overhead.md"
    comparison_json = tmp_path / "sketch_overhead_vs_savings.json"
    comparison_md = tmp_path / "sketch_overhead_vs_savings.md"
    accounting_json = (
        tmp_path / "event_aware_sketch_buffer_accounting.json"
    )
    accounting_md = (
        tmp_path / "event_aware_sketch_buffer_accounting.md"
    )
    pipeline = tmp_path / "pipeline_summary.json"
    for path in (overhead_json, comparison_json):
        _write(path, {})
    for path in (overhead_md, comparison_md, accounting_md):
        path.write_text("# artifact\n", encoding="utf-8")
    _write(
        accounting_json,
        {
            "accounting_rows": [
                {
                    "sketch_type": "count_sketch",
                    "sketch_dim": 32,
                    "cumulative_request_model": {
                        "overhead_vs_cumulative_skipped_kv": 0.04,
                        "classification": classification,
                    },
                    "break_even_model": {
                        "break_even_events": 1,
                        "break_even_events_classification": "immediate",
                        "break_even_skipped_blocks": 4,
                    },
                }
            ],
            "caveats": {
                "theoretical_only": True,
                "measured_runtime_reduction": False,
                "active_routing": False,
                "full_kv_still_allocated": True,
            },
        },
    )
    _write(
        pipeline,
        {
            "success": pipeline_success,
            "dry_run": False,
            "stages": [
                {"status": "succeeded"},
                {"status": "succeeded"},
                {"status": "succeeded"},
            ],
            "savings_are_theoretical_only": True,
            "measured_runtime_reduction": False,
            "active_routing": False,
            "output_files": {
                "sketch_overhead_json": str(overhead_json),
                "sketch_overhead_markdown": str(overhead_md),
                "overhead_vs_savings_json": str(comparison_json),
                "overhead_vs_savings_markdown": str(comparison_md),
                "event_accounting_json": str(accounting_json),
                "event_accounting_markdown": str(accounting_md),
                "pipeline_summary": str(pipeline),
            },
        },
    )
    return pipeline, accounting_json


def test_missing_artifacts_produce_warnings(tmp_path: Path) -> None:
    module = _load_module()

    report = module.build_readiness_report(
        pipeline_summary_path=tmp_path / "missing-pipeline.json",
        event_aware_accounting_path=tmp_path / "missing-accounting.json",
    )

    assert report["phase9_ready"] is False
    assert report["warnings"]
    assert not all(report["artifacts_present"].values())


def test_failed_pipeline_blocks_phase9(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, accounting = _artifact_bundle(
        tmp_path, pipeline_success=False
    )

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        event_aware_accounting_path=accounting,
    )

    assert report["phase9_ready"] is False
    assert report["checks"]["pipeline_success"] is False


def test_acceptable_compressed_config_allows_limited_phase9(
    tmp_path: Path,
) -> None:
    module = _load_module()
    pipeline, accounting = _artifact_bundle(
        tmp_path, classification="acceptable"
    )

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        event_aware_accounting_path=accounting,
    )

    assert report["phase9_ready"] is True
    assert report["best_recommended_config"]["sketch_dim"] == 32
    assert report["best_cumulative_overhead_classification"] == (
        "acceptable"
    )
    assert "temporary" in report["phase9_scope"]


def test_weak_configs_block_phase9(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, accounting = _artifact_bundle(
        tmp_path, classification="questionable"
    )

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        event_aware_accounting_path=accounting,
    )

    assert report["phase9_ready"] is False
    assert report["eligible_configs"] == []


def test_active_routing_remains_false(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, accounting = _artifact_bundle(tmp_path)

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        event_aware_accounting_path=accounting,
    )

    assert report["phase9_ready"] is True
    assert report["active_routing"] is False
    assert report["measured_runtime_reduction"] is False
    assert report["full_kv_still_allocated"] is True


def test_markdown_includes_caveats(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, accounting = _artifact_bundle(tmp_path)
    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        event_aware_accounting_path=accounting,
    )

    markdown = module.render_markdown(report)

    assert "theoretical only" in markdown
    assert "Full KV is still allocated" in markdown
    assert "Active routing remains false" in markdown
    assert "Measured runtime KV memory reduction remains false" in markdown


def test_readiness_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "check_phase8_readiness.py"

    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--pipeline-summary",
        "--event-aware-accounting",
        "--overhead-vs-savings",
        "--sketch-overhead",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
