# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "check_phase9_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase9_readiness", module_path
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
    pipeline_success: bool = True,
    selected_blocks: float = 4,
    ratio: float = 0.40,
) -> tuple[Path, Path, Path]:
    materialization = tmp_path / "selected_kv_materialization.json"
    materialization_md = tmp_path / "selected_kv_materialization.md"
    comparison = (
        tmp_path / "selected_kv_materialization_comparison.json"
    )
    comparison_md = (
        tmp_path / "selected_kv_materialization_comparison.md"
    )
    event_estimate = tmp_path / "event_estimate.json"
    sketch_accounting = tmp_path / "sketch_accounting.json"
    pipeline = tmp_path / "pipeline_summary.json"

    materialization_md.write_text("# materialization\n", encoding="utf-8")
    comparison_md.write_text("# comparison\n", encoding="utf-8")
    _write(event_estimate, {"aggregate": {}})
    _write(sketch_accounting, {"accounting_rows": []})
    _write(
        materialization,
        {
            "num_events_processed": 2,
            "aggregate": {
                "average_selected_blocks": selected_blocks,
                "average_copy_time_ms": 0.25,
                "average_materialization_ratio": ratio,
            },
            "per_event_rows": [
                {"selected_ids_preview_only": False},
                {"selected_ids_preview_only": False},
            ],
            "warnings": [],
            "caveats": {
                "synthetic_kv": True,
                "outside_attention_path": True,
                "full_kv_still_allocated": True,
                "active_routing": False,
                "measured_runtime_reduction": False,
            },
        },
    )
    _write(
        comparison,
        {
            "caveats": {
                "synthetic_kv": True,
                "outside_attention_path": True,
                "full_kv_still_allocated": True,
                "active_routing": False,
                "measured_runtime_reduction": False,
                "quality_not_measured": True,
            }
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
            ],
            "synthetic_kv": True,
            "outside_attention_path": True,
            "full_kv_still_allocated": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "quality_not_measured": True,
            "parameters": {
                "event_estimate": str(event_estimate),
                "sketch_accounting": str(sketch_accounting),
            },
            "output_files": {
                "materialization_json": str(materialization),
                "materialization_markdown": str(materialization_md),
                "comparison_json": str(comparison),
                "comparison_markdown": str(comparison_md),
                "pipeline_summary": str(pipeline),
            },
        },
    )
    return pipeline, materialization, comparison


def test_missing_artifacts_produce_warnings(tmp_path: Path) -> None:
    module = _load_module()

    report = module.build_readiness_report(
        pipeline_summary_path=tmp_path / "missing-pipeline.json",
        materialization_path=tmp_path / "missing-materialization.json",
        comparison_path=tmp_path / "missing-comparison.json",
    )

    assert report["phase10_ready"] is False
    assert report["warnings"]
    assert not all(report["artifacts_present"].values())


def test_failed_pipeline_blocks_phase10(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, materialization, comparison = _artifact_bundle(
        tmp_path, pipeline_success=False
    )

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        materialization_path=materialization,
        comparison_path=comparison,
    )

    assert report["phase10_ready"] is False
    assert report["checks"]["pipeline_success"] is False


def test_successful_pipeline_allows_limited_phase10(
    tmp_path: Path,
) -> None:
    module = _load_module()
    pipeline, materialization, comparison = _artifact_bundle(tmp_path)

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        materialization_path=materialization,
        comparison_path=comparison,
    )

    assert report["phase10_ready"] is True
    assert report["materialization_ratio_classification"] == "promising"
    assert "standalone" in report["allowed_scope"]
    assert "outside vLLM" in report["allowed_scope"]


def test_zero_selected_blocks_blocks_phase10(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, materialization, comparison = _artifact_bundle(
        tmp_path, selected_blocks=0
    )

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        materialization_path=materialization,
        comparison_path=comparison,
    )

    assert report["phase10_ready"] is False
    assert report["checks"]["selected_blocks_nonzero"] is False


def test_safety_claims_remain_false_or_unmeasured(
    tmp_path: Path,
) -> None:
    module = _load_module()
    pipeline, materialization, comparison = _artifact_bundle(tmp_path)

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        materialization_path=materialization,
        comparison_path=comparison,
    )

    assert report["active_routing"] is False
    assert report["measured_runtime_reduction"] is False
    assert report["quality_not_measured"] is True
    assert report["full_kv_still_allocated"] is True


def test_ratio_classification_boundaries() -> None:
    module = _load_module()

    assert "strong" in module.classify_materialization_ratio(0.24)
    assert module.classify_materialization_ratio(0.25) == "promising"
    assert module.classify_materialization_ratio(0.50) == "moderate_signal"
    assert module.classify_materialization_ratio(0.80) == "weak_signal"


def test_markdown_includes_caveats(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, materialization, comparison = _artifact_bundle(tmp_path)
    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        materialization_path=materialization,
        comparison_path=comparison,
    )

    markdown = module.render_markdown(report)

    assert "KV tensors are synthetic" in markdown
    assert "outside the attention path" in markdown
    assert "Full KV is still allocated" in markdown
    assert "Active routing remains false" in markdown
    assert "Measured runtime memory reduction remains false" in markdown
    assert "Quality is not measured" in markdown


def test_readiness_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "check_phase9_readiness.py"
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--pipeline-summary",
        "--materialization",
        "--comparison",
        "--event-estimate",
        "--sketch-accounting",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
