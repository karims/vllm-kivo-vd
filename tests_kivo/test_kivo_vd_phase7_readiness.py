# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "check_phase7_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase7_readiness", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _artifact_bundle(tmp_path: Path, ratio: float = 0.30) -> tuple[Path, Path, Path]:
    baseline = tmp_path / "baseline_memory.json"
    kivo = tmp_path / "kivo_dry_run_memory.json"
    events = tmp_path / "kivo_dry_run_events.jsonl"
    estimate = tmp_path / "kivo_event_memory_estimate.json"
    estimate_md = tmp_path / "kivo_event_memory_estimate.md"
    comparison = tmp_path / "memory_comparison.json"
    comparison_md = tmp_path / "memory_comparison.md"
    pipeline = tmp_path / "pipeline_summary.json"

    _write(baseline, {"output_text": "same"})
    _write(kivo, {"output_text": "same", "num_events_exported": 4})
    events.write_text('{"event_type":"dry_run_routing_decision"}\n')
    _write(
        estimate,
        {
            "model_kv_metadata": {
                "model": "gpt2",
                "num_layers": 12,
                "num_kv_heads": 12,
                "head_dim": 64,
                "block_size": 16,
                "dtype_bytes": 2,
            },
            "aggregate": {
                "average_selected_blocks": 6,
                "average_skipped_blocks": 4,
                "average_estimated_reduction_ratio": ratio,
            },
            "warnings": [],
        },
    )
    estimate_md.write_text("# estimate\n")
    _write(
        comparison,
        {
            "conclusion": {"measured_runtime_drop_observed": True},
            "caveats": {"measured_runtime_reduction": False},
        },
    )
    comparison_md.write_text("# comparison\n")
    _write(
        pipeline,
        {
            "success": True,
            "dry_run": False,
            "stages": [{"status": "succeeded"} for _ in range(4)],
            "output_files": {
                "baseline_memory": str(baseline),
                "kivo_dry_run_memory": str(kivo),
                "kivo_events": str(events),
                "event_estimate_json": str(estimate),
                "event_estimate_markdown": str(estimate_md),
                "comparison_json": str(comparison),
                "comparison_markdown": str(comparison_md),
                "pipeline_summary": str(pipeline),
            },
        },
    )
    return pipeline, comparison, estimate


def test_missing_artifacts_produce_warnings(tmp_path: Path) -> None:
    module = _load_module()

    report = module.build_readiness_report(
        pipeline_summary_path=tmp_path / "missing-pipeline.json",
        memory_comparison_path=tmp_path / "missing-comparison.json",
        event_estimate_path=None,
    )

    assert report["phase8_ready"] is False
    assert report["warnings"]
    assert not all(report["artifacts_present"].values())


def test_theoretical_reduction_threshold_classification() -> None:
    module = _load_module()

    assert "below_10" in module.classify_theoretical_reduction(0.09)
    assert "10_to_25" in module.classify_theoretical_reduction(0.10)
    assert "25_to_40" in module.classify_theoretical_reduction(0.25)
    assert "above_40" in module.classify_theoretical_reduction(0.40)


def test_ready_bundle_preserves_measured_reduction_false(
    tmp_path: Path,
) -> None:
    module = _load_module()
    pipeline, comparison, estimate = _artifact_bundle(tmp_path)

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        memory_comparison_path=comparison,
        event_estimate_path=estimate,
    )

    assert report["phase8_ready"] is True
    assert report["measured_runtime_drop_observed"] is True
    assert report["measured_runtime_reduction"] is False
    assert report["theoretical_threshold_for_phase8_met"] is True
    assert "overhead measurement" in report["recommended_next_step"]


def test_weak_reduction_blocks_phase8(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, comparison, estimate = _artifact_bundle(tmp_path, ratio=0.20)

    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        memory_comparison_path=comparison,
        event_estimate_path=estimate,
    )

    assert report["phase8_ready"] is False
    assert report["theoretical_threshold_for_phase8_met"] is False


def test_markdown_includes_caveats(tmp_path: Path) -> None:
    module = _load_module()
    pipeline, comparison, estimate = _artifact_bundle(tmp_path)
    report = module.build_readiness_report(
        pipeline_summary_path=pipeline,
        memory_comparison_path=comparison,
        event_estimate_path=estimate,
    )

    markdown = module.render_markdown(report)

    assert "not demonstrated measured runtime KV memory reduction" in markdown
    assert "active routing" in markdown
    assert "research heuristics" in markdown


def test_readiness_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "check_phase7_readiness.py"

    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--pipeline-summary",
        "--memory-comparison",
        "--event-estimate",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
