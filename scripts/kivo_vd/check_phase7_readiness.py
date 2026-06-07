#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Evaluate whether Phase 7 evidence supports a Phase 8.0 experiment."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_KEYS = (
    "baseline_memory",
    "kivo_dry_run_memory",
    "event_estimate_json",
    "event_estimate_markdown",
    "comparison_json",
    "comparison_markdown",
    "pipeline_summary",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Kivo-VD Phase 7 readiness for Phase 8.0."
    )
    parser.add_argument("--pipeline-summary", required=True)
    parser.add_argument("--memory-comparison", required=True)
    parser.add_argument("--event-estimate")
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase7_readiness.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase7_readiness.md",
    )
    return parser.parse_args(argv)


def classify_theoretical_reduction(ratio: float | None) -> str:
    if ratio is None:
        return "unavailable"
    if ratio < 0.10:
        return "below_10_percent_not_compelling"
    if ratio < 0.25:
        return "10_to_25_percent_weak_signal"
    if ratio < 0.40:
        return "25_to_40_percent_promising_signal"
    return "above_40_percent_strong_research_signal"


def _load_optional_json(
    path: str | Path | None,
    label: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    if path is None:
        warnings.append(f"{label} path was not provided")
        return None
    input_path = Path(path)
    if not input_path.exists():
        warnings.append(f"{label} file is missing: {input_path}")
        return None
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"{label} contains invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"{label} must contain a JSON object")
        return None
    return value


def _artifact_paths(
    pipeline: dict[str, Any] | None,
    pipeline_path: str | Path,
    comparison_path: str | Path,
    event_estimate_path: str | Path | None,
) -> dict[str, str | None]:
    output_files = pipeline.get("output_files", {}) if pipeline else {}
    if not isinstance(output_files, dict):
        output_files = {}
    return {
        "baseline_memory": output_files.get("baseline_memory"),
        "kivo_dry_run_memory": output_files.get("kivo_dry_run_memory"),
        "event_estimate_json": (
            str(event_estimate_path)
            if event_estimate_path is not None
            else output_files.get("event_estimate_json")
        ),
        "event_estimate_markdown": output_files.get(
            "event_estimate_markdown"
        ),
        "comparison_json": str(comparison_path),
        "comparison_markdown": output_files.get("comparison_markdown"),
        "pipeline_summary": str(pipeline_path),
        "kivo_events": output_files.get("kivo_events"),
    }


def _present(path: str | None) -> bool:
    return path is not None and Path(path).is_file()


def _outputs_match(
    paths: dict[str, str | None],
    warnings: list[str],
) -> bool:
    baseline = _load_optional_json(
        paths["baseline_memory"],
        "baseline memory",
        warnings,
    )
    kivo = _load_optional_json(
        paths["kivo_dry_run_memory"],
        "Kivo dry-run memory",
        warnings,
    )
    if baseline is None or kivo is None:
        return False
    baseline_text = baseline.get("output_text")
    kivo_text = kivo.get("output_text")
    if not isinstance(baseline_text, str) or not isinstance(kivo_text, str):
        warnings.append("baseline/Kivo output text is unavailable")
        return False
    if baseline_text != kivo_text:
        warnings.append("baseline and Kivo dry-run outputs do not match")
        return False
    return True


def _observer_events_exported(
    paths: dict[str, str | None],
    warnings: list[str],
) -> bool:
    kivo = _load_optional_json(
        paths["kivo_dry_run_memory"],
        "Kivo dry-run memory",
        warnings,
    )
    if kivo is None:
        return False
    count = kivo.get("num_events_exported")
    exported = isinstance(count, int | float) and count > 0
    if not exported:
        warnings.append("Kivo dry-run exported no observer events")
    if not _present(paths["kivo_events"]):
        warnings.append("Kivo observer event JSONL is missing")
        return False
    return exported


def _event_checks(
    estimate: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    if estimate is None:
        return {
            "theoretical_reduction": None,
            "selected_blocks_nonzero": False,
            "skipped_blocks_nonzero": False,
            "metadata_explicit": False,
            "event_warnings_clear": False,
        }
    aggregate = estimate.get("aggregate", {})
    metadata = estimate.get("model_kv_metadata", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(metadata, dict):
        metadata = {}
    estimate_warnings = estimate.get("warnings", [])
    if not isinstance(estimate_warnings, list):
        estimate_warnings = ["event estimate warnings field is malformed"]
    warnings.extend(f"event estimate: {warning}" for warning in estimate_warnings)

    selected = aggregate.get("average_selected_blocks")
    skipped = aggregate.get("average_skipped_blocks")
    ratio = aggregate.get("average_estimated_reduction_ratio")
    selected_nonzero = isinstance(selected, int | float) and selected > 0
    skipped_nonzero = isinstance(skipped, int | float) and skipped > 0
    if not selected_nonzero:
        warnings.append("event estimate selected-block statistics are zero")
    if not skipped_nonzero:
        warnings.append("event estimate skipped-block statistics are zero")

    metadata_explicit = all(
        metadata.get(key) is not None
        for key in (
            "model",
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "block_size",
            "dtype_bytes",
        )
    )
    if not metadata_explicit:
        warnings.append("model/KV metadata is incomplete")
    return {
        "theoretical_reduction": (
            float(ratio) if isinstance(ratio, int | float) else None
        ),
        "selected_blocks_nonzero": selected_nonzero,
        "skipped_blocks_nonzero": skipped_nonzero,
        "metadata_explicit": metadata_explicit,
        "event_warnings_clear": not estimate_warnings,
    }


def build_readiness_report(
    *,
    pipeline_summary_path: str | Path,
    memory_comparison_path: str | Path,
    event_estimate_path: str | Path | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    pipeline = _load_optional_json(
        pipeline_summary_path,
        "pipeline summary",
        warnings,
    )
    comparison = _load_optional_json(
        memory_comparison_path,
        "memory comparison",
        warnings,
    )
    paths = _artifact_paths(
        pipeline,
        pipeline_summary_path,
        memory_comparison_path,
        event_estimate_path,
    )
    artifacts_present = {
        key: _present(paths.get(key)) for key in REQUIRED_OUTPUT_KEYS
    }
    missing = [key for key, present in artifacts_present.items() if not present]
    warnings.extend(f"required artifact is missing: {key}" for key in missing)

    estimate = _load_optional_json(
        paths["event_estimate_json"],
        "event estimate",
        warnings,
    )
    event_checks = _event_checks(estimate, warnings)
    pipeline_success = bool(pipeline and pipeline.get("success"))
    pipeline_dry_run = bool(pipeline and pipeline.get("dry_run"))
    stages = pipeline.get("stages", []) if pipeline else []
    stages_succeeded = bool(stages) and all(
        isinstance(stage, dict) and stage.get("status") == "succeeded"
        for stage in stages
    )
    if not pipeline_success:
        warnings.append("Phase 7 pipeline did not report success")
    if pipeline_dry_run:
        warnings.append("pipeline summary is a dry-run plan, not execution")
    if not stages_succeeded:
        warnings.append("not all Phase 7 pipeline stages succeeded")

    outputs_match = _outputs_match(paths, warnings)
    events_exported = _observer_events_exported(paths, warnings)
    conclusion = comparison.get("conclusion", {}) if comparison else {}
    caveats = comparison.get("caveats", {}) if comparison else {}
    measured_drop_observed = bool(
        isinstance(conclusion, dict)
        and conclusion.get("measured_runtime_drop_observed")
    )
    measured_runtime_reduction = bool(
        isinstance(caveats, dict)
        and caveats.get("measured_runtime_reduction", False)
    )
    ratio = event_checks["theoretical_reduction"]
    classification = classify_theoretical_reduction(ratio)
    threshold_met = ratio is not None and ratio >= 0.25

    phase8_ready = all([
        not missing,
        pipeline_success,
        not pipeline_dry_run,
        stages_succeeded,
        outputs_match,
        events_exported,
        event_checks["selected_blocks_nonzero"],
        event_checks["skipped_blocks_nonzero"],
        event_checks["metadata_explicit"],
        event_checks["event_warnings_clear"],
        threshold_met,
        not measured_runtime_reduction,
    ])
    if phase8_ready:
        next_step = (
            "Proceed only to Phase 8.0 compact sketch-buffer allocation and "
            "overhead measurement on gpt2. Do not enable active routing."
        )
    elif ratio is not None and ratio < 0.25:
        next_step = (
            "Keep Phase 8 deferred and improve or validate the offline/dry-run "
            "candidate policy before runtime memory work."
        )
    else:
        next_step = (
            "Resolve missing artifacts, warnings, output mismatches, or failed "
            "Phase 7 stages before starting Phase 8."
        )

    return {
        "input_paths": {
            "pipeline_summary": str(pipeline_summary_path),
            "memory_comparison": str(memory_comparison_path),
            "event_estimate": paths["event_estimate_json"],
        },
        "artifacts_present": artifacts_present,
        "checks": {
            "pipeline_success": pipeline_success,
            "pipeline_dry_run": pipeline_dry_run,
            "all_stages_succeeded": stages_succeeded,
            "greedy_outputs_match": outputs_match,
            "observer_events_exported": events_exported,
            **event_checks,
        },
        "measured_runtime_drop_observed": measured_drop_observed,
        "measured_runtime_reduction": measured_runtime_reduction,
        "theoretical_reduction": ratio,
        "theoretical_reduction_classification": classification,
        "theoretical_threshold_for_phase8_met": threshold_met,
        "warnings": list(dict.fromkeys(warnings)),
        "recommended_next_step": next_step,
        "phase8_ready": phase8_ready,
        "phase8_scope": (
            "sketch-buffer overhead measurement only; active routing remains "
            "out of scope"
        ),
        "estimated_only_for_savings": True,
    }


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [
        "# Kivo-VD Phase 7 Readiness Gate",
        "",
        "**Current claim:** Kivo-VD has validated dry-run runtime "
        "instrumentation and theoretical active-KV memory accounting. It has "
        "not demonstrated measured runtime KV memory reduction.",
        "",
        "## Decision",
        "",
        f"- Phase 8.0 ready: `{_format(report['phase8_ready'])}`",
        f"- Allowed scope: {report['phase8_scope']}",
        f"- Recommended next step: {report['recommended_next_step']}",
        "",
        "## Evidence Checks",
        "",
        "| check | value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | `{_format(value)}` |"
        for name, value in checks.items()
    )
    lines.extend([
        "",
        "## Reduction Signal",
        "",
        "| field | value |",
        "| --- | ---: |",
        (
            "| Theoretical reduction | "
            f"`{_format(report['theoretical_reduction'])}` |"
        ),
        (
            "| Classification | "
            f"`{report['theoretical_reduction_classification']}` |"
        ),
        (
            "| Measured runtime drop observed | "
            f"`{_format(report['measured_runtime_drop_observed'])}` |"
        ),
        (
            "| Measured runtime reduction | "
            f"`{_format(report['measured_runtime_reduction'])}` |"
        ),
        "",
        "The thresholds are research heuristics, not memory or quality claims:",
        "",
        "- below 10%: probably not worth runtime work;",
        "- 10% to below 25%: weak signal; continue offline testing;",
        "- 25% to below 40%: promising for a first overhead experiment;",
        "- 40% or above: strong research signal only if quality risk is "
        "controlled.",
        "",
        "## Proven Vs Not Proven",
        "",
        "Proven:",
        "",
        "- Phase 7 artifacts can be checked reproducibly;",
        "- theoretical selected/skipped KV accounting can inform a go/no-go "
        "decision.",
        "",
        "Not proven:",
        "",
        "- measured runtime KV memory reduction;",
        "- active routing, latency improvement, or quality preservation.",
        "",
        "## Warnings",
        "",
    ])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    report = build_readiness_report(
        pipeline_summary_path=args.pipeline_summary,
        memory_comparison_path=args.memory_comparison,
        event_estimate_path=args.event_estimate,
    )
    _write(
        args.output_json,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "output_md": args.output_md,
                "phase8_ready": report["phase8_ready"],
                "theoretical_reduction": report["theoretical_reduction"],
                "recommended_next_step": report["recommended_next_step"],
                "measured_runtime_reduction": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
