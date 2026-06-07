#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Phase 9 evidence for limited standalone attention experiments."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_KEYS = (
    "materialization_json",
    "materialization_markdown",
    "comparison_json",
    "comparison_markdown",
    "pipeline_summary",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Kivo-VD Phase 9 readiness for Phase 10."
    )
    parser.add_argument("--pipeline-summary", required=True)
    parser.add_argument("--materialization", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--event-estimate")
    parser.add_argument("--sketch-accounting")
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase9_readiness.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase9_readiness.md",
    )
    return parser.parse_args(argv)


def classify_materialization_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "unavailable"
    if ratio < 0:
        raise ValueError("materialization ratio must be non-negative")
    if ratio < 0.25:
        return "strong_materialization_compression_signal"
    if ratio < 0.50:
        return "promising"
    if ratio < 0.80:
        return "moderate_signal"
    return "weak_signal"


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


def _present(path: str | None) -> bool:
    return path is not None and Path(path).is_file()


def _artifact_paths(
    pipeline: dict[str, Any] | None,
    pipeline_path: str | Path,
    materialization_path: str | Path,
    comparison_path: str | Path,
    event_estimate_path: str | Path | None,
    sketch_accounting_path: str | Path | None,
) -> dict[str, str | None]:
    outputs = pipeline.get("output_files", {}) if pipeline else {}
    parameters = pipeline.get("parameters", {}) if pipeline else {}
    if not isinstance(outputs, dict):
        outputs = {}
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "materialization_json": str(materialization_path),
        "materialization_markdown": outputs.get(
            "materialization_markdown"
        ),
        "comparison_json": str(comparison_path),
        "comparison_markdown": outputs.get("comparison_markdown"),
        "pipeline_summary": str(pipeline_path),
        "event_estimate": (
            str(event_estimate_path)
            if event_estimate_path is not None
            else parameters.get("event_estimate")
        ),
        "sketch_accounting": (
            str(sketch_accounting_path)
            if sketch_accounting_path is not None
            else parameters.get("sketch_accounting")
        ),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _materialization_checks(
    materialization: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    if materialization is None:
        return {
            "events_processed": 0,
            "events_processed_nonzero": False,
            "average_selected_blocks": None,
            "selected_blocks_nonzero": False,
            "average_copy_time_ms": None,
            "copy_time_observed": False,
            "average_materialization_ratio": None,
            "materialization_ratio_below_one": False,
            "preview_only_event_count": 0,
            "missing_selected_ids_warning": True,
        }
    aggregate = materialization.get("aggregate", {})
    rows = materialization.get("per_event_rows", [])
    source_warnings = materialization.get("warnings", [])
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(rows, list):
        rows = []
    if not isinstance(source_warnings, list):
        source_warnings = ["materialization warnings field is malformed"]
    warnings.extend(
        f"materialization: {warning}" for warning in source_warnings
    )
    events = _number(materialization.get("num_events_processed")) or 0
    selected = _number(aggregate.get("average_selected_blocks"))
    copy_ms = _number(aggregate.get("average_copy_time_ms"))
    ratio = _number(aggregate.get("average_materialization_ratio"))
    preview_count_from_rows = sum(
        isinstance(row, dict)
        and row.get("selected_ids_preview_only") is True
        for row in rows
    )
    preview_count = int(
        _number(aggregate.get("preview_only_event_count"))
        or preview_count_from_rows
    )
    full_id_count = int(
        _number(aggregate.get("full_block_ids_exported_count")) or 0
    )
    missing_ids = any(
        "lacks selected block id" in str(warning).lower()
        or "missing selected block id" in str(warning).lower()
        for warning in source_warnings
    )
    if events <= 0:
        warnings.append("no materialization events were processed")
    if selected is None or selected <= 0:
        warnings.append("average selected block count is zero or unavailable")
    if copy_ms is None:
        warnings.append("copy time was not measured")
    if ratio is None:
        warnings.append("average materialization ratio is unavailable")
    elif ratio >= 1.0:
        warnings.append("average materialization ratio is not below 1.0")
    if preview_count:
        warnings.append(
            f"{preview_count} materialization rows use preview-only block IDs"
        )
    return {
        "events_processed": int(events),
        "events_processed_nonzero": events > 0,
        "average_selected_blocks": selected,
        "selected_blocks_nonzero": selected is not None and selected > 0,
        "average_copy_time_ms": copy_ms,
        "copy_time_observed": copy_ms is not None,
        "average_materialization_ratio": ratio,
        "materialization_ratio_below_one": (
            ratio is not None and ratio < 1.0
        ),
        "preview_only_event_count": preview_count,
        "full_block_ids_exported_count": full_id_count,
        "missing_selected_ids_warning": missing_ids,
    }


def _caveat_checks(
    pipeline: dict[str, Any] | None,
    materialization: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, bool]:
    materialization_caveats = (
        materialization.get("caveats", {}) if materialization else {}
    )
    comparison_caveats = comparison.get("caveats", {}) if comparison else {}
    if not isinstance(materialization_caveats, dict):
        materialization_caveats = {}
    if not isinstance(comparison_caveats, dict):
        comparison_caveats = {}

    checks = {
        "synthetic_kv": (
            pipeline is not None
            and pipeline.get("synthetic_kv") is True
            and materialization_caveats.get("synthetic_kv") is True
            and comparison_caveats.get("synthetic_kv") is True
        ),
        "outside_attention_path": (
            pipeline is not None
            and pipeline.get("outside_attention_path") is True
            and materialization_caveats.get("outside_attention_path") is True
            and comparison_caveats.get("outside_attention_path") is True
        ),
        "full_kv_still_allocated": (
            pipeline is not None
            and pipeline.get("full_kv_still_allocated") is True
            and materialization_caveats.get(
                "full_kv_still_allocated"
            )
            is True
            and comparison_caveats.get("full_kv_still_allocated") is True
        ),
        "active_routing_false": (
            pipeline is not None
            and pipeline.get("active_routing") is False
            and materialization_caveats.get("active_routing") is False
            and comparison_caveats.get("active_routing") is False
        ),
        "measured_runtime_reduction_false": (
            pipeline is not None
            and pipeline.get("measured_runtime_reduction") is False
            and materialization_caveats.get(
                "measured_runtime_reduction"
            )
            is False
            and comparison_caveats.get(
                "measured_runtime_reduction"
            )
            is False
        ),
        "quality_not_measured": (
            pipeline is not None
            and pipeline.get("quality_not_measured") is True
            and comparison_caveats.get("quality_not_measured") is True
        ),
    }
    warnings.extend(
        f"required caveat is not preserved: {name}"
        for name, passed in checks.items()
        if not passed
    )
    return checks


def build_readiness_report(
    *,
    pipeline_summary_path: str | Path,
    materialization_path: str | Path,
    comparison_path: str | Path,
    event_estimate_path: str | Path | None = None,
    sketch_accounting_path: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    pipeline = _load_optional_json(
        pipeline_summary_path, "pipeline summary", warnings
    )
    materialization = _load_optional_json(
        materialization_path, "materialization", warnings
    )
    comparison = _load_optional_json(
        comparison_path, "comparison", warnings
    )
    paths = _artifact_paths(
        pipeline,
        pipeline_summary_path,
        materialization_path,
        comparison_path,
        event_estimate_path,
        sketch_accounting_path,
    )
    artifacts_present = {
        key: _present(paths[key]) for key in REQUIRED_OUTPUT_KEYS
    }
    warnings.extend(
        f"required artifact is missing: {key}"
        for key, present in artifacts_present.items()
        if not present
    )

    event_estimate_present = _present(paths["event_estimate"])
    sketch_accounting_present = _present(paths["sketch_accounting"])
    if not event_estimate_present:
        warnings.append("Phase 7 event estimate is missing")
    if paths["sketch_accounting"] is not None and not sketch_accounting_present:
        warnings.append("optional Phase 8 sketch accounting is missing")

    pipeline_success = bool(pipeline and pipeline.get("success"))
    pipeline_dry_run = bool(pipeline and pipeline.get("dry_run"))
    stages = pipeline.get("stages", []) if pipeline else []
    stages_succeeded = bool(stages) and all(
        isinstance(stage, dict) and stage.get("status") == "succeeded"
        for stage in stages
    )
    if not pipeline_success:
        warnings.append("Phase 9.2 pipeline did not report success")
    if pipeline_dry_run:
        warnings.append("pipeline summary is a dry-run plan, not execution")
    if not stages_succeeded:
        warnings.append("not all Phase 9.2 pipeline stages succeeded")

    materialization_checks = _materialization_checks(
        materialization, warnings
    )
    caveat_checks = _caveat_checks(
        pipeline, materialization, comparison, warnings
    )
    phase10_ready = all([
        all(artifacts_present.values()),
        event_estimate_present,
        pipeline_success,
        not pipeline_dry_run,
        stages_succeeded,
        materialization_checks["events_processed_nonzero"],
        materialization_checks["selected_blocks_nonzero"],
        materialization_checks["copy_time_observed"],
        materialization_checks["materialization_ratio_below_one"],
        not materialization_checks["missing_selected_ids_warning"],
        materialization_checks["preview_only_event_count"] == 0,
        all(caveat_checks.values()),
    ])
    ratio = materialization_checks["average_materialization_ratio"]
    classification = classify_materialization_ratio(ratio)
    if phase10_ready:
        next_step = (
            "Proceed only to standalone torch reference-attention "
            "equivalence experiments on synthetic Q/K/V outside vLLM."
        )
    else:
        next_step = (
            "Resolve missing artifacts, failed stages, missing selected IDs, "
            "unavailable measurements, or caveat regressions before Phase 10."
        )

    return {
        "input_paths": paths,
        "artifacts_present": artifacts_present,
        "context_artifacts": {
            "event_estimate_present": event_estimate_present,
            "sketch_accounting_present": sketch_accounting_present,
        },
        "checks": {
            "pipeline_success": pipeline_success,
            "pipeline_dry_run": pipeline_dry_run,
            "all_stages_succeeded": stages_succeeded,
            **materialization_checks,
            **caveat_checks,
        },
        "materialization_ratio": ratio,
        "materialization_ratio_classification": classification,
        "copy_time_status": "observed_only",
        "warnings": list(dict.fromkeys(warnings)),
        "phase10_ready": phase10_ready,
        "allowed_scope": (
            "standalone selected-KV torch reference-attention experiments "
            "on synthetic tensors outside vLLM only"
        ),
        "recommended_next_step": next_step,
        "synthetic_kv": True,
        "outside_attention_path": True,
        "full_kv_still_allocated": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "quality_not_measured": True,
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
    lines = [
        "# Kivo-VD Phase 9 Readiness Gate",
        "",
        "**Current claim:** Kivo-VD can dry-run selected block decisions, "
        "estimate active-KV savings, account for sketch-buffer overhead, and "
        "materialize selected KV subsets into synthetic temporary buffers "
        "outside attention. It has not demonstrated active attention "
        "routing, measured memory reduction, or quality preservation.",
        "",
        "## Decision",
        "",
        f"- Phase 10 ready: `{_format(report['phase10_ready'])}`",
        f"- Allowed scope: {report['allowed_scope']}",
        f"- Recommended next step: {report['recommended_next_step']}",
        "",
        "## Evidence Checks",
        "",
        "| check | value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | `{_format(value)}` |"
        for name, value in report["checks"].items()
    )
    lines.extend([
        "",
        "## Materialization Signal",
        "",
        "| field | value |",
        "| --- | ---: |",
        (
            "| Average materialization ratio | "
            f"`{_format(report['materialization_ratio'])}` |"
        ),
        (
            "| Classification | "
            f"`{report['materialization_ratio_classification']}` |"
        ),
        "| Copy time classification | `observed only` |",
        "",
        "Research heuristics:",
        "",
        "- ratio at or above 0.80: weak signal;",
        "- 0.50 to below 0.80: moderate signal;",
        "- 0.25 to below 0.50: promising;",
        "- below 0.25: strong materialization compression signal.",
        "",
        "Copy time has no hard threshold yet. Repeated-run validation is "
        "required before performance conclusions.",
        "",
        "## Allowed Phase 10 Sequence",
        "",
        "1. Tiny standalone synthetic Q/K/V attention equivalence.",
        "2. Selected-KV torch reference attention outside vLLM.",
        "3. Selected versus full synthetic-attention output comparison.",
        "4. Only later consider an isolated vLLM-adjacent prototype.",
        "",
        "## Initially Out Of Scope",
        "",
        "- Block-table or slot-mapping mutation.",
        "- Scheduler behavior changes.",
        "- Production attention-kernel changes.",
        "- Selected-KV attention inside real vLLM.",
        "- Memory, latency, or quality claims.",
        "",
        "## Caveats",
        "",
        "- KV tensors are synthetic.",
        "- Materialization remains outside the attention path.",
        "- Full KV is still allocated.",
        "- Active routing remains false.",
        "- Measured runtime memory reduction remains false.",
        "- Quality is not measured.",
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
    try:
        args = _parse_args()
        report = build_readiness_report(
            pipeline_summary_path=args.pipeline_summary,
            materialization_path=args.materialization,
            comparison_path=args.comparison,
            event_estimate_path=args.event_estimate,
            sketch_accounting_path=args.sketch_accounting,
        )
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "phase10_ready": report["phase10_ready"],
                    "allowed_scope": report["allowed_scope"],
                    "materialization_ratio_classification": report[
                        "materialization_ratio_classification"
                    ],
                    "warnings": report["warnings"],
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "synthetic_kv": True,
                    "outside_attention_path": True,
                    "full_kv_still_allocated": True,
                    "active_routing": False,
                    "measured_runtime_reduction": False,
                    "quality_not_measured": True,
                },
                separators=(",", ":"),
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "synthetic_kv": True,
                    "outside_attention_path": True,
                    "full_kv_still_allocated": True,
                    "active_routing": False,
                    "measured_runtime_reduction": False,
                    "quality_not_measured": True,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
