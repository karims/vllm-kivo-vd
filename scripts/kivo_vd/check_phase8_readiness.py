#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Phase 8 evidence for a limited Phase 9 experiment."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_KEYS = (
    "sketch_overhead_json",
    "sketch_overhead_markdown",
    "overhead_vs_savings_json",
    "overhead_vs_savings_markdown",
    "event_accounting_json",
    "event_accounting_markdown",
)
ALLOWED_CUMULATIVE_CLASSES = {"excellent", "acceptable"}
CLASS_RANK = {
    "excellent": 0,
    "acceptable": 1,
    "questionable": 2,
    "poor": 3,
    "unavailable": 4,
}
BREAK_EVEN_RANK = {
    "immediate": 0,
    "fast": 1,
    "moderate": 2,
    "slow": 3,
    "not_applicable": 4,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Kivo-VD Phase 8 readiness for Phase 9."
    )
    parser.add_argument("--pipeline-summary", required=True)
    parser.add_argument("--event-aware-accounting", required=True)
    parser.add_argument("--overhead-vs-savings")
    parser.add_argument("--sketch-overhead")
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase8_readiness.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase8_readiness.md",
    )
    return parser.parse_args(argv)


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
    accounting_path: str | Path,
    overhead_vs_savings_path: str | Path | None,
    sketch_overhead_path: str | Path | None,
) -> dict[str, str | None]:
    output_files = pipeline.get("output_files", {}) if pipeline else {}
    if not isinstance(output_files, dict):
        output_files = {}
    return {
        "sketch_overhead_json": (
            str(sketch_overhead_path)
            if sketch_overhead_path is not None
            else output_files.get("sketch_overhead_json")
        ),
        "sketch_overhead_markdown": output_files.get(
            "sketch_overhead_markdown"
        ),
        "overhead_vs_savings_json": (
            str(overhead_vs_savings_path)
            if overhead_vs_savings_path is not None
            else output_files.get("overhead_vs_savings_json")
        ),
        "overhead_vs_savings_markdown": output_files.get(
            "overhead_vs_savings_markdown"
        ),
        "event_accounting_json": str(accounting_path),
        "event_accounting_markdown": output_files.get(
            "event_accounting_markdown"
        ),
        "pipeline_summary": str(pipeline_path),
    }


def _config_summary(row: dict[str, Any]) -> dict[str, Any]:
    cumulative = row.get("cumulative_request_model", {})
    break_even = row.get("break_even_model", {})
    if not isinstance(cumulative, dict):
        cumulative = {}
    if not isinstance(break_even, dict):
        break_even = {}
    return {
        "sketch_type": row.get("sketch_type"),
        "sketch_dim": row.get("sketch_dim"),
        "cumulative_overhead_ratio": cumulative.get(
            "overhead_vs_cumulative_skipped_kv"
        ),
        "cumulative_classification": cumulative.get("classification"),
        "break_even_events": break_even.get("break_even_events"),
        "break_even_classification": break_even.get(
            "break_even_events_classification"
        ),
        "break_even_skipped_blocks": break_even.get(
            "break_even_skipped_blocks"
        ),
    }


def _eligible_configs(
    accounting: dict[str, Any] | None,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if accounting is None:
        return [], []
    rows = accounting.get("accounting_rows")
    if not isinstance(rows, list) or not rows:
        warnings.append("event-aware accounting has no configuration rows")
        return [], []
    configs = [
        _config_summary(row) for row in rows if isinstance(row, dict)
    ]
    eligible = [
        config
        for config in configs
        if config["sketch_dim"] in {16, 32}
        and config["cumulative_classification"]
        in ALLOWED_CUMULATIVE_CLASSES
    ]
    eligible.sort(
        key=lambda config: (
            CLASS_RANK.get(
                str(config["cumulative_classification"]), 99
            ),
            (
                config["cumulative_overhead_ratio"]
                if isinstance(
                    config["cumulative_overhead_ratio"], int | float
                )
                else float("inf")
            ),
            BREAK_EVEN_RANK.get(
                str(config["break_even_classification"]), 99
            ),
            config["sketch_dim"],
            str(config["sketch_type"]),
        )
    )
    if not eligible:
        warnings.append(
            "no dim-16 or dim-32 configuration is excellent or acceptable"
        )
    return configs, eligible


def _caveat_checks(
    pipeline: dict[str, Any] | None,
    accounting: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, bool]:
    caveats = accounting.get("caveats", {}) if accounting else {}
    if not isinstance(caveats, dict):
        caveats = {}
    checks = {
        "theoretical_only": (
            caveats.get("theoretical_only") is True
            and bool(pipeline and pipeline.get(
                "savings_are_theoretical_only"
            ))
        ),
        "measured_runtime_reduction_false": (
            caveats.get("measured_runtime_reduction") is False
            and pipeline is not None
            and pipeline.get("measured_runtime_reduction") is False
        ),
        "active_routing_false": (
            caveats.get("active_routing") is False
            and pipeline is not None
            and pipeline.get("active_routing") is False
        ),
        "full_kv_still_allocated": (
            caveats.get("full_kv_still_allocated") is True
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
    event_aware_accounting_path: str | Path,
    overhead_vs_savings_path: str | Path | None = None,
    sketch_overhead_path: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    pipeline = _load_optional_json(
        pipeline_summary_path,
        "pipeline summary",
        warnings,
    )
    accounting = _load_optional_json(
        event_aware_accounting_path,
        "event-aware accounting",
        warnings,
    )
    paths = _artifact_paths(
        pipeline,
        pipeline_summary_path,
        event_aware_accounting_path,
        overhead_vs_savings_path,
        sketch_overhead_path,
    )
    artifacts_present = {
        key: _present(paths[key]) for key in REQUIRED_OUTPUT_KEYS
    }
    warnings.extend(
        f"required artifact is missing: {key}"
        for key, present in artifacts_present.items()
        if not present
    )

    pipeline_success = bool(pipeline and pipeline.get("success"))
    pipeline_dry_run = bool(pipeline and pipeline.get("dry_run"))
    stages = pipeline.get("stages", []) if pipeline else []
    stages_succeeded = bool(stages) and all(
        isinstance(stage, dict) and stage.get("status") == "succeeded"
        for stage in stages
    )
    if not pipeline_success:
        warnings.append("Phase 8.3 pipeline did not report success")
    if pipeline_dry_run:
        warnings.append("pipeline summary is a dry-run plan, not execution")
    if not stages_succeeded:
        warnings.append("not all Phase 8.3 pipeline stages succeeded")

    configs, eligible = _eligible_configs(accounting, warnings)
    caveat_checks = _caveat_checks(pipeline, accounting, warnings)
    phase9_ready = all([
        all(artifacts_present.values()),
        pipeline_success,
        not pipeline_dry_run,
        stages_succeeded,
        bool(eligible),
        all(caveat_checks.values()),
    ])
    best_config = eligible[0] if eligible else None
    if phase9_ready:
        next_step = (
            "Proceed only to Phase 9 selected-KV materialization into "
            "temporary buffers outside the attention path. Do not mutate "
            "block tables, scheduling, or attention."
        )
    else:
        next_step = (
            "Resolve missing artifacts, failed stages, caveat regressions, or "
            "weak cumulative accounting before Phase 9."
        )

    return {
        "input_paths": paths,
        "artifacts_present": artifacts_present,
        "checks": {
            "pipeline_success": pipeline_success,
            "pipeline_dry_run": pipeline_dry_run,
            "all_stages_succeeded": stages_succeeded,
            "eligible_compressed_config_exists": bool(eligible),
            **caveat_checks,
        },
        "configuration_count": len(configs),
        "eligible_configs": eligible,
        "best_recommended_config": best_config,
        "best_cumulative_overhead_classification": (
            best_config["cumulative_classification"]
            if best_config is not None
            else None
        ),
        "best_break_even_classification": (
            best_config["break_even_classification"]
            if best_config is not None
            else None
        ),
        "warnings": list(dict.fromkeys(warnings)),
        "phase9_ready": phase9_ready,
        "phase9_scope": (
            "selected-KV gather/copy into temporary measurement buffers "
            "outside attention only"
        ),
        "recommended_next_step": next_step,
        "theoretical_only": True,
        "measured_runtime_reduction": False,
        "active_routing": False,
        "full_kv_still_allocated": True,
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
        "# Kivo-VD Phase 8 Readiness Gate",
        "",
        "**Current claim:** Kivo-VD has runtime dry-run instrumentation, "
        "theoretical active-KV accounting, and compact sketch-buffer "
        "overhead accounting. It has not demonstrated measured runtime KV "
        "memory reduction or active attention routing.",
        "",
        "## Decision",
        "",
        f"- Phase 9 ready: `{_format(report['phase9_ready'])}`",
        f"- Allowed scope: {report['phase9_scope']}",
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
        "## Best Configuration",
        "",
        "| field | value |",
        "| --- | ---: |",
    ])
    best = report["best_recommended_config"] or {}
    for field in (
        "sketch_type",
        "sketch_dim",
        "cumulative_overhead_ratio",
        "cumulative_classification",
        "break_even_events",
        "break_even_classification",
        "break_even_skipped_blocks",
    ):
        lines.append(f"| {field.replace('_', ' ')} | `{_format(best.get(field))}` |")
    lines.extend([
        "",
        "## Thresholds",
        "",
        "- cumulative overhead at or below 5%: excellent;",
        "- above 5% through 15%: acceptable;",
        "- above 15% through 30%: questionable;",
        "- above 30%: poor;",
        "- break-even events at or below 1: immediate;",
        "- at or below 4: fast;",
        "- at or below 16: moderate;",
        "- above 16: slow.",
        "",
        "These thresholds are research heuristics only.",
        "",
        "## Caveats",
        "",
        "- Savings and active-KV opportunities are theoretical only.",
        "- Full KV is still allocated.",
        "- Active routing remains false.",
        "- Measured runtime KV memory reduction remains false.",
        "- A passing gate permits temporary selected-KV materialization "
        "outside attention only.",
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
            event_aware_accounting_path=args.event_aware_accounting,
            overhead_vs_savings_path=args.overhead_vs_savings,
            sketch_overhead_path=args.sketch_overhead,
        )
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "phase9_ready": report["phase9_ready"],
                    "best_recommended_config": report[
                        "best_recommended_config"
                    ],
                    "warnings": report["warnings"],
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "theoretical_only": True,
                    "measured_runtime_reduction": False,
                    "active_routing": False,
                    "full_kv_still_allocated": True,
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
                    "theoretical_only": True,
                    "measured_runtime_reduction": False,
                    "active_routing": False,
                    "full_kv_still_allocated": True,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
