#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S4.1 low-overhead measurement report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.kivo_vd import run_source_s4_1_low_overhead_measurement as runner


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S4.1 low-overhead measurement output."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement_validation.md"
        ),
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S4.1 input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def _claim_paths(value: Any, path: str = "") -> list[str]:
    if isinstance(value, dict):
        found: list[str] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            key_lower = key.lower()
            if (
                "memory" in key_lower
                or "latency" in key_lower
                or "selected_attention" in key_lower
                or "performance_claim" in key_lower
            ) and child is True:
                found.append(child_path)
            found.extend(_claim_paths(child, child_path))
        return found
    if isinstance(value, list):
        found: list[str] = []
        for index, child in enumerate(value):
            found.extend(_claim_paths(child, f"{path}[{index}]"))
        return found
    return []


def _mode_blockers(mode_report: dict[str, Any]) -> list[str]:
    reasons = set(mode_report.get("blocker_reasons") or [])
    event_summary = mode_report.get("event_summary") or {}
    counter_summary = mode_report.get("counter_summary") or {}
    reasons.update(event_summary.get("blocker_reasons") or [])
    for key, value in (counter_summary.get("blocker_reason_counts") or {}).items():
        if int(value or 0) > 0:
            reasons.add(str(key))
    return sorted(reasons)


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    total_prompts = int(report.get("total_prompts", 0) or 0)
    repeats = int(report.get("repeats", 0) or 0)
    warmup = int(report.get("warmup", 0) or 0)
    expected_measured_runs = total_prompts * repeats
    expected_total_runs = total_prompts * (repeats + warmup) * len(runner.MODE_ORDER)

    if total_prompts <= 0:
        errors.append("total_prompts must be > 0")
    if repeats <= 0:
        errors.append("repeats must be > 0")
    if report.get("modes") != runner.MODE_ORDER:
        errors.append("modes must match the S4.1 mode order")
    if report.get("passed") is not True:
        errors.append("passed must be true")

    per_mode = report.get("per_mode")
    if not isinstance(per_mode, dict):
        errors.append("per_mode must be a mapping")
        per_mode = {}

    required_top_level = [
        "baseline_success_count",
        "recent_window_verbose_success_count",
        "recent_window_counters_success_count",
        "sketch_active_verbose_success_count",
        "sketch_active_counters_success_count",
        "verbose_event_record_count",
        "counter_event_count",
        "measured_runtime_reduction",
        "memory_claim_allowed",
        "quality_claim_allowed",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
        "caveats",
    ]
    missing_top_level = [field for field in required_top_level if field not in report]
    if missing_top_level:
        errors.append(f"missing required fields: {', '.join(missing_top_level)}")

    claim_paths = sorted(set(_claim_paths(report)))
    if claim_paths:
        errors.append(
            "memory, latency, selected-attention, or performance claims must "
            f"remain false: {', '.join(claim_paths)}"
        )
    for field in [
        "memory_claim_allowed",
        "quality_claim_allowed",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")

    if len(report.get("caveats", [])) <= 0:
        errors.append("caveats must be present")

    baseline = per_mode.get(runner.BASELINE_MODE, {})
    recent_verbose = per_mode.get(runner.RECENT_WINDOW_VERBOSE_MODE, {})
    recent_counters = per_mode.get(runner.RECENT_WINDOW_COUNTERS_MODE, {})
    sketch_verbose = per_mode.get(runner.SKETCH_ACTIVE_VERBOSE_MODE, {})
    sketch_counters = per_mode.get(runner.SKETCH_ACTIVE_COUNTERS_MODE, {})

    for mode_name, mode_report in [
        (runner.BASELINE_MODE, baseline),
        (runner.RECENT_WINDOW_VERBOSE_MODE, recent_verbose),
        (runner.RECENT_WINDOW_COUNTERS_MODE, recent_counters),
        (runner.SKETCH_ACTIVE_VERBOSE_MODE, sketch_verbose),
        (runner.SKETCH_ACTIVE_COUNTERS_MODE, sketch_counters),
    ]:
        if not isinstance(mode_report, dict):
            errors.append(f"{mode_name} summary missing")
            continue
        if mode_report.get("success_count") != expected_measured_runs:
            errors.append(f"{mode_name} success_count must equal measured run count")
        if mode_report.get("counter_summary") is None:
            errors.append(f"{mode_name} counter_summary missing")
        if mode_report.get("event_summary") is None:
            errors.append(f"{mode_name} event_summary missing")

    if report.get("baseline_success_count") != expected_measured_runs:
        errors.append("baseline_success_count must equal measured run count")
    if report.get("recent_window_verbose_success_count") != expected_measured_runs:
        errors.append(
            "recent_window_verbose_success_count must equal measured run count"
        )
    if report.get("recent_window_counters_success_count") != expected_measured_runs:
        errors.append(
            "recent_window_counters_success_count must equal measured run count"
        )
    if report.get("sketch_active_verbose_success_count") != expected_measured_runs:
        errors.append(
            "sketch_active_verbose_success_count must equal measured run count"
        )
    if report.get("sketch_active_counters_success_count") != expected_measured_runs:
        errors.append(
            "sketch_active_counters_success_count must equal measured run count"
        )

    for field in [
        "recent_window_verbose_vs_baseline_latency_ratio",
        "recent_window_counters_vs_baseline_latency_ratio",
        "recent_window_counters_vs_verbose_latency_ratio",
        "sketch_active_verbose_vs_baseline_latency_ratio",
        "sketch_active_counters_vs_baseline_latency_ratio",
        "sketch_active_counters_vs_verbose_latency_ratio",
    ]:
        if report.get(field) is None:
            errors.append(f"{field} must be present")

    if not isinstance(report.get("logging_overhead_reduction"), dict):
        errors.append("logging_overhead_reduction must be present")
    else:
        if (
            report["logging_overhead_reduction"].get(
                "recent_window_latency_improvement_from_counters"
            )
            is None
        ):
            errors.append(
                "recent_window_latency_improvement_from_counters must be present"
            )
        if (
            report["logging_overhead_reduction"].get(
                "sketch_active_latency_improvement_from_counters"
            )
            is None
        ):
            errors.append(
                "sketch_active_latency_improvement_from_counters must be present"
            )

    recent_verbose_events = int(
        recent_verbose.get("verbose_event_record_count", 0) or 0
    )
    recent_counter_events = int(recent_counters.get("verbose_event_record_count", 0) or 0)
    sketch_verbose_events = int(
        sketch_verbose.get("verbose_event_record_count", 0) or 0
    )
    sketch_counter_events = int(
        sketch_counters.get("verbose_event_record_count", 0) or 0
    )

    if recent_verbose_events <= 0:
        errors.append("recent_window_verbose must emit verbose records")
    if sketch_verbose_events <= 0:
        errors.append("sketch_active_verbose must emit verbose records")
    if recent_counter_events >= recent_verbose_events:
        errors.append(
            "recent_window_counters must emit fewer verbose records than recent_window_verbose"
        )
    if sketch_counter_events >= sketch_verbose_events:
        errors.append(
            "sketch_active_counters must emit fewer verbose records than sketch_active_verbose"
        )

    recent_counter_summary = recent_counters.get("counter_summary") or {}
    sketch_counter_summary = sketch_counters.get("counter_summary") or {}
    if int(recent_counter_summary.get("event_count", 0) or 0) <= 0:
        errors.append("recent_window_counters counter_summary must have events")
    if int(sketch_counter_summary.get("event_count", 0) or 0) <= 0:
        errors.append("sketch_active_counters counter_summary must have events")

    if (
        int(recent_counter_summary.get("mutation_applied_count", 0) or 0) <= 0
        and not _mode_blockers(recent_counters)
    ):
        errors.append(
            "recent_window_counters must have mutation_applied_count > 0 or blockers"
        )
    if (
        int(sketch_counter_summary.get("sketch_computed_count", 0) or 0) <= 0
        and not _mode_blockers(sketch_counters)
    ):
        errors.append(
            "sketch_active_counters must have sketch_computed_count > 0 or blockers"
        )
    if (
        int(sketch_counter_summary.get("mutation_applied_count", 0) or 0) <= 0
        and not _mode_blockers(sketch_counters)
    ):
        errors.append(
            "sketch_active_counters must have mutation_applied_count > 0 or blockers"
        )

    all_success = bool(
        baseline.get("success_count") == expected_measured_runs
        and recent_verbose.get("success_count") == expected_measured_runs
        and recent_counters.get("success_count") == expected_measured_runs
        and sketch_verbose.get("success_count") == expected_measured_runs
        and sketch_counters.get("success_count") == expected_measured_runs
    )
    if report.get("measured_runtime_reduction") is True:
        recent_ratio = report.get(
            "recent_window_counters_vs_baseline_latency_ratio"
        )
        sketch_ratio = report.get("sketch_active_counters_vs_baseline_latency_ratio")
        if not (
            all_success
            and recent_ratio is not None
            and sketch_ratio is not None
            and float(recent_ratio) <= 0.95
            and float(sketch_ratio) <= 0.95
        ):
            errors.append(
                "measured_runtime_reduction may only be true when counters-mode "
                "latency beats baseline by at least 5% with all runs successful"
            )
    elif report.get("measured_runtime_reduction") is not False:
        errors.append("measured_runtime_reduction must be false or a strict win")

    total_verbose_records = sum(
        int(mode_report.get("verbose_event_record_count", 0) or 0)
        for mode_report in per_mode.values()
    )
    total_counter_events = sum(
        int(mode_report.get("counter_event_count", 0) or 0)
        for mode_report in per_mode.values()
    )
    if total_verbose_records != int(report.get("verbose_event_record_count", 0) or 0):
        errors.append("verbose_event_record_count must match per-mode totals")
    if total_counter_events != int(report.get("counter_event_count", 0) or 0):
        errors.append("counter_event_count must match per-mode totals")

    if report.get("passed") is not True and not errors:
        errors.append("passed must be true")

    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "repeats": repeats,
        "warmup": warmup,
        "expected_measured_runs": expected_measured_runs,
        "expected_total_runs": expected_total_runs,
        "baseline_success_count": report.get("baseline_success_count"),
        "recent_window_verbose_success_count": report.get(
            "recent_window_verbose_success_count"
        ),
        "recent_window_counters_success_count": report.get(
            "recent_window_counters_success_count"
        ),
        "sketch_active_verbose_success_count": report.get(
            "sketch_active_verbose_success_count"
        ),
        "sketch_active_counters_success_count": report.get(
            "sketch_active_counters_success_count"
        ),
        "verbose_event_record_count": report.get("verbose_event_record_count"),
        "counter_event_count": report.get("counter_event_count"),
        "recent_window_verbose_vs_baseline_latency_ratio": report.get(
            "recent_window_verbose_vs_baseline_latency_ratio"
        ),
        "recent_window_counters_vs_baseline_latency_ratio": report.get(
            "recent_window_counters_vs_baseline_latency_ratio"
        ),
        "recent_window_counters_vs_verbose_latency_ratio": report.get(
            "recent_window_counters_vs_verbose_latency_ratio"
        ),
        "sketch_active_verbose_vs_baseline_latency_ratio": report.get(
            "sketch_active_verbose_vs_baseline_latency_ratio"
        ),
        "sketch_active_counters_vs_baseline_latency_ratio": report.get(
            "sketch_active_counters_vs_baseline_latency_ratio"
        ),
        "sketch_active_counters_vs_verbose_latency_ratio": report.get(
            "sketch_active_counters_vs_verbose_latency_ratio"
        ),
        "measured_runtime_reduction": report.get("measured_runtime_reduction"),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S4.1 Low-Overhead Measurement Validation",
        "",
        f"- Passed: `{result['validation_passed']}`",
        f"- Total prompts: `{result['total_prompts']}`",
        f"- Repeats: `{result['repeats']}`",
        f"- Warmup: `{result['warmup']}`",
        f"- Expected measured runs: `{result['expected_measured_runs']}`",
        f"- Baseline success count: `{result['baseline_success_count']}`",
        (
            "- Recent-window verbose success count: "
            f"`{result['recent_window_verbose_success_count']}`"
        ),
        (
            "- Recent-window counters success count: "
            f"`{result['recent_window_counters_success_count']}`"
        ),
        (
            "- Sketch-active verbose success count: "
            f"`{result['sketch_active_verbose_success_count']}`"
        ),
        (
            "- Sketch-active counters success count: "
            f"`{result['sketch_active_counters_success_count']}`"
        ),
        f"- Verbose event record count: `{result['verbose_event_record_count']}`",
        f"- Counter event count: `{result['counter_event_count']}`",
        "",
        "## Errors",
        "",
    ]
    if result["errors"]:
        lines.extend(f"- {error}" for error in result["errors"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This validation separates verbose JSONL overhead from counters-only overhead. "
            "It still does not prove memory reduction, quality preservation, or selected attention.",
            "",
        ]
    )
    return "\n".join(lines)


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = validate_report(load_report(args.input_json))
    except Exception as exc:
        result = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
            "repeats": 0,
            "warmup": 0,
            "expected_measured_runs": 0,
            "expected_total_runs": 0,
            "baseline_success_count": 0,
            "recent_window_verbose_success_count": 0,
            "recent_window_counters_success_count": 0,
            "sketch_active_verbose_success_count": 0,
            "sketch_active_counters_success_count": 0,
            "verbose_event_record_count": 0,
            "counter_event_count": 0,
        }
    _write(args.output_json, json.dumps(result, indent=2) + "\n")
    _write(args.output_md, render_markdown(result))
    print(
        json.dumps(
            {
                "validation_passed": result["validation_passed"],
                "total_prompts": result["total_prompts"],
                "repeats": result["repeats"],
                "baseline_success_count": result["baseline_success_count"],
                "recent_window_counters_success_count": result[
                    "recent_window_counters_success_count"
                ],
                "sketch_active_counters_success_count": result[
                    "sketch_active_counters_success_count"
                ],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
