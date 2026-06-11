#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S4.0 quick measurement output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.kivo_vd import run_source_s4_0_quick_measurement as runner


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S4.0 quick measurement output."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s4_0_quick_measurement_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s4_0_quick_measurement_validation.md",
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S4.0 input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def load_events(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S4.0 events input is missing: {input_path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSONL row {line_number}: {input_path}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        events.append(event)
    return events


def split_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    recent = [
        event
        for event in events
        if event.get("schema_version") == runner.S3_2B_SCHEMA
    ]
    sketch_plan = [
        event
        for event in events
        if event.get("schema_version") == runner.S3_3C_PLAN_SCHEMA
    ]
    sketch_metadata = [
        event
        for event in events
        if event.get("schema_version") == runner.S3_3C_METADATA_SCHEMA
    ]
    return {
        "recent": recent,
        "sketch_plan": sketch_plan,
        "sketch_metadata": sketch_metadata,
    }


def validate_report(
    report: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    total_prompts = int(report.get("total_prompts", 0) or 0)
    repeats = int(report.get("repeats", 0) or 0)
    warmup = int(report.get("warmup", 0) or 0)
    expected_measured_runs = total_prompts * repeats
    expected_total_runs = total_prompts * (repeats + warmup) * len(runner.MODE_ORDER)
    event_groups = split_events(events)
    total_raw_events = len(events)
    total_s3_2b_events = len(event_groups["recent"])
    total_s3_3c_plan_events = len(event_groups["sketch_plan"])
    total_s3_3c_metadata_events = len(event_groups["sketch_metadata"])
    ignored_non_s3_events = (
        total_raw_events
        - total_s3_2b_events
        - total_s3_3c_plan_events
        - total_s3_3c_metadata_events
    )

    if total_prompts <= 0:
        errors.append("total_prompts must be > 0")
    if repeats <= 0:
        errors.append("repeats must be > 0")
    if report.get("modes") != runner.MODE_ORDER:
        errors.append("modes must include baseline, recent-window, sketch-active")

    per_mode = report.get("per_mode")
    if not isinstance(per_mode, dict):
        errors.append("per_mode must be a mapping")
        per_mode = {}

    for field in [
        "memory_claim_allowed",
        "quality_claim_allowed",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")

    baseline = per_mode.get(runner.BASELINE_MODE, {})
    recent = per_mode.get(runner.RECENT_WINDOW_MODE, {})
    sketch = per_mode.get(runner.SKETCH_MODE, {})

    for mode_name, mode_report in [
        (runner.BASELINE_MODE, baseline),
        (runner.RECENT_WINDOW_MODE, recent),
        (runner.SKETCH_MODE, sketch),
    ]:
        if not isinstance(mode_report, dict):
            errors.append(f"{mode_name} summary missing")
            continue
        if mode_report.get("success_count") != expected_measured_runs:
            errors.append(
                f"{mode_name} success_count must equal measured run count"
            )

    if report.get("baseline_success_count") != expected_measured_runs:
        errors.append("baseline_success_count must equal measured run count")
    if report.get("recent_window_success_count") != expected_measured_runs:
        errors.append("recent_window_success_count must equal measured run count")
    if report.get("sketch_active_success_count") != expected_measured_runs:
        errors.append("sketch_active_success_count must equal measured run count")

    if report.get("passed") is not True:
        errors.append("passed must be true")

    if total_raw_events <= 0:
        errors.append("total_raw_events must be > 0")
    if total_s3_2b_events <= 0:
        errors.append("total_s3_2b_events must be > 0")
    if total_s3_3c_plan_events <= 0:
        errors.append("total_s3_3c_sketch_plan_events must be > 0")
    if total_s3_3c_metadata_events <= 0:
        errors.append("total_s3_3c_metadata_alias_events must be > 0")

    if total_raw_events != int(report.get("total_raw_events", total_raw_events) or 0):
        errors.append("total_raw_events must match events JSONL")
    if total_s3_2b_events != int(report.get("total_s3_2b_events", total_s3_2b_events) or 0):
        errors.append("total_s3_2b_events must match events JSONL")
    if total_s3_3c_plan_events != int(
        report.get("total_s3_3c_sketch_plan_events", total_s3_3c_plan_events) or 0
    ):
        errors.append("total_s3_3c_sketch_plan_events must match events JSONL")
    if total_s3_3c_metadata_events != int(
        report.get("total_s3_3c_metadata_alias_events", total_s3_3c_metadata_events)
        or 0
    ):
        errors.append("total_s3_3c_metadata_alias_events must match events JSONL")

    if ignored_non_s3_events != int(
        report.get("ignored_non_s3_events", ignored_non_s3_events) or 0
    ):
        errors.append("ignored_non_s3_events must match events JSONL")

    recent_summary = recent.get("event_summary", {})
    sketch_summary = sketch.get("event_summary", {})
    if not isinstance(recent_summary, dict):
        errors.append("recent event_summary missing")
        recent_summary = {}
    if not isinstance(sketch_summary, dict):
        errors.append("sketch event_summary missing")
        sketch_summary = {}

    if (
        recent_summary.get("total_s3_2b_events", 0) <= 0
        and not recent_summary.get("blocker_reasons")
    ):
        errors.append("recent-window mode must have events or blockers")
    if (
        sketch_summary.get("total_s3_3c_sketch_plan_events", 0) <= 0
        and not sketch_summary.get("blocker_reasons")
    ):
        errors.append("sketch-active mode must have plan events or blockers")
    if (
        sketch_summary.get("total_s3_3c_metadata_alias_events", 0) <= 0
        and not sketch_summary.get("blocker_reasons")
    ):
        errors.append("sketch-active mode must have metadata alias events or blockers")

    if report.get("measured_runtime_reduction") is not False:
        errors.append("measured_runtime_reduction must be false")

    if report.get("baseline_to_recent_window_latency_ratio") is None:
        errors.append("baseline_to_recent_window_latency_ratio must be present")
    if report.get("baseline_to_sketch_active_latency_ratio") is None:
        errors.append("baseline_to_sketch_active_latency_ratio must be present")
    if report.get("recent_window_to_sketch_active_latency_ratio") is None:
        errors.append("recent_window_to_sketch_active_latency_ratio must be present")

    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "repeats": repeats,
        "warmup": warmup,
        "total_raw_events": total_raw_events,
        "total_s3_2b_events": total_s3_2b_events,
        "total_s3_3c_sketch_plan_events": total_s3_3c_plan_events,
        "total_s3_3c_metadata_alias_events": total_s3_3c_metadata_events,
        "ignored_non_s3_events": ignored_non_s3_events,
        "expected_measured_runs": expected_measured_runs,
        "expected_total_runs": expected_total_runs,
        "baseline_success_count": report.get("baseline_success_count"),
        "recent_window_success_count": report.get("recent_window_success_count"),
        "sketch_active_success_count": report.get("sketch_active_success_count"),
        "baseline_mean_latency_seconds": baseline.get("mean_latency_seconds"),
        "recent_window_mean_latency_seconds": recent.get("mean_latency_seconds"),
        "sketch_active_mean_latency_seconds": sketch.get("mean_latency_seconds"),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S4.0 Quick Measurement Validation",
        "",
        f"- Passed: `{result['validation_passed']}`",
        f"- Total prompts: `{result['total_prompts']}`",
        f"- Repeats: `{result['repeats']}`",
        f"- Warmup: `{result['warmup']}`",
        f"- Total raw events: `{result['total_raw_events']}`",
        f"- Total S3.2B events: `{result['total_s3_2b_events']}`",
        f"- Total S3.3C plan events: `{result['total_s3_3c_sketch_plan_events']}`",
        (
            "- Total S3.3C metadata alias events: "
            f"`{result['total_s3_3c_metadata_alias_events']}`"
        ),
        f"- Ignored non-S3 events: `{result['ignored_non_s3_events']}`",
        f"- Expected measured runs: `{result['expected_measured_runs']}`",
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
            "This validation only confirms the quick measurement bookkeeping and observed event counts.",
            "It does not claim memory savings, quality preservation, or final selected-attention behavior.",
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
        result = validate_report(load_report(args.input_json), load_events(args.events_jsonl))
    except Exception as exc:
        result = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
            "repeats": 0,
            "warmup": 0,
            "total_raw_events": 0,
            "total_s3_2b_events": 0,
            "total_s3_3c_sketch_plan_events": 0,
            "total_s3_3c_metadata_alias_events": 0,
            "ignored_non_s3_events": 0,
            "expected_measured_runs": 0,
            "expected_total_runs": 0,
            "baseline_success_count": 0,
            "recent_window_success_count": 0,
            "sketch_active_success_count": 0,
            "baseline_mean_latency_seconds": None,
            "recent_window_mean_latency_seconds": None,
            "sketch_active_mean_latency_seconds": None,
        }
    _write(args.output_json, json.dumps(result, indent=2) + "\n")
    _write(args.output_md, render_markdown(result))
    print(
        json.dumps(
            {
                "validation_passed": result["validation_passed"],
                "total_raw_events": result["total_raw_events"],
                "total_s3_2b_events": result["total_s3_2b_events"],
                "total_s3_3c_sketch_plan_events": result[
                    "total_s3_3c_sketch_plan_events"
                ],
                "total_s3_3c_metadata_alias_events": result[
                    "total_s3_3c_metadata_alias_events"
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
