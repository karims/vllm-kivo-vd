#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S3.2B active recent-window attention report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "kivo_source_s3_2b_active_recent_window_attention_metadata_v1"
POLICY_NAME = "active_recent_window_attention_metadata"
ACTIVE_FILTER_MODE = "compact_to_recent_window"
SELECTION_POLICY_NAME = "recent_window_compaction"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S3.2B active recent-window metadata."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata_validation.md",
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.2B input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def load_events(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.2B events input is missing: {input_path}")
    events = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSONL row {line_number}: {input_path}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        events.append(value)
    if not events:
        raise ValueError(f"events input is empty: {input_path}")
    return events


def filter_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    filtered = [event for event in events if event.get("schema_version") == SCHEMA]
    return filtered, len(events) - len(filtered)


def validate_events(events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    saw_compacted_seq_len = False
    saw_positive_reduction = False
    for index, event in enumerate(events):
        missing = [
            field
            for field in [
                "schema_version",
                "policy_name",
                "hook_point",
                "active_filter_mode",
                "selection_policy_name",
                "original_seq_len",
                "modified_seq_len",
                "original_visible_block_count",
                "selected_block_count",
                "excluded_block_count",
                "selected_block_ids_sample",
                "excluded_block_ids_sample",
                "keep_recent_blocks",
                "selected_token_length",
                "theoretical_attention_visible_block_reduction",
                "theoretical_attention_visible_block_reduction_ratio",
                "mutation_attempted",
                "mutation_applied",
                "mutation_blocker_reason",
                "active_routing",
                "runtime_behavior_changed",
                "measured_runtime_reduction",
                "selected_attention_claim_allowed",
                "performance_claim_allowed",
            ]
            if field not in event
        ]
        if missing:
            errors.append(f"event {index} missing fields: {', '.join(missing)}")
        if event.get("policy_name") != POLICY_NAME:
            errors.append(f"event {index} has unexpected policy_name")
        if event.get("active_filter_mode") != ACTIVE_FILTER_MODE:
            errors.append(f"event {index} has unexpected active_filter_mode")
        if event.get("selection_policy_name") != SELECTION_POLICY_NAME:
            errors.append(f"event {index} has unexpected selection_policy_name")
        if event.get("measured_runtime_reduction") is not False:
            errors.append(f"event {index} measured_runtime_reduction must be false")
        if event.get("selected_attention_claim_allowed") is not False:
            errors.append(
                f"event {index} selected_attention_claim_allowed must be false"
            )
        if event.get("performance_claim_allowed") is not False:
            errors.append(f"event {index} performance_claim_allowed must be false")
        mutation_applied = event.get("mutation_applied") is True
        if mutation_applied:
            if event.get("active_routing") is not True:
                errors.append(
                    f"event {index} applied mutation without active_routing"
                )
            if event.get("mutation_blocker_reason") is not None:
                errors.append(
                    f"event {index} reports blocker after mutation_applied"
                )
            original_seq_len = event.get("original_seq_len")
            modified_seq_len = event.get("modified_seq_len")
            if not isinstance(original_seq_len, int) or not isinstance(
                modified_seq_len, int
            ):
                errors.append(
                    f"event {index} missing valid original/modified seq len"
                )
            elif modified_seq_len < original_seq_len:
                saw_compacted_seq_len = True
            if int(event.get("theoretical_attention_visible_block_reduction", 0) or 0) > 0:
                saw_positive_reduction = True
        elif event.get("mutation_attempted") is True and not event.get(
            "mutation_blocker_reason"
        ):
            errors.append(
                f"event {index} attempted mutation without result or blocker"
            )
    if not saw_compacted_seq_len:
        errors.append(
            "at least one applied event must have modified_seq_len < original_seq_len"
        )
    if not saw_positive_reduction:
        errors.append(
            "at least one applied event must report positive theoretical reduction"
        )
    return errors


def validate_recent_window(
    report: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    s3_events, ignored_non_s3_events = filter_events(events)
    total_prompts = int(report.get("total_prompts", 0) or 0)
    required = [
        "total_prompts",
        "baseline_success_count",
        "active_success_count",
        "mutation_attempted_event_count",
        "mutation_applied_event_count",
        "active_routing_event_count",
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
        "s3_2b_active_recent_window_passed",
        "prompt_results",
    ]
    missing = [field for field in required if field not in report]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if total_prompts <= 0:
        errors.append("total_prompts must be > 0")
    if report.get("baseline_success_count") != total_prompts:
        errors.append("baseline_success_count must equal total_prompts")
    if report.get("active_success_count") != total_prompts:
        errors.append("active_success_count must equal total_prompts")
    if not s3_events:
        errors.append("total_s3_2b_events must be > 0")
    if int(report.get("mutation_attempted_event_count", 0) or 0) <= 0:
        errors.append("mutation_attempted_event_count must be > 0")
    if int(report.get("mutation_applied_event_count", 0) or 0) <= 0:
        errors.append("mutation_applied_event_count must be > 0")
    if int(report.get("active_routing_event_count", 0) or 0) <= 0:
        errors.append("active_routing_event_count must be > 0")
    for field in [
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")
    if report.get("s3_2b_active_recent_window_passed") is not True:
        errors.append("s3_2b_active_recent_window_passed must be true")
    prompt_results = report.get("prompt_results")
    if not isinstance(prompt_results, list):
        errors.append("prompt_results must be a list")
    elif len(prompt_results) != total_prompts:
        errors.append("prompt_results must contain one entry per prompt")
    else:
        for index, item in enumerate(prompt_results):
            if item.get("baseline_status") != "succeeded":
                errors.append(f"prompt result {index} baseline must succeed")
            if item.get("active_status") != "succeeded":
                errors.append(f"prompt result {index} active must succeed")

    errors.extend(validate_events(s3_events))
    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "total_raw_events": len(events),
        "total_s3_2b_events": len(s3_events),
        "ignored_non_s3_events": ignored_non_s3_events,
        "output_changed_count": int(
            report.get("output_changed_count", 0) or 0
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.2B Active Recent-Window Validation",
        "",
        f"- Passed: `{report['validation_passed']}`",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Total raw events: `{report['total_raw_events']}`",
        f"- Total S3.2B events: `{report['total_s3_2b_events']}`",
        f"- Ignored non-S3 events: `{report['ignored_non_s3_events']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This validates active recent-window metadata compaction only. It "
            "does not prove KV memory reduction, latency improvement, or "
            "quality preservation.",
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
        report = validate_recent_window(
            load_report(args.input_json),
            load_events(args.events_jsonl),
        )
    except Exception as exc:
        report = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
            "total_raw_events": 0,
            "total_s3_2b_events": 0,
            "ignored_non_s3_events": 0,
            "output_changed_count": 0,
        }
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "validation_passed": report["validation_passed"],
                "total_prompts": report["total_prompts"],
                "total_s3_2b_events": report["total_s3_2b_events"],
                "output_changed_count": report["output_changed_count"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
