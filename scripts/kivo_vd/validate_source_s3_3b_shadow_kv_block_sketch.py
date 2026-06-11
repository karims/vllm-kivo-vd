#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate Phase S3.3B shadow KV block sketch artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "kivo_source_s3_3b_shadow_kv_block_sketch_v1"
POLICY = "shadow_kv_block_sketch"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S3.3B shadow KV block sketch output."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch_validation.md"
        ),
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.3B input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def load_events(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.3B events input is missing: {input_path}")
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


def filter_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    filtered = [event for event in events if event.get("schema_version") == SCHEMA]
    return filtered, len(events) - len(filtered)


def validate_events(events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_version",
        "policy_name",
        "hook_point",
        "sketch_source",
        "sketch_method",
        "candidate_block_count",
        "selected_block_count",
        "excluded_block_count",
        "block_sketch_sample",
        "sketch_computed",
        "mutation_attempted",
        "mutation_applied",
        "active_routing",
        "runtime_behavior_changed",
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]
    for index, event in enumerate(events):
        missing = [field for field in required if field not in event]
        if missing:
            errors.append(f"event {index} missing fields: {', '.join(missing)}")
        if event.get("policy_name") != POLICY:
            errors.append(f"event {index} has unexpected policy_name")
        if event.get("sketch_source") != "kv_cache":
            errors.append(f"event {index} sketch_source must be kv_cache")
        for field in [
            "mutation_attempted",
            "mutation_applied",
            "active_routing",
            "runtime_behavior_changed",
            "measured_runtime_reduction",
            "selected_attention_claim_allowed",
            "performance_claim_allowed",
        ]:
            if event.get(field) is not False:
                errors.append(f"event {index} {field} must be false")
    return errors


def validate_shadow_kv_block_sketch(
    report: dict[str, Any],
    raw_events: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    events, ignored = filter_events(raw_events)
    total_prompts = int(report.get("total_prompts", 0) or 0)
    if total_prompts <= 0:
        errors.append("total_prompts must be > 0")
    if report.get("baseline_success_count") != total_prompts:
        errors.append("baseline_success_count must equal total_prompts")
    if report.get("shadow_success_count") != total_prompts:
        errors.append("shadow_success_count must equal total_prompts")
    if int(report.get("output_changed_count", 0) or 0) != 0:
        errors.append("output_changed_count must be 0")
    if not events:
        errors.append("total_s3_3b_events must be > 0")
    if int(report.get("sketch_computed_event_count", 0) or 0) <= 0:
        errors.append("sketch_computed_event_count must be > 0")
    if events and not any(
        int(event.get("candidate_block_count", 0) or 0) > 0 for event in events
    ):
        errors.append("at least one event must have candidate blocks")
    if events and not any(
        isinstance(event.get("block_sketch_sample"), list)
        and len(event.get("block_sketch_sample", [])) > 0
        for event in events
    ):
        errors.append("at least one event must have block_sketch_sample")
    for field in [
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")
    prompt_results = report.get("prompt_results")
    if not isinstance(prompt_results, list) or len(prompt_results) != total_prompts:
        errors.append("prompt_results must contain one entry per prompt")
    else:
        for index, item in enumerate(prompt_results):
            if item.get("baseline_status") != "succeeded":
                errors.append(f"prompt result {index} baseline must succeed")
            if item.get("shadow_status") != "succeeded":
                errors.append(f"prompt result {index} shadow must succeed")
            if item.get("output_changed") is not False:
                errors.append(f"prompt result {index} output_changed must be false")
    errors.extend(validate_events(events))
    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "total_raw_events": len(raw_events),
        "total_s3_3b_events": len(events),
        "ignored_non_s3_events": ignored,
        "sketch_computed_event_count": sum(
            event.get("sketch_computed") is True for event in events
        ),
        "max_candidate_block_count": max(
            (int(event.get("candidate_block_count", 0) or 0) for event in events),
            default=0,
        ),
        "max_selected_block_count": max(
            (int(event.get("selected_block_count", 0) or 0) for event in events),
            default=0,
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.3B Shadow KV Block Sketch Validation",
        "",
        f"- Passed: `{result['validation_passed']}`",
        f"- Total prompts: `{result['total_prompts']}`",
        f"- Total raw events: `{result['total_raw_events']}`",
        f"- Total S3.3B events: `{result['total_s3_3b_events']}`",
        f"- Ignored non-S3 events: `{result['ignored_non_s3_events']}`",
        (
            "- Sketch computed events: "
            f"`{result['sketch_computed_event_count']}`"
        ),
        (
            "- Max candidate block count: "
            f"`{result['max_candidate_block_count']}`"
        ),
        (
            "- Max selected block count: "
            f"`{result['max_selected_block_count']}`"
        ),
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
            "This validation confirms shadow-mode real KV block sketching only. "
            "It does not mutate runtime state or support memory, latency, "
            "quality, or selected-attention claims.",
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
        result = validate_shadow_kv_block_sketch(
            load_report(args.input_json),
            load_events(args.events_jsonl),
        )
    except Exception as exc:
        result = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
            "total_raw_events": 0,
            "total_s3_3b_events": 0,
            "ignored_non_s3_events": 0,
            "sketch_computed_event_count": 0,
            "max_candidate_block_count": 0,
            "max_selected_block_count": 0,
        }
    _write(args.output_json, json.dumps(result, indent=2) + "\n")
    _write(args.output_md, render_markdown(result))
    print(
        json.dumps(
            {
                "validation_passed": result["validation_passed"],
                "total_raw_events": result["total_raw_events"],
                "total_s3_3b_events": result["total_s3_3b_events"],
                "ignored_non_s3_events": result["ignored_non_s3_events"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
