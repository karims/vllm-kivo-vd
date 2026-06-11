#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate Phase S3.3C active sketch metadata alias output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "kivo_source_s3_3c_active_sketch_plan_v1"
METADATA_SCHEMA = "kivo_source_s3_3c_active_sketch_metadata_alias_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S3.3C active sketch metadata aliasing."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s3_3c_active_sketch_kv_metadata_alias_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s3_3c_active_sketch_kv_metadata_alias_validation.md"
        ),
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.3C input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def load_events(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.3C events input is missing: {input_path}")
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


def split_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    plan_events = [
        event for event in events if event.get("schema_version") == PLAN_SCHEMA
    ]
    metadata_events = [
        event for event in events if event.get("schema_version") == METADATA_SCHEMA
    ]
    ignored = len(events) - len(plan_events) - len(metadata_events)
    return plan_events, metadata_events, ignored


def validate_events(
    plan_events: list[dict[str, Any]],
    metadata_events: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not any(event.get("sketch_computed") is True for event in plan_events):
        errors.append("sketch_computed_event_count must be > 0")
    if not any(event.get("sketch_plan_used") is True for event in metadata_events):
        errors.append("sketch_plan_used_event_count must be > 0")
    if not any(event.get("mutation_attempted") is True for event in metadata_events):
        errors.append("mutation_attempted_event_count must be > 0")
    if not any(event.get("mutation_applied") is True for event in metadata_events):
        errors.append("mutation_applied_event_count must be > 0")
    if not any(event.get("active_routing") is True for event in metadata_events):
        errors.append("active_routing_event_count must be > 0")

    for index, event in enumerate(plan_events + metadata_events):
        for field in [
            "measured_runtime_reduction",
            "selected_attention_claim_allowed",
            "performance_claim_allowed",
        ]:
            if event.get(field) is not False:
                errors.append(f"event {index} {field} must be false")

    for index, event in enumerate(metadata_events):
        if (
            event.get("mutation_applied") is True
            and event.get("sketch_plan_used") is not True
        ):
            errors.append(
                f"metadata event {index} mutation_applied requires sketch_plan_used"
            )
        if (
            event.get("mutation_applied") is True
            and not event.get("alias_pairs_sample")
        ):
            errors.append(
                f"metadata event {index} mutation_applied requires alias_pairs_sample"
            )
        alias_target = event.get("alias_target_block_id")
        if event.get("mutation_applied") is True and (
            alias_target is None or int(alias_target) < 0
        ):
            errors.append(f"metadata event {index} invalid alias target")
    return errors


def validate_active_sketch_alias(
    report: dict[str, Any],
    raw_events: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    plan_events, metadata_events, ignored = split_events(raw_events)
    total_prompts = int(report.get("total_prompts", 0) or 0)
    if total_prompts <= 0:
        errors.append("total_prompts must be > 0")
    if report.get("baseline_success_count") != total_prompts:
        errors.append("baseline_success_count must equal total_prompts")
    if report.get("active_success_count") != total_prompts:
        errors.append("active_success_count must equal total_prompts")
    if len(plan_events) <= 0:
        errors.append("sketch plan events must be > 0")
    if len(metadata_events) <= 0:
        errors.append("metadata alias events must be > 0")
    prompt_results = report.get("prompt_results")
    if not isinstance(prompt_results, list) or len(prompt_results) != total_prompts:
        errors.append("prompt_results must contain one entry per prompt")
    else:
        for index, item in enumerate(prompt_results):
            if item.get("baseline_status") != "succeeded":
                errors.append(f"prompt result {index} baseline must succeed")
            if item.get("active_status") != "succeeded":
                errors.append(f"prompt result {index} active must succeed")
    for field in [
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")
    errors.extend(validate_events(plan_events, metadata_events))
    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "total_raw_events": len(raw_events),
        "total_s3_3c_sketch_plan_events": len(plan_events),
        "total_s3_3c_metadata_alias_events": len(metadata_events),
        "ignored_non_s3_events": ignored,
        "sketch_computed_event_count": sum(
            event.get("sketch_computed") is True for event in plan_events
        ),
        "sketch_plan_used_event_count": sum(
            event.get("sketch_plan_used") is True for event in metadata_events
        ),
        "mutation_applied_event_count": sum(
            event.get("mutation_applied") is True for event in metadata_events
        ),
        "active_routing_event_count": sum(
            event.get("active_routing") is True for event in metadata_events
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.3C Active Sketch Metadata Alias Validation",
        "",
        f"- Passed: `{result['validation_passed']}`",
        f"- Total prompts: `{result['total_prompts']}`",
        (
            "- Sketch plan events: "
            f"`{result['total_s3_3c_sketch_plan_events']}`"
        ),
        (
            "- Metadata alias events: "
            f"`{result['total_s3_3c_metadata_alias_events']}`"
        ),
        (
            "- Mutation applied events: "
            f"`{result['mutation_applied_event_count']}`"
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
            "This validation confirms active sketch-driven cloned metadata aliasing "
            "only. It does not support memory, latency, quality, or final "
            "selected-attention claims.",
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
        result = validate_active_sketch_alias(
            load_report(args.input_json),
            load_events(args.events_jsonl),
        )
    except Exception as exc:
        result = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
            "total_raw_events": 0,
            "total_s3_3c_sketch_plan_events": 0,
            "total_s3_3c_metadata_alias_events": 0,
            "ignored_non_s3_events": 0,
            "sketch_computed_event_count": 0,
            "sketch_plan_used_event_count": 0,
            "mutation_applied_event_count": 0,
            "active_routing_event_count": 0,
        }
    _write(args.output_json, json.dumps(result, indent=2) + "\n")
    _write(args.output_md, render_markdown(result))
    print(
        json.dumps(
            {
                "validation_passed": result["validation_passed"],
                "total_raw_events": result["total_raw_events"],
                "total_s3_3c_sketch_plan_events": (
                    result["total_s3_3c_sketch_plan_events"]
                ),
                "total_s3_3c_metadata_alias_events": (
                    result["total_s3_3c_metadata_alias_events"]
                ),
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
