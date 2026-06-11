#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S3.0B attention metadata observer report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "kivo_source_s3_0b_attention_metadata_observer_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S3.0B attention metadata observer JSON."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer_validation.md",
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.0B input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def load_events(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S3.0B events input is missing: {input_path}")
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


def _true_claims(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            key_lower = key.lower()
            prohibited = (
                key == "measured_runtime_reduction"
                or (
                    "memory" in key_lower
                    and any(
                        term in key_lower
                        for term in ("reduction", "saving", "improvement")
                    )
                )
                or (
                    "latency" in key_lower
                    and any(
                        term in key_lower
                        for term in ("reduction", "improvement")
                    )
                )
                or "selected_attention_claim" in key_lower
                or "performance_claim" in key_lower
            )
            if prohibited and child is True:
                found.append(child_path)
            found.extend(_true_claims(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_true_claims(child, f"{path}[{index}]"))
    return found


def validate_events(events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, event in enumerate(events):
        missing = [
            field
            for field in [
                "schema_version",
                "policy_name",
                "hook_point",
                "mutation_attempted",
                "mutation_applied",
                "active_routing",
                "runtime_behavior_changed",
                "measured_runtime_reduction",
                "selected_attention_claim_allowed",
                "performance_claim_allowed",
                "selected_block_count",
            ]
            if field not in event
        ]
        if missing:
            errors.append(f"event {index} missing fields: {', '.join(missing)}")
        if event.get("schema_version") != SCHEMA:
            errors.append(f"event {index} has unsupported schema_version")
        if event.get("policy_name") != "observe_attention_metadata":
            errors.append(f"event {index} has unexpected policy_name")
        if event.get("mutation_attempted") is not False:
            errors.append(f"event {index} must not attempt mutation")
        if event.get("mutation_applied") is not False:
            errors.append(f"event {index} must not apply mutation")
        if event.get("selected_block_count") is not None:
            errors.append(f"event {index} selected_block_count must be null")
        if event.get("active_routing") is not False:
            errors.append(f"event {index} must not claim active routing")
        if event.get("runtime_behavior_changed") is not False:
            errors.append(
                f"event {index} must not claim runtime behavior change"
            )
        if event.get("measured_runtime_reduction") is not False:
            errors.append(f"event {index} measured_runtime_reduction must be false")
        if event.get("selected_attention_claim_allowed") is not False:
            errors.append(
                f"event {index} selected_attention_claim_allowed must be false"
            )
        if event.get("performance_claim_allowed") is not False:
            errors.append(f"event {index} performance_claim_allowed must be false")
        claim_paths = sorted(set(_true_claims(event)))
        if claim_paths:
            errors.append(
                f"event {index} contains prohibited true claims: "
                f"{', '.join(claim_paths)}"
            )
    return errors


def validate_observer(
    report: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    required = [
        "total_prompts",
        "baseline_success_count",
        "observer_success_count",
        "output_changed_count",
        "total_events",
        "metadata_observed_prompt_count",
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
        "s3_0b_observer_passed",
        "prompt_results",
    ]
    missing = [field for field in required if field not in report]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    total_prompts = int(report.get("total_prompts", 0) or 0)
    if total_prompts <= 0:
        errors.append("total_prompts must be > 0")
    if report.get("baseline_success_count") != total_prompts:
        errors.append("baseline_success_count must equal total_prompts")
    if report.get("observer_success_count") != total_prompts:
        errors.append("observer_success_count must equal total_prompts")
    if int(report.get("output_changed_count", 0) or 0) != 0:
        errors.append("output_changed_count must be 0")
    if int(report.get("total_events", 0) or 0) <= 0:
        errors.append("total_events must be > 0")
    if int(report.get("metadata_observed_prompt_count", 0) or 0) <= 0:
        errors.append("metadata_observed_prompt_count must be > 0")
    for field in [
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")
    if report.get("s3_0b_observer_passed") is not True:
        errors.append("s3_0b_observer_passed must be true")
    if not isinstance(report.get("prompt_results"), list):
        errors.append("prompt_results must be a list")
    elif len(report["prompt_results"]) != total_prompts:
        errors.append("prompt_results must contain one entry per prompt")

    for index, item in enumerate(report.get("prompt_results", [])):
        for field in [
            "prompt_index",
            "prompt",
            "baseline_status",
            "observer_status",
            "baseline_output",
            "observer_output",
            "baseline_error",
            "observer_error",
            "output_changed",
            "records_written",
            "max_block_table_rows",
            "max_block_table_cols",
            "max_slot_mapping_len",
        ]:
            if field not in item:
                errors.append(f"prompt result {index} missing field {field}")
        if item.get("baseline_status") != "succeeded":
            errors.append(f"prompt result {index} baseline must succeed")
        if item.get("observer_status") != "succeeded":
            errors.append(f"prompt result {index} observer must succeed")
        if item.get("output_changed") is not False:
            errors.append(f"prompt result {index} output_changed must be false")
        if int(item.get("records_written", 0) or 0) <= 0:
            errors.append(f"prompt result {index} must observe at least one event")

    event_errors = validate_events(events)
    errors.extend(event_errors)
    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "total_events": len(events),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.0B Attention Metadata Observer Validation",
        "",
        f"- Passed: `{report['validation_passed']}`",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Total events: `{report['total_events']}`",
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
            "This validation confirms metadata visibility only. It does not "
            "mutate attention, block tables, slot mappings, KV cache state, "
            "or model outputs.",
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
        report = validate_observer(
            load_report(args.input_json),
            load_events(args.events_jsonl),
        )
    except Exception as exc:
        report = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
            "total_events": 0,
        }
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "validation_passed": report["validation_passed"],
                "total_prompts": report["total_prompts"],
                "total_events": report["total_events"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
