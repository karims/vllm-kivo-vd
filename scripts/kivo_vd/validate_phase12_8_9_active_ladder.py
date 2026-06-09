#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate Phase 12.8/12.9 active-ladder observation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "phase12_8_9_active_ladder_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate active-ladder baseline and mutation JSONL."
    )
    parser.add_argument("--baseline-input", required=True)
    parser.add_argument("--metadata-input", required=True)
    parser.add_argument("--selected-slot-input")
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/phase12_8_9_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/phase12_8_9_validation.md",
    )
    return parser.parse_args(argv)


def load_records(
    path: str | Path,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        if required:
            raise FileNotFoundError(f"observation input is missing: {input_path}")
        return []
    records = []
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
        records.append(value)
    if required and not records:
        raise ValueError(f"observation input is empty: {input_path}")
    return records


def validate_stage(
    records: list[dict[str, Any]],
    expected_stage: str,
) -> dict[str, Any]:
    errors = []
    warnings = []
    for index, record in enumerate(records):
        if record.get("schema_version") != SCHEMA:
            errors.append(f"record {index}: unsupported schema")
        stage = str(record.get("mutation_stage", ""))
        if expected_stage == "metadata":
            if stage not in ("metadata", "metadata_drop_one_key"):
                errors.append(f"record {index}: unexpected metadata stage")
        elif expected_stage == "selected_slot":
            if stage not in ("selected_slot", "selected_slot_drop_one"):
                errors.append(f"record {index}: unexpected selected-slot stage")
        elif stage != "baseline":
            errors.append(f"record {index}: unexpected baseline stage")
        attempted = record.get("mutation_attempted")
        applied = record.get("mutation_applied")
        if applied is True and attempted is not True:
            errors.append(f"record {index}: applied mutation was not attempted")
        if record.get("measured_runtime_reduction") is not False:
            errors.append(f"record {index}: measured reduction must be false")
        if expected_stage == "selected_slot":
            if attempted is True and applied is False:
                if not record.get("blocker_reason"):
                    errors.append(
                        f"record {index}: blocked selected slot needs reason"
                    )
        if expected_stage == "baseline" and (attempted or applied):
            errors.append(f"record {index}: baseline must not mutate")
    if not records and expected_stage == "selected_slot":
        warnings.append("selected-slot stage was skipped or produced no records")
    return {
        "record_count": len(records),
        "mutation_attempted": any(
            record.get("mutation_attempted") is True for record in records
        ),
        "mutation_applied": any(
            record.get("mutation_applied") is True for record in records
        ),
        "errors": errors,
        "warnings": warnings,
    }


def validate_ladder(
    baseline: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    selected_slot: list[dict[str, Any]],
) -> dict[str, Any]:
    stages = {
        "baseline": validate_stage(baseline, "baseline"),
        "metadata": validate_stage(metadata, "metadata"),
        "selected_slot": validate_stage(selected_slot, "selected_slot"),
    }
    errors = [
        f"{stage}: {message}"
        for stage, result in stages.items()
        for message in result["errors"]
    ]
    warnings = [
        f"{stage}: {message}"
        for stage, result in stages.items()
        for message in result["warnings"]
    ]
    selected_applied = stages["selected_slot"]["mutation_applied"]
    return {
        "validation_passed": not errors,
        "stages": stages,
        "errors": errors,
        "warnings": warnings,
        "active_routing": selected_applied,
        "runtime_behavior_changed": bool(
            stages["metadata"]["mutation_applied"] or selected_applied
        ),
        "measured_runtime_reduction": False,
        "production_selected_attention_claim": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 12.8/12.9 Active Ladder Validation",
        "",
        f"- Passed: `{report['validation_passed']}`",
        f"- Active routing attempted: `{report['active_routing']}`",
        f"- Runtime behavior changed: `{report['runtime_behavior_changed']}`",
        "- Measured runtime reduction: `false`",
        "- Production selected-attention claim: `false`",
        "",
        "## Stages",
        "",
        "| stage | records | attempted | applied |",
        "| --- | ---: | --- | --- |",
    ]
    for stage, result in report["stages"].items():
        lines.append(
            f"| {stage} | {result['record_count']} | "
            f"{result['mutation_attempted']} | {result['mutation_applied']} |"
        )
    for title, key in (("Errors", "errors"), ("Warnings", "warnings")):
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {item}" for item in report[key])
        if not report[key]:
            lines.append("- none")
    lines.extend([
        "",
        "This validates experiment records only. It does not establish safe",
        "selected attention, quality, latency, or memory reduction.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = validate_ladder(
            load_records(args.baseline_input, required=True),
            load_records(args.metadata_input, required=True),
            load_records(args.selected_slot_input, required=False)
            if args.selected_slot_input
            else [],
        )
    except Exception as exc:
        report = {
            "validation_passed": False,
            "stages": {},
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
            "active_routing": False,
            "runtime_behavior_changed": False,
            "measured_runtime_reduction": False,
            "production_selected_attention_claim": False,
        }
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "validation_passed": report["validation_passed"],
        "active_routing": report["active_routing"],
        "runtime_behavior_changed": report["runtime_behavior_changed"],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
