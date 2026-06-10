#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate source-level Phase S1 observation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "kivo_source_s1_block_table_v1"
REQUIRED_FIELDS = (
    "schema_version",
    "timestamp",
    "pid",
    "hook_name",
    "class_name",
    "function_name",
    "args_summary",
    "slot_mapping_present",
    "slot_mapping_type",
    "slot_mapping_shape",
    "slot_mapping_dtype",
    "slot_mapping_device",
    "block_table_present",
    "block_table_type",
    "block_table_shape",
    "block_table_dtype",
    "block_table_device",
    "block_size",
    "num_blocks_per_row",
    "max_num_blocks_per_req",
    "max_num_reqs",
    "active_enabled",
    "mutation_attempted",
    "mutation_applied",
    "mutation_policy",
    "mutation_blocker_reason",
    "valid_slot_count",
    "pad_slot_id",
    "valid_mutation_index",
    "previous_valid_index",
    "old_new_differ",
    "runtime_behavior_changed",
    "active_routing",
    "measured_runtime_reduction",
    "caveats",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S1 source-level observation JSONL."
    )
    parser.add_argument("--baseline-input", required=True)
    parser.add_argument("--observation-input", required=True)
    parser.add_argument("--active-input", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s1_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s1_validation.md",
    )
    return parser.parse_args(argv)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"observation input is missing: {input_path}")
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
    if not records:
        raise ValueError(f"observation input is empty: {input_path}")
    return records


def validate_record(record: dict[str, Any], *, active_expected: bool) -> list[str]:
    errors = []
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if record.get("schema_version") != SCHEMA:
        errors.append("unsupported schema_version")
    if record.get("measured_runtime_reduction") is not False:
        errors.append("measured_runtime_reduction must be false")
    if active_expected:
        if record.get("active_enabled") is not True:
            errors.append("active observation must have active_enabled=true")
        if record.get("mutation_applied") is True:
            if record.get("mutation_attempted") is not True:
                errors.append("applied mutation requires attempted=true")
            if record.get("active_routing") is not True:
                errors.append("applied mutation requires active_routing=true")
            if record.get("runtime_behavior_changed") is not True:
                errors.append(
                    "applied mutation requires runtime_behavior_changed=true"
                )
            if record.get("valid_slot_count", 0) < 2:
                errors.append("applied mutation requires valid_slot_count>=2")
            if record.get("old_new_differ") is not True:
                errors.append("applied mutation requires old_new_differ=true")
            if record.get("old_value") is None:
                errors.append("applied mutation requires old_value")
            if record.get("new_value") is None:
                errors.append("applied mutation requires new_value")
            if record.get("mutation_index") is None:
                errors.append("applied mutation requires mutation_index")
            if record.get("valid_mutation_index") is None:
                errors.append("applied mutation requires valid_mutation_index")
            if record.get("previous_valid_index") is None:
                errors.append("applied mutation requires previous_valid_index")
        else:
            if not record.get("mutation_blocker_reason"):
                errors.append("blocked active observation requires blocker reason")
    else:
        if record.get("mutation_attempted") is True:
            errors.append("baseline/observation records must not attempt mutation")
        if record.get("mutation_applied") is True:
            errors.append("baseline/observation records must not apply mutation")
        if record.get("active_routing") is True:
            errors.append("baseline/observation records must not route actively")
    return errors


def validate_observations(
    baseline_records: list[dict[str, Any]],
    observation_records: list[dict[str, Any]],
    active_records: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = []
    for index, record in enumerate(baseline_records):
        for message in validate_record(record, active_expected=False):
            errors.append(f"baseline record {index}: {message}")
    for index, record in enumerate(observation_records):
        for message in validate_record(record, active_expected=False):
            errors.append(f"observation record {index}: {message}")
    for index, record in enumerate(active_records):
        for message in validate_record(record, active_expected=True):
            errors.append(f"active record {index}: {message}")
    return {
        "validation_passed": not errors,
        "baseline_records": len(baseline_records),
        "observation_records": len(observation_records),
        "active_records": len(active_records),
        "mutation_applied_records": sum(
            record.get("mutation_applied") is True for record in active_records
        ),
        "errors": errors,
        "measured_runtime_reduction": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Kivo-VD Phase S1 Source-Level Observation Validation",
        "",
        f"- Passed: `{report['validation_passed']}`",
        f"- Baseline records: `{report['baseline_records']}`",
        f"- Observation records: `{report['observation_records']}`",
        f"- Active records: `{report['active_records']}`",
        (
            "- Mutation applied records: "
            f"`{report['mutation_applied_records']}`"
        ),
        "- Measured runtime reduction: `false`",
        "",
        "## Errors",
        "",
        *(f"- {item}" for item in report["errors"]),
    ]) + ("\n" if report["errors"] else "- none\n")


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = validate_observations(
            load_records(args.baseline_input),
            load_records(args.observation_input),
            load_records(args.active_input),
        )
    except Exception as exc:
        report = {
            "validation_passed": False,
            "baseline_records": 0,
            "observation_records": 0,
            "active_records": 0,
            "mutation_applied_records": 0,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "measured_runtime_reduction": False,
        }
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "validation_passed": report["validation_passed"],
        "baseline_records": report["baseline_records"],
        "observation_records": report["observation_records"],
        "active_records": report["active_records"],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
