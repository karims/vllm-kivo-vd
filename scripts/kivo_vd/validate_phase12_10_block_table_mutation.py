#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate Phase 12.10 BlockTable mutation observation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "phase12_10_block_table_slot_mapping_v1"
REQUIRED_FIELDS = (
    "schema_version",
    "timestamp",
    "pid",
    "hook_name",
    "module_file",
    "class_name",
    "function_name",
    "self_type",
    "args_summary",
    "kwargs_keys",
    "result_type",
    "result_summary",
    "slot_like_result_found",
    "block_like_result_found",
    "tensor_like_result_found",
    "python_mutable_result_found",
    "mutation_attempted",
    "mutation_applied",
    "mutation_policy",
    "blocker_reason",
    "runtime_behavior_changed",
    "active_routing",
    "measured_runtime_reduction",
    "caveats",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase 12.10 BlockTable mutation JSONL."
    )
    parser.add_argument("--baseline-input", required=True)
    parser.add_argument("--active-input", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/phase12_10_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/phase12_10_validation.md",
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


def validate_record(
    record: dict[str, Any],
    *,
    active_expected: bool,
) -> list[str]:
    errors = []
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if record.get("schema_version") != SCHEMA:
        errors.append("unsupported schema_version")
    if record.get("measured_runtime_reduction") is not False:
        errors.append("measured_runtime_reduction must be false")
    attempted = record.get("mutation_attempted") is True
    applied = record.get("mutation_applied") is True
    if not active_expected and attempted:
        errors.append("baseline records must not attempt mutation")
    if applied:
        if not attempted:
            errors.append("applied mutation requires attempted=true")
        if record.get("active_routing") is not True:
            errors.append("applied mutation requires active_routing=true")
        if record.get("runtime_behavior_changed") is not True:
            errors.append(
                "applied mutation requires runtime_behavior_changed=true"
            )
        if not record.get("mutation_policy"):
            errors.append("applied mutation requires mutation_policy")
    else:
        if active_expected and not record.get("blocker_reason"):
            errors.append("blocked active mutation requires blocker_reason")
    return errors


def validate_records(
    baseline_records: list[dict[str, Any]],
    active_records: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = []
    for index, record in enumerate(baseline_records):
        for message in validate_record(record, active_expected=False):
            errors.append(f"baseline record {index}: {message}")
    for index, record in enumerate(active_records):
        for message in validate_record(record, active_expected=True):
            errors.append(f"active record {index}: {message}")
    return {
        "validation_passed": not errors,
        "baseline_records": len(baseline_records),
        "active_records": len(active_records),
        "mutation_applied_records": sum(
            record.get("mutation_applied") is True
            for record in active_records
        ),
        "tensor_like_blocker_records": sum(
            record.get("tensor_like_result_found") is True
            and record.get("mutation_applied") is not True
            for record in active_records
        ),
        "errors": errors,
        "measured_runtime_reduction": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 12.10 BlockTable Mutation Validation",
        "",
        f"- Passed: `{report['validation_passed']}`",
        f"- Baseline records: `{report['baseline_records']}`",
        f"- Active records: `{report['active_records']}`",
        (
            "- Mutation applied records: "
            f"`{report['mutation_applied_records']}`"
        ),
        (
            "- Tensor-like blocker records: "
            f"`{report['tensor_like_blocker_records']}`"
        ),
        "- Measured runtime reduction: `false`",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {item}" for item in report["errors"])
    if not report["errors"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = validate_records(
            load_records(args.baseline_input),
            load_records(args.active_input),
        )
    except Exception as exc:
        report = {
            "validation_passed": False,
            "baseline_records": 0,
            "active_records": 0,
            "mutation_applied_records": 0,
            "tensor_like_blocker_records": 0,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "measured_runtime_reduction": False,
        }
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "validation_passed": report["validation_passed"],
        "baseline_records": report["baseline_records"],
        "active_records": report["active_records"],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
