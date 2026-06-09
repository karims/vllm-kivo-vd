#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate Phase 12.7 installed-runtime observation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "schema_version",
    "timestamp",
    "pid",
    "hook_name",
    "module_file",
    "function_name",
    "self_type",
    "args_summary",
    "kwargs_keys",
    "result_type",
    "result_summary",
    "metadata_keys_found",
    "block_like_fields_found",
    "slot_like_fields_found",
    "attention_like_fields_found",
    "kv_like_fields_found",
    "active_enabled",
    "mutation_attempted",
    "mutation_applied",
    "runtime_behavior_changed",
    "active_routing",
    "measured_runtime_reduction",
    "caveats",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase 12.7 runtime observation JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_7_runtime_observation_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_7_runtime_observation_validation.md"
        ),
    )
    return parser.parse_args(argv)


def load_observations(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"runtime observation input is missing: {input_path}"
        )
    records: list[dict[str, Any]] = []
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
    return records


def validate_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if record.get("schema_version") != "phase12_7_runtime_observation_v1":
        errors.append("unsupported schema_version")
    for field in (
        "active_enabled",
        "mutation_attempted",
        "mutation_applied",
        "runtime_behavior_changed",
        "active_routing",
        "measured_runtime_reduction",
    ):
        if not isinstance(record.get(field), bool):
            errors.append(f"{field} must be boolean")
    for field in (
        "runtime_behavior_changed",
        "active_routing",
        "measured_runtime_reduction",
    ):
        if record.get(field) is not False:
            errors.append(f"{field} must be false")

    active = record.get("active_enabled") is True
    attempted = record.get("mutation_attempted") is True
    applied = record.get("mutation_applied") is True
    blocked = record.get("active_experiment_blocked")
    blocker = record.get("blocker_reason")
    if not active and attempted:
        errors.append("observation-only mode must not attempt mutation")
    if applied:
        errors.append("Phase 12.7 must not apply runtime mutation")
    if active:
        if attempted is not True:
            warnings.append("active mode did not record a decision attempt")
        if blocked is True and not blocker:
            errors.append("blocked active mode requires blocker_reason")
        if blocked is not True:
            warnings.append("active mode should remain blocked in Phase 12.7")

    return {
        "index": index,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_observations(records: list[dict[str, Any]]) -> dict[str, Any]:
    results = [
        validate_record(record, index)
        for index, record in enumerate(records)
    ]
    errors = [
        {"index": result["index"], "message": message}
        for result in results
        for message in result["errors"]
    ]
    warnings = [
        {"index": result["index"], "message": message}
        for result in results
        for message in result["warnings"]
    ]
    valid = sum(result["valid"] for result in results)
    return {
        "validation_passed": bool(records) and not errors,
        "total_records": len(records),
        "valid_records": valid,
        "invalid_records": len(records) - valid,
        "active_records": sum(
            record.get("active_enabled") is True for record in records
        ),
        "mutation_attempted_records": sum(
            record.get("mutation_attempted") is True for record in records
        ),
        "mutation_applied_records": sum(
            record.get("mutation_applied") is True for record in records
        ),
        "errors": errors,
        "warnings": warnings,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "runtime_behavior_changed": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 12.7 Runtime Observation Validation",
        "",
        f"- Passed: `{str(report['validation_passed']).lower()}`",
        f"- Total records: `{report['total_records']}`",
        f"- Valid records: `{report['valid_records']}`",
        f"- Invalid records: `{report['invalid_records']}`",
        f"- Active records: `{report['active_records']}`",
        (
            "- Mutation attempted records: "
            f"`{report['mutation_attempted_records']}`"
        ),
        (
            "- Mutation applied records: "
            f"`{report['mutation_applied_records']}`"
        ),
        "- Runtime behavior changed: `false`",
        "- Active routing: `false`",
        "- Measured runtime reduction: `false`",
    ]
    for title, key in (("Errors", "errors"), ("Warnings", "warnings")):
        lines.extend(["", f"## {title}", ""])
        if report[key]:
            lines.extend(
                f"- Record `{item['index']}`: {item['message']}"
                for item in report[key]
            )
        else:
            lines.append("- none")
    lines.extend([
        "",
        "## Caveat",
        "",
        "Validation covers side-channel observations only. It does not prove",
        "selected attention, memory reduction, latency, or quality.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        report = validate_observations(load_observations(args.input))
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(json.dumps({
            "validation_passed": report["validation_passed"],
            "total_records": report["total_records"],
            "valid_records": report["valid_records"],
            "invalid_records": report["invalid_records"],
            "output_json": args.output_json,
            "output_md": args.output_md,
        }, separators=(",", ":")))
        return 0 if report["validation_passed"] else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
