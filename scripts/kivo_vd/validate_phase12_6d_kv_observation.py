#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate Phase 12.6D copied KV block-ID observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "schema_version",
    "timestamp",
    "pid",
    "hook",
    "class_name",
    "method_name",
    "result_type",
    "result_repr_preview",
    "block_ids_preview",
    "block_id_count",
    "active_routing",
    "measured_runtime_reduction",
    "runtime_behavior_changed",
    "mutation",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase 12.6D KV observation JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_6d_kv_observation_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_6d_kv_observation_validation.md"
        ),
    )
    return parser.parse_args(argv)


def load_observations(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"KV observation input is missing: {input_path}")
    observations: list[dict[str, Any]] = []
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
        observations.append(value)
    return observations


def validate_observation(
    observation: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [
        field for field in REQUIRED_FIELDS if field not in observation
    ]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    if observation.get("schema_version") != (
        "phase12_6d_kv_observation_v1"
    ):
        errors.append("unsupported schema_version")
    for field in (
        "active_routing",
        "measured_runtime_reduction",
        "runtime_behavior_changed",
        "mutation",
    ):
        if observation.get(field) is not False:
            errors.append(f"{field} must be false")

    preview = observation.get("block_ids_preview")
    count = observation.get("block_id_count")
    truncated = observation.get("block_ids_preview_truncated")
    if not isinstance(preview, list):
        errors.append("block_ids_preview must be a list")
        preview = []
    elif not all(
        isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
        for item in preview
    ):
        errors.append("block_ids_preview must contain non-negative integers")

    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        errors.append("block_id_count must be a non-negative integer")
    elif len(preview) > count:
        errors.append("block_ids_preview cannot exceed block_id_count")
    elif truncated is False and len(preview) != count:
        errors.append("untruncated preview length must match block_id_count")

    minimum = observation.get("min_block_id")
    maximum = observation.get("max_block_id")
    if isinstance(count, int) and count > 0:
        if not isinstance(minimum, int) or minimum < 0:
            errors.append("min_block_id must be a non-negative integer")
        if not isinstance(maximum, int) or maximum < 0:
            errors.append("max_block_id must be a non-negative integer")
        if (
            isinstance(minimum, int)
            and isinstance(maximum, int)
            and minimum > maximum
        ):
            errors.append("min_block_id must not exceed max_block_id")
    elif count == 0 and (minimum is not None or maximum is not None):
        warnings.append("empty observation should normally have null min/max")

    return {
        "index": index,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_observations(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    results = [
        validate_observation(observation, index)
        for index, observation in enumerate(observations)
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
    valid_records = sum(result["valid"] for result in results)
    return {
        "validation_passed": bool(observations) and not errors,
        "total_records": len(observations),
        "valid_records": valid_records,
        "invalid_records": len(observations) - valid_records,
        "errors": errors,
        "warnings": warnings,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "runtime_behavior_changed": False,
        "mutation": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 12.6D KV Observation Validation",
        "",
        "## Status",
        "",
        f"- Passed: `{str(report['validation_passed']).lower()}`",
        f"- Total records: `{report['total_records']}`",
        f"- Valid records: `{report['valid_records']}`",
        f"- Invalid records: `{report['invalid_records']}`",
        "- Active routing: `false`",
        "- Measured runtime reduction: `false`",
        "- Runtime behavior changed: `false`",
        "- Mutation: `false`",
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
        "## Caveats",
        "",
        "- Validation applies to copied observation records only.",
        "- It does not prove active KV selection or memory reduction.",
        "- The wrapped method must return its original result unchanged.",
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
