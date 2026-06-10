#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S1.2 source-level valid-slot mutation quality sanity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Phase S1.2 source-level valid-slot mutation quality "
            "sanity JSON."
        )
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s1_2_quality_sanity_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s1_2_quality_sanity_validation.md",
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"quality sanity input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    errors = []
    required = [
        "total_prompts",
        "baseline_success_count",
        "active_success_count",
        "mutation_applied_prompt_count",
        "output_changed_count",
        "output_unchanged_count",
        "total_mutation_applied_records",
        "total_active_records",
        "measured_runtime_reduction",
        "quality_sanity_passed",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
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
    if report.get("active_success_count") != total_prompts:
        errors.append("active_success_count must equal total_prompts")
    if report.get("total_mutation_applied_records", 0) <= 0:
        errors.append("total_mutation_applied_records must be > 0")
    if report.get("measured_runtime_reduction") is not False:
        errors.append("measured_runtime_reduction must be false")
    if report.get("selected_attention_claim_allowed") is not False:
        errors.append("selected_attention_claim_allowed must be false")
    if report.get("performance_claim_allowed") is not False:
        errors.append("performance_claim_allowed must be false")
    if report.get("quality_sanity_passed") is not True:
        errors.append("quality_sanity_passed must be true")

    prompt_results = report.get("prompt_results", [])
    if not isinstance(prompt_results, list) or len(prompt_results) != total_prompts:
        errors.append("prompt_results must contain one entry per prompt")
        prompt_results = []

    required_prompt_fields = [
        "prompt_index",
        "prompt",
        "baseline_status",
        "active_status",
        "baseline_output",
        "active_output",
        "output_changed",
        "active_error",
        "mutation_attempted_count",
        "mutation_applied_count",
        "active_routing_count",
        "runtime_behavior_changed_count",
        "max_valid_slot_count",
        "min_valid_slot_count",
        "old_new_differ_count",
        "blocker_reasons",
        "baseline_records_written",
        "active_records_written",
        "measured_runtime_reduction",
    ]
    for index, item in enumerate(prompt_results):
        missing_prompt_fields = [
            field for field in required_prompt_fields if field not in item
        ]
        if missing_prompt_fields:
            errors.append(
                f"prompt result {index} missing fields: "
                f"{', '.join(missing_prompt_fields)}"
            )
            continue
        if item.get("active_status") == "succeeded":
            if item.get("mutation_applied_count", 0) == 0:
                if not item.get("blocker_reasons"):
                    errors.append(
                        f"prompt result {index} with succeeded active run "
                        "needs blocker_reasons when no mutation was applied"
                    )
            else:
                if item.get("old_new_differ_count", 0) <= 0:
                    errors.append(
                        f"prompt result {index} with mutation applied "
                        "must have old_new_differ_count > 0"
                    )
                if item.get("active_routing_count", 0) <= 0:
                    errors.append(
                        f"prompt result {index} with mutation applied "
                        "must have active_routing_count > 0"
                    )
                if item.get("runtime_behavior_changed_count", 0) <= 0:
                    errors.append(
                        f"prompt result {index} with mutation applied "
                        "must have runtime_behavior_changed_count > 0"
                    )

    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S1.2 Source-Level Quality Sanity Validation",
        "",
        f"- Passed: `{report['validation_passed']}`",
        f"- Total prompts: `{report.get('total_prompts')}`",
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        lines.extend(f"- {item}" for item in report["errors"])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = validate_report(load_report(args.input_json))
    except Exception as exc:
        report = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
        }
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "validation_passed": report["validation_passed"],
                "total_prompts": report["total_prompts"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
