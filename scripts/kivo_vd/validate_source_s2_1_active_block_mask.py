#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate a Phase S2.1 active block mask report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S2.1 active block mask JSON."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s2_1_active_block_mask_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s2_1_active_block_mask_validation.md",
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S2.1 input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


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


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    required = [
        "total_prompts",
        "baseline_success_count",
        "active_success_count",
        "mutation_applied_prompt_count",
        "total_remapped_slot_count",
        "max_visible_block_count",
        "max_selected_block_count",
        "max_unselected_block_count",
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
        "s2_1_active_mask_passed",
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
    if int(report.get("mutation_applied_prompt_count", 0) or 0) <= 0:
        errors.append("mutation_applied_prompt_count must be > 0")
    if int(report.get("total_remapped_slot_count", 0) or 0) <= 0:
        errors.append("total_remapped_slot_count must be > 0")
    for field in [
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")

    claim_paths = sorted(set(_true_claims(report)))
    if claim_paths:
        errors.append(
            "memory, latency, selected-attention, or performance claims "
            f"must remain false: {', '.join(claim_paths)}"
        )
    if report.get("s2_1_active_mask_passed") is not True:
        errors.append("s2_1_active_mask_passed must be true")
    if int(report.get("mutation_applied_prompt_count", 0) or 0) > 0:
        if int(report.get("active_routing_count", 0) or 0) <= 0:
            errors.append("active_routing_count must be > 0 when remaps apply")
        if int(report.get("runtime_behavior_changed_count", 0) or 0) <= 0:
            errors.append(
                "runtime_behavior_changed_count must be > 0 when remaps apply"
            )

    prompt_results = report.get("prompt_results", [])
    if (
        not isinstance(prompt_results, list)
        or len(prompt_results) != total_prompts
    ):
        errors.append("prompt_results must contain one entry per prompt")

    for index, item in enumerate(prompt_results):
        for field in [
            "prompt_index",
            "prompt",
            "baseline_status",
            "active_status",
            "baseline_output",
            "active_output",
            "baseline_error",
            "active_error",
            "output_changed",
            "records_written",
            "max_visible_block_count",
            "max_selected_block_count",
            "max_unselected_block_count",
            "total_remapped_slot_count",
            "mutation_attempted_count",
            "mutation_applied_count",
            "active_routing_count",
            "runtime_behavior_changed_count",
        ]:
            if field not in item:
                errors.append(
                    f"prompt result {index} missing field {field}"
                )
        if item.get("active_status") == "succeeded":
            if item.get("mutation_applied_count", 0) <= 0:
                errors.append(
                    f"prompt result {index} must remap at least one slot"
                )
            if item.get("active_routing_count", 0) <= 0:
                errors.append(
                    f"prompt result {index} must have active_routing_count > 0"
                )
            if item.get("runtime_behavior_changed_count", 0) <= 0:
                errors.append(
                    f"prompt result {index} must have "
                    "runtime_behavior_changed_count > 0"
                )

    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "max_visible_block_count": int(
            report.get("max_visible_block_count", 0) or 0
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S2.1 Active Block Mask Validation",
        "",
        f"- Passed: `{report['validation_passed']}`",
        f"- Total prompts: `{report['total_prompts']}`",
        (
            "- Maximum visible block count: "
            f"`{report['max_visible_block_count']}`"
        ),
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
            "This validation confirms only source-level active remapping. It "
            "does not validate memory reduction, latency improvement, or "
            "selected attention.",
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
        report = validate_report(load_report(args.input_json))
    except Exception as exc:
        report = {
            "validation_passed": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "total_prompts": 0,
            "max_visible_block_count": 0,
        }
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "validation_passed": report["validation_passed"],
                "total_prompts": report["total_prompts"],
                "max_visible_block_count": report["max_visible_block_count"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
