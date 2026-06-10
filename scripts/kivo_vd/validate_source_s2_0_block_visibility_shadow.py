#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate a Phase S2.0 block visibility shadow report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S2.0 block visibility shadow JSON."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s2_0_block_visibility_shadow_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s2_0_block_visibility_shadow_validation.md"
        ),
    )
    return parser.parse_args(argv)


def load_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S2.0 input is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def _true_claims(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            key_lower = key.lower()
            prohibited_claim = (
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
            if prohibited_claim and child is True:
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
        "shadow_success_count",
        "total_records",
        "mutation_applied_count",
        "active_routing_count",
        "runtime_behavior_changed_count",
        "max_visible_block_count",
        "max_theoretical_visible_block_reduction",
        "max_theoretical_visible_block_reduction_ratio",
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
        "s2_shadow_passed",
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
    if report.get("shadow_success_count") != total_prompts:
        errors.append("shadow_success_count must equal total_prompts")
    if int(report.get("total_records", 0) or 0) <= 0:
        errors.append("total_records must be > 0")
    for field in [
        "mutation_applied_count",
        "active_routing_count",
        "runtime_behavior_changed_count",
    ]:
        if report.get(field) != 0:
            errors.append(f"{field} must be 0")
    for field in [
        "measured_runtime_reduction",
        "selected_attention_claim_allowed",
        "performance_claim_allowed",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field} must be false")

    max_visible = int(report.get("max_visible_block_count", 0) or 0)
    if max_visible < 1:
        errors.append("max_visible_block_count must be >= 1")
    if max_visible >= 2:
        if int(
            report.get(
                "max_theoretical_visible_block_reduction", -1
            )
        ) < 0:
            errors.append(
                "max_theoretical_visible_block_reduction must be >= 0"
            )
        if report.get(
            "max_theoretical_visible_block_reduction_ratio"
        ) is None:
            errors.append(
                "max_theoretical_visible_block_reduction_ratio is required"
            )

    claim_paths = sorted(set(_true_claims(report)))
    if claim_paths:
        errors.append(
            "memory, latency, selected-attention, or performance claims "
            f"must remain false: {', '.join(claim_paths)}"
        )
    if report.get("s2_shadow_passed") is not True:
        errors.append("s2_shadow_passed must be true")

    prompt_results = report.get("prompt_results", [])
    if (
        not isinstance(prompt_results, list)
        or len(prompt_results) != total_prompts
    ):
        errors.append("prompt_results must contain one entry per prompt")

    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_prompts": total_prompts,
        "max_visible_block_count": max_visible,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S2.0 Shadow Validation",
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
            "This validation confirms observational shadow behavior only. "
            "It does not validate memory reduction, latency improvement, "
            "or selected attention.",
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
                "max_visible_block_count": report[
                    "max_visible_block_count"
                ],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
