#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S5.19 demotable transport probe summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S5.19 demotable transport probe JSON."
    )
    parser.add_argument("--input", required=True)
    return parser.parse_args(argv)


def load_summary(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"S5.19 summary is missing: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counters = summary.get("counters")
    if not isinstance(counters, dict):
        errors.append("counters must exist and be an object")
        counters = {}

    if summary.get("generation_success") is not True:
        errors.append("generation_success must be true")
    if int(summary.get("prompt_count", 0) or 0) <= 0:
        errors.append("prompt_count must be > 0")
    if summary.get("req_to_blocks_removed", counters.get("req_to_blocks_removed", 0)) != 0:
        errors.append("req_to_blocks_removed must stay 0")
    if summary.get("free_to_pool_calls", counters.get("free_to_pool_calls", 0)) != 0:
        errors.append("free_to_pool_calls must stay 0")
    if summary.get("memory_claim_allowed") is not False:
        errors.append("memory_claim_allowed must be false")
    if summary.get("free_to_pool_claim_allowed") is not False:
        errors.append("free_to_pool_claim_allowed must be false")

    transport_observed = bool(summary.get("transport_observed"))
    if not transport_observed:
        warnings.append("no_demotable_blocks_or_runtime_policy_did_not_emit")

    return {
        "validation_passed": not errors,
        "transport_observed": transport_observed,
        "reason": None if transport_observed else "no_demotable_blocks_or_runtime_policy_did_not_emit",
        "errors": errors,
        "warnings": warnings,
        "counters": counters,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_summary(load_summary(args.input))
    print(json.dumps(result, indent=2))
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
