#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S5.24 ownership-removal validation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.kivo_vd.validate_source_s5_19_demotable_transport_probe import (
    load_summary,
    validate_summary as validate_transport_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S5.24 ownership-removal validation JSON."
    )
    parser.add_argument("--input", required=True)
    return parser.parse_args(argv)


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    base = validate_transport_summary(summary)
    errors = list(base["errors"])
    warnings = list(base["warnings"])
    counters = base["counters"] if isinstance(base.get("counters"), dict) else {}

    ownership_remove_succeeded = int(
        summary.get(
            "ownership_remove_succeeded",
            counters.get("ownership_remove_succeeded", 0),
        )
        or 0
    )
    req_to_blocks_removed = int(
        summary.get("req_to_blocks_removed", counters.get("req_to_blocks_removed", 0))
        or 0
    )
    ownership_remaining_blocks_last = int(
        summary.get(
            "ownership_remaining_blocks_last",
            counters.get("ownership_remaining_blocks_last", 0),
        )
        or 0
    )
    ownership_remove_invariant_failed = int(
        summary.get(
            "ownership_remove_invariant_failed",
            counters.get("ownership_remove_invariant_failed", 0),
        )
        or 0
    )
    ownership_remove_observed = (
        ownership_remove_succeeded > 0 or req_to_blocks_removed > 0
    )

    if summary.get("memory_claim_allowed") is not False:
        errors.append("memory_claim_allowed must be false")
    if summary.get("performance_claim_allowed", False) is not False:
        errors.append("performance_claim_allowed must be false")
    if summary.get("quality_claim_allowed", False) is not False:
        errors.append("quality_claim_allowed must be false")
    if summary.get("live_kv_free_enabled", False) is not False:
        errors.append("live_kv_free_enabled must be false")

    if ownership_remove_observed:
        if summary.get("generation_success") is not True:
            errors.append(
                "generation_success must be true when ownership removal is observed"
            )
        if not base.get("transport_observed"):
            errors.append(
                "transport_observed must be true when ownership removal is observed"
            )
        if ownership_remove_succeeded <= 0:
            errors.append(
                "ownership_remove_succeeded must be > 0 when ownership removal is observed"
            )
        if req_to_blocks_removed <= 0:
            errors.append(
                "req_to_blocks_removed must be > 0 when ownership removal is observed"
            )
        if ownership_remaining_blocks_last <= 0:
            errors.append(
                "ownership_remaining_blocks_last must be > 0 when ownership removal is observed"
            )
        if ownership_remove_invariant_failed != 0:
            errors.append("ownership_remove_invariant_failed must stay 0")
        if int(summary.get("free_to_pool_calls", counters.get("free_to_pool_calls", 0)) or 0) != 0:
            errors.append("free_to_pool_calls must stay 0")
    else:
        warnings.append("ownership_remove_observed=false")

    return {
        "validation_passed": not errors,
        "ownership_remove_observed": ownership_remove_observed,
        "transport_observed": base.get("transport_observed", False),
        "ownership_remove_succeeded": ownership_remove_succeeded,
        "req_to_blocks_removed": req_to_blocks_removed,
        "ownership_remaining_blocks_last": ownership_remaining_blocks_last,
        "ownership_remove_invariant_failed": ownership_remove_invariant_failed,
        "errors": errors,
        "warnings": warnings,
        "counters": counters,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_summary(load_summary(Path(args.input)))
    print(json.dumps(result, indent=2))
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
