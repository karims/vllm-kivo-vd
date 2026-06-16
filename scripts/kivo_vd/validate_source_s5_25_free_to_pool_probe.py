#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S5.25 free-to-pool probe summary."""

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
        description="Validate Phase S5.25 free-to-pool probe JSON."
    )
    parser.add_argument("--input", required=True)
    return parser.parse_args(argv)


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    base = validate_transport_summary(summary)
    errors = [
        error
        for error in list(base["errors"])
        if error != "free_to_pool_calls must stay 0"
    ]
    warnings = list(base["warnings"])
    counters = base["counters"] if isinstance(base.get("counters"), dict) else {}

    free_to_pool_attempted = int(
        summary.get("free_to_pool_attempted", counters.get("free_to_pool_attempted", 0))
        or 0
    )
    free_to_pool_succeeded = int(
        summary.get("free_to_pool_succeeded", counters.get("free_to_pool_succeeded", 0))
        or 0
    )
    free_to_pool_rejected = int(
        summary.get("free_to_pool_rejected", counters.get("free_to_pool_rejected", 0))
        or 0
    )
    free_to_pool_blocks = int(
        summary.get("free_to_pool_blocks", counters.get("free_to_pool_blocks", 0))
        or 0
    )
    free_to_pool_calls = int(
        summary.get("free_to_pool_calls", counters.get("free_to_pool_calls", 0))
        or 0
    )
    free_to_pool_double_free_prevented = int(
        summary.get(
            "free_to_pool_double_free_prevented",
            counters.get("free_to_pool_double_free_prevented", 0),
        )
        or 0
    )
    last_freed_block_ids_sample = tuple(
        summary.get(
            "last_freed_block_ids_sample",
            counters.get("last_freed_block_ids_sample", ()),
        )
        or ()
    )
    last_remaining_block_ids_sample = tuple(
        summary.get(
            "last_remaining_block_ids_sample",
            counters.get("last_remaining_block_ids_sample", ()),
        )
        or ()
    )
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

    if summary.get("generation_success") is not True:
        errors.append("generation_success must be true")
    if not base.get("transport_observed"):
        errors.append("transport_observed must be true")
    if ownership_remove_succeeded <= 0:
        errors.append("ownership_remove_succeeded must be > 0")
    if req_to_blocks_removed <= 0:
        errors.append("req_to_blocks_removed must be > 0")
    if ownership_remaining_blocks_last <= 0:
        errors.append("ownership_remaining_blocks_last must be > 0")
    if ownership_remove_invariant_failed != 0:
        errors.append("ownership_remove_invariant_failed must stay 0")

    if free_to_pool_attempted > 0:
        if free_to_pool_calls > 0 and free_to_pool_succeeded <= 0:
            errors.append(
                "free_to_pool_calls > 0 requires free_to_pool_succeeded > 0"
            )
        if free_to_pool_double_free_prevented > 0:
            errors.append("free_to_pool_double_free_prevented must stay 0")
        if set(last_freed_block_ids_sample) & set(last_remaining_block_ids_sample):
            errors.append("freed block ids must not overlap remaining owned ids")
        if free_to_pool_succeeded <= 0 and free_to_pool_rejected <= 0:
            errors.append(
                "free_to_pool_attempted > 0 requires success or explicit rejection"
            )
    else:
        warnings.append("free_to_pool_not_attempted")

    return {
        "validation_passed": not errors,
        "transport_observed": base.get("transport_observed", False),
        "ownership_remove_observed": ownership_remove_observed,
        "free_to_pool_attempted": free_to_pool_attempted,
        "free_to_pool_succeeded": free_to_pool_succeeded,
        "free_to_pool_rejected": free_to_pool_rejected,
        "free_to_pool_blocks": free_to_pool_blocks,
        "free_to_pool_calls": free_to_pool_calls,
        "free_to_pool_double_free_prevented": free_to_pool_double_free_prevented,
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
