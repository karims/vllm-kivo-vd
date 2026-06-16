#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the Phase S5.26 block-pool accounting probe summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.kivo_vd.validate_source_s5_25_free_to_pool_probe import (
    validate_summary as validate_s5_25_summary,
)
from scripts.kivo_vd.validate_source_s5_19_demotable_transport_probe import (
    load_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase S5.26 block-pool accounting probe JSON."
    )
    parser.add_argument("--input", required=True)
    return parser.parse_args(argv)


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    base = validate_s5_25_summary(summary)
    errors = list(base["errors"])
    warnings = list(base["warnings"])
    counters = base["counters"] if isinstance(base.get("counters"), dict) else {}

    block_pool_free_capacity_before = int(
        summary.get(
            "block_pool_free_capacity_before",
            counters.get("block_pool_free_capacity_before", 0),
        )
        or 0
    )
    block_pool_free_capacity_after = int(
        summary.get(
            "block_pool_free_capacity_after",
            counters.get("block_pool_free_capacity_after", 0),
        )
        or 0
    )
    block_pool_free_capacity_delta = int(
        summary.get(
            "block_pool_free_capacity_delta",
            counters.get("block_pool_free_capacity_delta", 0),
        )
        or 0
    )
    block_pool_num_free_blocks_before = int(
        summary.get(
            "block_pool_num_free_blocks_before",
            counters.get("block_pool_num_free_blocks_before", 0),
        )
        or 0
    )
    block_pool_num_free_blocks_after = int(
        summary.get(
            "block_pool_num_free_blocks_after",
            counters.get("block_pool_num_free_blocks_after", 0),
        )
        or 0
    )
    block_pool_num_free_blocks_delta = int(
        summary.get(
            "block_pool_num_free_blocks_delta",
            counters.get("block_pool_num_free_blocks_delta", 0),
        )
        or 0
    )
    block_pool_free_accounting_observed = int(
        summary.get(
            "block_pool_free_accounting_observed",
            counters.get("block_pool_free_accounting_observed", 0),
        )
        or 0
    )
    block_pool_free_accounting_increased = int(
        summary.get(
            "block_pool_free_accounting_increased",
            counters.get("block_pool_free_accounting_increased", 0),
        )
        or 0
    )
    block_pool_free_accounting_rejected = int(
        summary.get(
            "block_pool_free_accounting_rejected",
            counters.get("block_pool_free_accounting_rejected", 0),
        )
        or 0
    )
    block_pool_free_accounting_blocker_reasons = dict(
        summary.get(
            "block_pool_free_accounting_blocker_reasons",
            counters.get("block_pool_free_accounting_blocker_reasons", {}),
        )
        or {}
    )
    reuse_probe_enabled = bool(summary.get("reuse_probe_enabled", False))
    reuse_probe_success = summary.get("reuse_probe_success")

    if block_pool_free_accounting_observed > 0:
        if block_pool_free_capacity_delta < 0:
            errors.append("block_pool_free_capacity_delta must not be negative")
        if block_pool_num_free_blocks_delta < 0:
            errors.append("block_pool_num_free_blocks_delta must not be negative")
        if (
            block_pool_free_capacity_after < block_pool_free_capacity_before
            or block_pool_num_free_blocks_after < block_pool_num_free_blocks_before
        ):
            errors.append("block pool free counts must not decrease after free")
    else:
        if block_pool_free_accounting_rejected <= 0:
            warnings.append("block_pool_free_accounting_not_observed")
        if not block_pool_free_accounting_blocker_reasons:
            warnings.append("block_pool_free_accounting_missing_blocker_reason")

    if reuse_probe_enabled and reuse_probe_success is not True:
        errors.append("reuse_probe_success must be true when reuse probe is enabled")

    if summary.get("memory_claim_allowed") is not False:
        errors.append("memory_claim_allowed must be false")
    if summary.get("performance_claim_allowed") is not False:
        errors.append("performance_claim_allowed must be false")
    if summary.get("quality_claim_allowed") is not False:
        errors.append("quality_claim_allowed must be false")

    return {
        "validation_passed": not errors,
        "transport_observed": base.get("transport_observed", False),
        "ownership_remove_observed": base.get("ownership_remove_observed", False),
        "free_to_pool_succeeded": base.get("free_to_pool_succeeded", 0),
        "free_to_pool_calls": base.get("free_to_pool_calls", 0),
        "block_pool_free_accounting_observed": block_pool_free_accounting_observed > 0,
        "block_pool_free_capacity_before": block_pool_free_capacity_before,
        "block_pool_free_capacity_after": block_pool_free_capacity_after,
        "block_pool_free_capacity_delta": block_pool_free_capacity_delta,
        "block_pool_num_free_blocks_before": block_pool_num_free_blocks_before,
        "block_pool_num_free_blocks_after": block_pool_num_free_blocks_after,
        "block_pool_num_free_blocks_delta": block_pool_num_free_blocks_delta,
        "reuse_probe_success": reuse_probe_success,
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
