#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S5.25 free-to-pool probe."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (
    parse_args,
    run_generation,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_generation(args)
    summary["phase"] = "S5.25"
    summary["goal"] = "first_gated_free_to_pool_for_removed_demoted_blocks"
    summary["ownership_remove_observed"] = (
        int(summary.get("ownership_remove_succeeded", 0) or 0) > 0
    )
    summary["live_kv_free_enabled"] = (
        int(summary.get("free_to_pool_attempted", 0) or 0) > 0
        or int(summary.get("free_to_pool_succeeded", 0) or 0) > 0
        or int(summary.get("free_to_pool_rejected", 0) or 0) > 0
    )
    summary["memory_claim_allowed"] = False
    summary["quality_claim_allowed"] = False
    summary["performance_claim_allowed"] = False
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("generation_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
