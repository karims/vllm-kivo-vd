#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S5.26 block-pool accounting probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (
    parse_args,
    run_generation,
)


def _snapshot_cuda_memory() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "cuda_memory_snapshot_observed": False,
            "cuda_memory_snapshot_blocker": "torch_unavailable",
        }

    if not torch.cuda.is_available():
        return {
            "cuda_memory_snapshot_observed": False,
            "cuda_memory_snapshot_blocker": "cuda_unavailable",
        }

    try:
        return {
            "cuda_memory_snapshot_observed": True,
            "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
            "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
        }
    except Exception:
        return {
            "cuda_memory_snapshot_observed": False,
            "cuda_memory_snapshot_blocker": "cuda_snapshot_failed",
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cuda_before = _snapshot_cuda_memory()
    first_summary = run_generation(args)
    cuda_after = _snapshot_cuda_memory()

    summary = dict(first_summary)
    summary["phase"] = "S5.26"
    summary["goal"] = "observe_block_pool_accounting_after_gated_free_to_pool"
    summary["ownership_remove_observed"] = (
        int(summary.get("ownership_remove_succeeded", 0) or 0) > 0
    )
    summary["live_kv_free_enabled"] = (
        int(summary.get("free_to_pool_attempted", 0) or 0) > 0
        or int(summary.get("free_to_pool_succeeded", 0) or 0) > 0
        or int(summary.get("free_to_pool_rejected", 0) or 0) > 0
    )
    summary["block_pool_accounting_observed"] = (
        int(summary.get("block_pool_free_accounting_observed", 0) or 0) > 0
    )
    summary["first_generation_success"] = bool(summary.get("generation_success"))
    summary["second_generation_success"] = None
    summary["reuse_probe_enabled"] = False
    summary["reuse_probe_success"] = None
    summary["cuda_memory_allocated_before"] = cuda_before.get("cuda_memory_allocated")
    summary["cuda_memory_reserved_before"] = cuda_before.get("cuda_memory_reserved")
    summary["cuda_memory_allocated_after"] = cuda_after.get("cuda_memory_allocated")
    summary["cuda_memory_reserved_after"] = cuda_after.get("cuda_memory_reserved")
    summary["cuda_memory_snapshot_observed"] = bool(
        cuda_before.get("cuda_memory_snapshot_observed")
        and cuda_after.get("cuda_memory_snapshot_observed")
    )
    summary["cuda_memory_snapshot_blocker"] = (
        cuda_before.get("cuda_memory_snapshot_blocker")
        or cuda_after.get("cuda_memory_snapshot_blocker")
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
