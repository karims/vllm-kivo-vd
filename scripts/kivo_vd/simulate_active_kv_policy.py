#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Offline active-KV policy simulator for Kivo-VD sketch sweep rows."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

METADATA_FIELDS = [
    "model_name",
    "extraction_mode",
    "qk_space",
    "num_query_heads",
    "num_key_value_heads",
    "selected_query_head",
    "selected_kv_head",
    "head_dim",
    "effective_input_dim",
    "effective_sketch_dim",
    "sketch_compression_ratio",
    "is_full_dimensional_sketch",
]


def _parse_int_csv(spec: str, *, label: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError(f"{label} values must be non-negative.")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"{label} list is empty.")
    return out


def _parse_group_by(spec: str) -> list[str]:
    out = [part.strip() for part in spec.split(",") if part.strip()]
    if not out:
        raise ValueError("group-by list is empty.")
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            rows.append(row)
    return rows


def _as_int_list(row: dict[str, Any], field: str) -> list[int]:
    value = row.get(field)
    if not isinstance(value, list):
        raise ValueError(f"Input row missing list field {field!r}.")
    return [int(v) for v in value]


def _ranked_block_ids(row: dict[str, Any]) -> list[int]:
    if "approx_ranked_block_ids" not in row:
        raise ValueError(
            "Input row is missing 'approx_ranked_block_ids'. Rerun "
            "scripts/kivo_vd/run_hf_qk_head_sweep.py with "
            "--include-ranked-blocks before policy simulation."
        )
    return _as_int_list(row, "approx_ranked_block_ids")


def _recent_block_ids(num_total_blocks: int, recent_window_blocks: int) -> list[int]:
    if recent_window_blocks <= 0:
        return []
    start = max(0, num_total_blocks - recent_window_blocks)
    return list(range(start, num_total_blocks))


def simulate_policy_for_row(
    row: dict[str, Any],
    *,
    recent_window_blocks: int,
    candidate_budget_blocks: int,
    topk_blocks: int,
    min_total_blocks: int = 8,
) -> dict[str, Any] | None:
    """Simulate recent + sketch-candidate active block residency for one row."""

    num_keys_used = int(row["num_keys_used"])
    block_size = int(row["block_size"])
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    num_total_blocks = int(math.ceil(num_keys_used / block_size))
    if num_total_blocks < min_total_blocks:
        return None

    exact_top_block_ids = _as_int_list(row, "exact_top_block_ids")[:topk_blocks]
    approx_ranked_block_ids = _ranked_block_ids(row)
    recent = _recent_block_ids(num_total_blocks, recent_window_blocks)
    candidates = approx_ranked_block_ids[:candidate_budget_blocks]

    active_blocks = {
        int(block_id)
        for block_id in [*recent, *candidates]
        if 0 <= int(block_id) < num_total_blocks
    }
    active_block_count = len(active_blocks)
    active_block_ratio = (
        active_block_count / num_total_blocks if num_total_blocks else 0.0
    )
    exact_top_recall = (
        sum(1 for block_id in exact_top_block_ids if block_id in active_blocks)
        / len(exact_top_block_ids)
        if exact_top_block_ids
        else 0.0
    )

    output = {
        "sketch_type": row.get("sketch_type"),
        "sketch_dim": row.get("sketch_dim"),
        "layer": row.get("layer"),
        "head": row.get("head"),
        "query_position": row.get("query_position"),
        "num_total_blocks": num_total_blocks,
        "recent_window_blocks": int(recent_window_blocks),
        "candidate_budget_blocks": int(candidate_budget_blocks),
        "active_block_count": active_block_count,
        "active_block_ratio": float(active_block_ratio),
        "estimated_kv_reduction": float(1.0 - active_block_ratio),
        "exact_top_recall_in_active": float(exact_top_recall),
    }
    for field in METADATA_FIELDS:
        if field in row:
            output[field] = row[field]
    return output


def _aggregate(rows: list[dict[str, Any]], group_by: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = [*group_by, "recent_window_blocks", "candidate_budget_blocks"]
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for values, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        count = len(group_rows)
        payload = dict(zip(keys, values, strict=True))
        payload.update(
            {
                "count": count,
                "avg_active_block_ratio": sum(
                    float(r["active_block_ratio"]) for r in group_rows
                )
                / count,
                "avg_estimated_kv_reduction": sum(
                    float(r["estimated_kv_reduction"]) for r in group_rows
                )
                / count,
                "avg_exact_top_recall_in_active": sum(
                    float(r["exact_top_recall_in_active"]) for r in group_rows
                )
                / count,
            }
        )
        out.append(payload)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate Kivo-VD active KV block residency from HF sweep rows."
    )
    parser.add_argument(
        "--input", default="outputs/kivo_vd/hf_qk_head_sweep.jsonl"
    )
    parser.add_argument(
        "--output", default="outputs/kivo_vd/active_kv_policy_simulation.jsonl"
    )
    parser.add_argument("--recent-window-blocks", default="4,8,16")
    parser.add_argument("--candidate-budget-blocks", default="8,16,32")
    parser.add_argument("--topk-blocks", type=int, default=4)
    parser.add_argument("--group-by", default="sketch_type,sketch_dim")
    parser.add_argument("--min-total-blocks", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    recent_windows = _parse_int_csv(
        args.recent_window_blocks, label="recent-window-blocks"
    )
    candidate_budgets = _parse_int_csv(
        args.candidate_budget_blocks, label="candidate-budget-blocks"
    )
    group_by = _parse_group_by(args.group_by)

    input_rows = _read_jsonl(input_path)
    output_rows: list[dict[str, Any]] = []
    for input_row in input_rows:
        for recent_window in recent_windows:
            for candidate_budget in candidate_budgets:
                simulated = simulate_policy_for_row(
                    input_row,
                    recent_window_blocks=recent_window,
                    candidate_budget_blocks=candidate_budget,
                    topk_blocks=args.topk_blocks,
                    min_total_blocks=args.min_total_blocks,
                )
                if simulated is not None:
                    output_rows.append(simulated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = _aggregate(output_rows, group_by)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "input_rows": len(input_rows),
                "output_rows": len(output_rows),
                "summary": summary,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
