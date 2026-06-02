#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare Kivo-VD sketch backends from offline HF/sweep JSONL rows."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC_KEYS = [
    "block_topk_recall",
    "block_recall_at_2x_budget",
    "block_recall_at_4x_budget",
    "block_score_correlation",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input file has no JSONL rows: {path}")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "sketch_type" not in row or "sketch_dim" not in row:
            continue
        if not all(key in row for key in METRIC_KEYS):
            continue
        grouped[(str(row["sketch_type"]), int(row["sketch_dim"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (sketch_type, sketch_dim), group_rows in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "sketch_type": sketch_type,
            "sketch_dim": sketch_dim,
            "count": len(group_rows),
        }
        for key in METRIC_KEYS:
            payload[f"avg_{key}"] = sum(float(row[key]) for row in group_rows) / len(
                group_rows
            )
        summary.append(payload)
    if not summary:
        raise ValueError(
            "input rows did not contain comparable sketch metrics "
            "(sketch_type, sketch_dim, and block recall/correlation fields)"
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare offline Kivo-VD sketch backend JSONL metrics."
    )
    parser.add_argument(
        "--input",
        default="outputs/kivo_vd/hf_qk_head_sweep_ranked.jsonl",
        help="HF sweep or synthetic sweep JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Summary is always printed to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = summarize_rows(read_jsonl(Path(args.input)))
        payload = {"input": args.input, "summary": summary}
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
