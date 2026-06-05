#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Summarize offline structured-sketch parameter sweep JSONL results."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_KEYS = [
    "sketch_type",
    "sketch_dim",
    "structured_alpha",
    "structured_coordinate_strategy",
]

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
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not all(key in row for key in GROUP_KEYS):
            continue
        if not all(key in row for key in METRIC_KEYS):
            continue
        grouped[tuple(row[key] for key in GROUP_KEYS)].append(row)

    summary: list[dict[str, Any]] = []
    for values, group_rows in grouped.items():
        payload = {key: value for key, value in zip(GROUP_KEYS, values, strict=True)}
        payload["count"] = len(group_rows)
        for key in METRIC_KEYS:
            payload[f"avg_{key}"] = sum(float(row[key]) for row in group_rows) / len(
                group_rows
            )
        summary.append(payload)

    if not summary:
        raise ValueError("no comparable structured sweep rows found")

    summary.sort(
        key=lambda row: (
            -float(row["avg_block_recall_at_2x_budget"]),
            -float(row["avg_block_topk_recall"]),
            -float(row["avg_block_score_correlation"]),
            str(row["sketch_type"]),
            int(row["sketch_dim"]),
        )
    )
    return summary


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(summary: list[dict[str, Any]], input_path: str) -> str:
    headers = [
        "sketch_type",
        "dim",
        "alpha",
        "strategy",
        "avg top-k",
        "avg recall@2x",
        "avg recall@4x",
        "avg score corr",
        "count",
    ]
    lines = [
        "# Kivo-VD Structured Sketch Parameter Sweep Summary",
        "",
        "This is an offline retrieval summary only. It is not measured vLLM "
        "runtime memory reduction, active routing, latency, or quality evidence.",
        "",
        f"Input: `{input_path}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sketch_type"]),
                    str(row["sketch_dim"]),
                    _fmt(row["structured_alpha"]),
                    str(row["structured_coordinate_strategy"]),
                    _fmt(row["avg_block_topk_recall"]),
                    _fmt(row["avg_block_recall_at_2x_budget"]),
                    _fmt(row["avg_block_recall_at_4x_budget"]),
                    _fmt(row["avg_block_score_correlation"]),
                    str(row["count"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Rows are sorted by avg recall@2x, then avg top-k recall, then avg "
        "block score correlation."
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize offline structured sketch parameter sweep rows."
    )
    parser.add_argument(
        "--input",
        default=(
            "outputs/kivo_vd/runs/phase6_2_structured_param_sweep/"
            "structured_param_sweep.jsonl"
        ),
    )
    parser.add_argument(
        "--json-output",
        default=(
            "outputs/kivo_vd/runs/phase6_2_structured_param_sweep/"
            "structured_param_sweep_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        default=(
            "outputs/kivo_vd/runs/phase6_2_structured_param_sweep/"
            "structured_param_sweep_summary.md"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rows = read_jsonl(Path(args.input))
        summary = summarize_rows(rows)
        payload = {
            "input": args.input,
            "num_input_rows": len(rows),
            "num_summary_rows": len(summary),
            "summary": summary,
            "offline_retrieval_only": True,
        }
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(
            render_markdown(summary, args.input), encoding="utf-8"
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({**payload, "summary": summary[:10]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
