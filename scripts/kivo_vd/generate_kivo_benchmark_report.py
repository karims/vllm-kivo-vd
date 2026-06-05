#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate a conservative Kivo-VD offline benchmark report."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HF_METRICS = [
    "block_topk_recall",
    "block_recall_at_2x_budget",
    "block_recall_at_4x_budget",
    "block_score_correlation",
]

POLICY_METRICS = [
    "active_block_ratio",
    "estimated_kv_reduction",
    "exact_top_recall_in_active",
]

METADATA_FIELDS = [
    "model_name",
    "extraction_mode",
    "qk_space",
    "num_query_heads",
    "num_key_value_heads",
    "head_dim",
    "effective_sketch_dim",
    "sketch_compression_ratio",
    "is_full_dimensional_sketch",
]


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{label} input file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{label} input has invalid JSON on line {line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{label} input line {line_no} is not a JSON object."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} input file has no JSONL rows: {path}")
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _group_rows(
    rows: list[dict[str, Any]], keys: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for values, group_rows in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        payload = dict(zip(keys, values, strict=True))
        payload["count"] = len(group_rows)
        for metric in HF_METRICS:
            if metric in group_rows[0]:
                payload[f"avg_{metric}"] = _mean(group_rows, metric)
        for metric in POLICY_METRICS:
            if metric in group_rows[0]:
                payload[f"avg_{metric}"] = _mean(group_rows, metric)
        out.append(payload)
    return out


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _retrieval_summary_table(hf_rows: list[dict[str, Any]]) -> str:
    group_keys = [
        key
        for key in [
            "model_name",
            "extraction_mode",
            "qk_space",
            "sketch_type",
            "sketch_dim",
            "head_dim",
            "effective_sketch_dim",
            "sketch_compression_ratio",
            "is_full_dimensional_sketch",
        ]
        if any(key in row for row in hf_rows)
    ]
    grouped = _group_rows(hf_rows, group_keys)
    rows: list[list[str]] = []
    for row in grouped:
        rows.append(
            [
                str(row.get("model_name", "-")),
                str(row.get("extraction_mode", "-")),
                str(row.get("qk_space", "-")),
                str(row["sketch_type"]),
                str(row["sketch_dim"]),
                str(row.get("head_dim", "-")),
                str(row.get("effective_sketch_dim", "-")),
                _format_float(row.get("sketch_compression_ratio")),
                str(row.get("is_full_dimensional_sketch", "-")),
                _format_float(row.get("avg_block_topk_recall")),
                _format_float(row.get("avg_block_recall_at_2x_budget")),
                _format_float(row.get("avg_block_recall_at_4x_budget")),
                _format_float(row.get("avg_block_score_correlation")),
                str(row["count"]),
            ]
        )
    return _markdown_table(
        [
            "model_name",
            "extraction_mode",
            "qk_space",
            "sketch_type",
            "sketch_dim",
            "head_dim",
            "effective_sketch_dim",
            "compression ratio",
            "full-dim",
            "avg block top-k recall",
            "avg recall@2x",
            "avg recall@4x",
            "avg block score corr",
            "count",
        ],
        rows,
    )


def _metadata_summary_table(hf_rows: list[dict[str, Any]]) -> str:
    present_fields = [
        field for field in METADATA_FIELDS if any(field in row for row in hf_rows)
    ]
    if not present_fields:
        return "No model/extraction metadata fields were present in the HF rows."

    grouped = _group_rows(hf_rows, present_fields)
    rows = [
        [str(row.get(field, "-")) for field in present_fields] + [str(row["count"])]
        for row in grouped
    ]
    return _markdown_table([*present_fields, "count"], rows)


def _has_pre_rope_rows(hf_rows: list[dict[str, Any]]) -> bool:
    return any(row.get("qk_space") == "pre_rope_projection" for row in hf_rows)


def _has_srht_rows(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("sketch_type") == "srht" for row in rows)


def _has_full_dimensional_rows(rows: list[dict[str, Any]]) -> bool:
    return any(bool(row.get("is_full_dimensional_sketch")) for row in rows)


def _selected_policy_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_types = {
        "bidiagonal_sign",
        "count_sketch",
        "random_projection",
        "srht",
    }
    selected_dims = {32, 64, 128}
    selected_policies = {(8, 16), (4, 8)}
    return [
        row
        for row in policy_rows
        if row.get("sketch_type") in selected_types
        and int(row.get("sketch_dim", -1)) in selected_dims
        and (
            int(row.get("recent_window_blocks", -1)),
            int(row.get("candidate_budget_blocks", -1)),
        )
        in selected_policies
    ]


def _policy_summary_table(policy_rows: list[dict[str, Any]]) -> str:
    selected = _selected_policy_rows(policy_rows)
    if not selected:
        selected = policy_rows

    grouped = _group_rows(
        selected,
        [
            "sketch_type",
            "sketch_dim",
            "recent_window_blocks",
            "candidate_budget_blocks",
        ],
    )
    rows: list[list[str]] = []
    for row in grouped:
        rows.append(
            [
                str(row["sketch_type"]),
                str(row["sketch_dim"]),
                str(row["recent_window_blocks"]),
                str(row["candidate_budget_blocks"]),
                _format_float(row.get("avg_active_block_ratio")),
                _format_float(row.get("avg_estimated_kv_reduction")),
                _format_float(row.get("avg_exact_top_recall_in_active")),
                str(row["count"]),
            ]
        )
    return _markdown_table(
        [
            "sketch_type",
            "sketch_dim",
            "recent",
            "candidates",
            "avg active ratio",
            "avg estimated reduction",
            "avg exact-top recall",
            "count",
        ],
        rows,
    )


def _policy_average(
    policy_rows: list[dict[str, Any]], recent: int, candidates: int
) -> dict[str, float] | None:
    matching = [
        row
        for row in policy_rows
        if int(row.get("recent_window_blocks", -1)) == recent
        and int(row.get("candidate_budget_blocks", -1)) == candidates
        and int(row.get("sketch_dim", -1)) in {64, 128}
        and row.get("sketch_type") in {"count_sketch", "random_projection"}
    ]
    if not matching:
        return None
    return {
        "active_block_ratio": _mean(matching, "active_block_ratio"),
        "estimated_kv_reduction": _mean(matching, "estimated_kv_reduction"),
        "exact_top_recall_in_active": _mean(
            matching, "exact_top_recall_in_active"
        ),
    }


def _percentage_sentence(value: float | None, noun: str) -> str:
    if value is None:
        return f"{noun} was not available in the selected policy rows"
    return f"{noun} was about {value * 100:.1f}%"


def generate_report(
    hf_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]
) -> str:
    conservative = _policy_average(policy_rows, recent=8, candidates=16)
    aggressive = _policy_average(policy_rows, recent=4, candidates=8)

    conservative_reduction = (
        conservative["estimated_kv_reduction"] if conservative else None
    )
    conservative_recall = (
        conservative["exact_top_recall_in_active"] if conservative else None
    )
    aggressive_reduction = (
        aggressive["estimated_kv_reduction"] if aggressive else None
    )
    aggressive_recall = (
        aggressive["exact_top_recall_in_active"] if aggressive else None
    )

    conservative_reduction_text = _percentage_sentence(
        conservative_reduction, "estimated active-KV reduction"
    )
    conservative_recall_text = _percentage_sentence(
        conservative_recall, "exact-top-block recall"
    )
    aggressive_reduction_text = _percentage_sentence(
        aggressive_reduction, "estimated active-KV reduction"
    )
    aggressive_recall_text = _percentage_sentence(
        aggressive_recall, "exact-top-block recall"
    )

    lines = [
        "# Kivo-VD Offline Benchmark Report",
        "",
        "## Status",
        "",
        "This report summarizes offline HuggingFace Q/K sketch retrieval and "
        "active-KV policy simulation. It is not a measured vLLM runtime memory "
        "reduction, latency result, or quality benchmark.",
        "",
        "## Executive Summary",
        "",
        "- Sketch-based candidate retrieval works well in these offline "
        "GPT-2-style Q/K tests.",
        "- Conservative policy estimate: "
        f"{conservative_reduction_text}; {conservative_recall_text}.",
        "- Aggressive policy estimates can show higher reduction, but they need "
        "runtime validation and quality checks before being treated as safe.",
        "- No model architecture, tokenizer, training, or weight changes are "
        "part of these results.",
        "",
        "## Model and Extraction Metadata",
        "",
        _metadata_summary_table(hf_rows),
        "",
        *(
            [
                "Note: at least one row uses `qk_space=pre_rope_projection`. "
                "Those results are based on Q/K after linear projection but "
                "before RoPE is applied. Runtime post-RoPE attention behavior "
                "may differ, so these numbers are not final vLLM runtime "
                "claims.",
                "",
            ]
            if _has_pre_rope_rows(hf_rows)
            else []
        ),
        "## Retrieval Benchmark Summary",
        "",
        _retrieval_summary_table(hf_rows),
        "",
        *(
            [
                "Note: `srht` rows are experimental. SRHT should be compared "
                "against CountSketch and Random Projection before being used "
                "as a default, and these offline rows do not imply runtime "
                "memory reduction.",
                "",
            ]
            if _has_srht_rows(hf_rows) or _has_srht_rows(policy_rows)
            else []
        ),
        *(
            [
                "Note: `bidiagonal_sign` rows are experimental structured "
                "linear-algebra sketch rows. They are baseline research "
                "signals only and do not imply active routing, quality "
                "preservation, or measured memory reduction.",
                "",
            ]
            if any(
                row.get("sketch_type") == "bidiagonal_sign"
                for row in [*hf_rows, *policy_rows]
            )
            else []
        ),
        *(
            [
                "## Full-Dimensional Sketch Caveat",
                "",
                "Rows with `is_full_dimensional_sketch=True` should not be "
                "treated as compressed KV sketches. For example, SRHT dim 64 "
                "on GPT-2 head_dim 64 is useful as a correctness/reference "
                "result, but it is not evidence of sketch compression.",
                "",
            ]
            if _has_full_dimensional_rows(hf_rows)
            or _has_full_dimensional_rows(policy_rows)
            else []
        ),
        "## Active KV Policy Simulation Summary",
        "",
        _policy_summary_table(policy_rows),
        "",
        "## Conservative Recommended Policy",
        "",
        "Recommended starting policy for future dry-run/runtime experiments:",
        "",
        "- `sketch_type`: `count_sketch` dim 64, with "
        "`random_projection` dim 64 retained as a baseline.",
        "- `srht` and `bidiagonal_sign` are experimental and should be "
        "compared offline before any runtime policy uses them.",
        "- `recent_window_blocks`: 8",
        "- `candidate_budget_blocks`: 16",
        "",
        "This policy is intentionally conservative: it aims for meaningful but "
        "not extreme active-KV reduction while keeping exact-top-block recall "
        "near the safest observed range.",
        "",
        "## Aggressive Policy Notes",
        "",
        "A stretch policy uses `recent_window_blocks=4` and "
        "`candidate_budget_blocks=8`.",
        "",
        f"- {aggressive_reduction_text}.",
        f"- {aggressive_recall_text}.",
        "",
        "Treat this as a research signal, not a product or runtime claim. It "
        "needs quality, latency, and real memory validation.",
        "",
        "## What Is Proven vs Not Proven",
        "",
        "Proven/offline in these experiments:",
        "",
        "- Sketch candidate retrieval on GPT-2-style Q/K tensors.",
        "- Active-KV policy simulation from ranked candidate blocks.",
        "",
        "Not proven yet:",
        "",
        "- Real vLLM runtime memory reduction.",
        "- Benchmark quality preservation.",
        "- Latency improvement.",
        "- Behavior on modern RoPE/GQA models.",
        "- Book-inspired variation-diminishing or bidiagonal sketches.",
        "",
        "## Next Experiments",
        "",
        "- Runtime dry-run on real vLLM inference.",
        "- Quality benchmarks with conservative and aggressive policies.",
        "- Real measured GPU memory experiments.",
        "- Modern model support, especially RoPE and GQA/MQA models.",
        "- Implement book-inspired sketch variants as experimental backends.",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a conservative Kivo-VD benchmark report."
    )
    parser.add_argument(
        "--hf-sweep", default="outputs/kivo_vd/hf_qk_head_sweep_ranked.jsonl"
    )
    parser.add_argument(
        "--policy-sim",
        default="outputs/kivo_vd/active_kv_policy_simulation.jsonl",
    )
    parser.add_argument(
        "--output", default="outputs/kivo_vd/kivo_vd_benchmark_report.md"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        hf_rows = _read_jsonl(Path(args.hf_sweep), "HF sweep")
        policy_rows = _read_jsonl(Path(args.policy_sim), "policy simulation")
        report = generate_report(hf_rows, policy_rows)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(output_path),
                "hf_rows": len(hf_rows),
                "policy_rows": len(policy_rows),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
