#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Model sketch-buffer overhead under several event-aware accounting modes."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PREFERRED_TYPES = (
    "count_sketch",
    "random_projection",
    "bidiagonal_sign_subsample",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model event-aware Kivo sketch-buffer accounting."
    )
    parser.add_argument("--event-estimate", required=True)
    parser.add_argument("--sketch-overhead", required=True)
    parser.add_argument("--memory-comparison")
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase8_2_sketch_buffer_accounting.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase8_2_sketch_buffer_accounting.md",
    )
    return parser.parse_args(argv)


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} file is missing: {input_path}")
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} contains invalid JSON: {input_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {input_path}")
    return value


def _optional_number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def cumulative_overhead_classification(ratio: float | None) -> str:
    if ratio is None:
        return "unavailable"
    if ratio < 0:
        raise ValueError("cumulative overhead ratio must be non-negative")
    if ratio <= 0.05:
        return "excellent"
    if ratio <= 0.15:
        return "acceptable"
    if ratio <= 0.30:
        return "questionable"
    return "poor"


def break_even_classification(events: int | None) -> str:
    if events is None:
        return "not_applicable"
    if events <= 0:
        raise ValueError("break-even events must be positive")
    if events <= 1:
        return "immediate"
    if events <= 4:
        return "fast"
    if events <= 16:
        return "moderate"
    return "slow"


def calculate_cumulative_skipped_kv(
    estimate: dict[str, Any],
    warnings: list[str],
) -> tuple[float | None, str]:
    aggregate = estimate.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    event_count_value = aggregate.get(
        "estimated_routing_events",
        aggregate.get("total_routing_events"),
    )
    event_count = (
        int(event_count_value)
        if isinstance(event_count_value, int | float)
        and event_count_value > 0
        else None
    )
    per_event = estimate.get("per_event_estimates", [])
    if isinstance(per_event, list) and per_event and event_count is not None:
        values = [
            _optional_number(row.get("skipped_kv_bytes"))
            for row in per_event
            if isinstance(row, dict)
        ]
        if len(values) == event_count and all(value is not None for value in values):
            return sum(value for value in values if value is not None), "per_event_sum"
        warnings.append(
            "per-event skipped-KV rows are incomplete; using aggregate fallback"
        )

    average = _optional_number(aggregate.get("average_skipped_kv_bytes"))
    if average is not None and event_count is not None:
        warnings.append(
            "cumulative skipped KV uses average bytes times routing-event count"
        )
        return average * event_count, "average_times_event_count"

    warnings.append(
        "cumulative skipped KV is unavailable: missing per-event rows or "
        "average/event-count data"
    )
    return None, "unavailable"


def _event_summary(
    estimate: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    aggregate = estimate.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("event estimate lacks an aggregate object")
    cumulative, cumulative_source = calculate_cumulative_skipped_kv(
        estimate, warnings
    )
    return {
        "total_routing_events": aggregate.get(
            "estimated_routing_events",
            aggregate.get("total_routing_events"),
        ),
        "bytes_per_block": _optional_number(estimate.get("bytes_per_block")),
        "average_selected_blocks": aggregate.get("average_selected_blocks"),
        "average_skipped_blocks": aggregate.get("average_skipped_blocks"),
        "average_active_kv_bytes": aggregate.get("average_active_kv_bytes"),
        "average_skipped_kv_bytes": aggregate.get("average_skipped_kv_bytes"),
        "average_estimated_reduction_ratio": aggregate.get(
            "average_estimated_reduction_ratio"
        ),
        "cumulative_skipped_kv_bytes": cumulative,
        "cumulative_skipped_kv_source": cumulative_source,
    }


def model_accounting_row(
    row: dict[str, Any],
    event: dict[str, Any],
    full_kv_pool_bytes: float | None,
) -> dict[str, Any]:
    sketch_bytes = _optional_number(row.get("theoretical_sketch_bytes"))
    if sketch_bytes is None or sketch_bytes < 0:
        raise ValueError("sketch row lacks non-negative theoretical bytes")
    average_skipped = _optional_number(event["average_skipped_kv_bytes"])
    cumulative_skipped = _optional_number(event["cumulative_skipped_kv_bytes"])
    bytes_per_block = _optional_number(event["bytes_per_block"])

    per_event_ratio = (
        sketch_bytes / average_skipped
        if average_skipped is not None and average_skipped > 0
        else None
    )
    cumulative_ratio = (
        sketch_bytes / cumulative_skipped
        if cumulative_skipped is not None and cumulative_skipped > 0
        else None
    )
    net_cumulative = (
        cumulative_skipped - sketch_bytes
        if cumulative_skipped is not None
        else None
    )
    break_even_events = (
        math.ceil(sketch_bytes / average_skipped)
        if average_skipped is not None and average_skipped > 0
        else None
    )
    break_even_blocks = (
        math.ceil(sketch_bytes / bytes_per_block)
        if bytes_per_block is not None and bytes_per_block > 0
        else None
    )
    pool_ratio = (
        sketch_bytes / full_kv_pool_bytes
        if full_kv_pool_bytes is not None and full_kv_pool_bytes > 0
        else row.get("sketch_overhead_ratio_vs_full_kv")
    )
    return {
        "sketch_type": row.get("sketch_type"),
        "sketch_dim": row.get("sketch_dim"),
        "measured_allocated_delta_bytes": row.get(
            "measured_allocated_delta_bytes"
        ),
        "global_pool_model": {
            "sketch_pool_bytes": sketch_bytes,
            "full_kv_pool_bytes": full_kv_pool_bytes,
            "sketch_pool_ratio_vs_full_kv_pool": pool_ratio,
        },
        "average_per_event_model": {
            "average_skipped_kv_bytes": average_skipped,
            "overhead_vs_avg_skipped_kv": per_event_ratio,
        },
        "cumulative_request_model": {
            "cumulative_skipped_kv_bytes": cumulative_skipped,
            "overhead_vs_cumulative_skipped_kv": cumulative_ratio,
            "net_cumulative_theoretical_bytes": net_cumulative,
            "classification": cumulative_overhead_classification(
                cumulative_ratio
            ),
        },
        "break_even_model": {
            "break_even_events": break_even_events,
            "break_even_events_classification": break_even_classification(
                break_even_events
            ),
            "break_even_skipped_blocks": break_even_blocks,
        },
    }


def _recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["sketch_type"] in PREFERRED_TYPES
        and row["sketch_dim"] in {16, 32}
    ]
    candidates.sort(
        key=lambda row: (
            row["cumulative_request_model"][
                "overhead_vs_cumulative_skipped_kv"
            ]
            if row["cumulative_request_model"][
                "overhead_vs_cumulative_skipped_kv"
            ]
            is not None
            else float("inf"),
            row["break_even_model"]["break_even_events"]
            if row["break_even_model"]["break_even_events"] is not None
            else float("inf"),
            row["sketch_dim"],
        )
    )
    return {
        "preferred_configs": [
            {
                "sketch_type": row["sketch_type"],
                "sketch_dim": row["sketch_dim"],
                "cumulative_classification": row[
                    "cumulative_request_model"
                ]["classification"],
                "break_even_events": row["break_even_model"][
                    "break_even_events"
                ],
            }
            for row in candidates
        ],
        "recommendation": (
            "Keep CountSketch and Random Projection dims 16/32 as simple "
            "accounting baselines. Keep bidiagonal_sign_subsample dims 16/32 "
            "experimental. Treat dim 64 as high-overhead/reference unless "
            "later evidence changes the tradeoff. Do not enable active routing."
        ),
    }


def build_report(
    *,
    event_estimate_path: str | Path,
    sketch_overhead_path: str | Path,
    memory_comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    estimate = _load_json(event_estimate_path, "event estimate")
    overhead = _load_json(sketch_overhead_path, "sketch overhead")
    comparison = (
        _load_json(memory_comparison_path, "memory comparison")
        if memory_comparison_path is not None
        else None
    )
    event = _event_summary(estimate, warnings)
    full_pool = _optional_number(overhead.get("full_kv_bytes"))
    overhead_rows = overhead.get("rows")
    if not isinstance(overhead_rows, list) or not overhead_rows:
        raise ValueError("sketch overhead JSON lacks non-empty rows")
    rows = [
        model_accounting_row(row, event, full_pool)
        for row in overhead_rows
        if isinstance(row, dict)
    ]
    if not rows:
        raise ValueError("sketch overhead JSON contains no valid row objects")

    measured_context = None
    if comparison is not None:
        conclusion = comparison.get("conclusion", {})
        caveats = comparison.get("caveats", {})
        measured_context = {
            "measured_runtime_drop_observed": (
                conclusion.get("measured_runtime_drop_observed")
                if isinstance(conclusion, dict)
                else None
            ),
            "measured_runtime_reduction": (
                caveats.get("measured_runtime_reduction", False)
                if isinstance(caveats, dict)
                else False
            ),
        }
    return {
        "input_paths": {
            "event_estimate": str(event_estimate_path),
            "sketch_overhead": str(sketch_overhead_path),
            "memory_comparison": (
                str(memory_comparison_path)
                if memory_comparison_path is not None
                else None
            ),
        },
        "event_estimate_summary": event,
        "sketch_overhead_metadata": {
            "model_kv_metadata": overhead.get("model_kv_metadata"),
            "buffer_assumption": overhead.get("buffer_assumption"),
            "full_kv_pool_bytes": full_pool,
        },
        "accounting_rows": rows,
        "recommendations": _recommend(rows),
        "measured_memory_context": measured_context,
        "warnings": warnings,
        "caveats": {
            "theoretical_only": True,
            "measured_runtime_reduction": False,
            "active_routing": False,
            "full_kv_still_allocated": True,
            "latency_improvement_claimed": False,
        },
    }


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _append_table(
    lines: list[str],
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append(
            "| " + " | ".join(f"`{_format(value)}`" for value in row) + " |"
        )


def render_markdown(report: dict[str, Any]) -> str:
    event = report["event_estimate_summary"]
    rows = report["accounting_rows"]
    lines = [
        "# Kivo-VD Phase 8.2 Event-Aware Sketch-Buffer Accounting",
        "",
        "**Status:** Theoretical-only accounting. Full KV is still allocated, "
        "and no active routing is implemented.",
        "",
        "## Event Estimate Summary",
        "",
    ]
    _append_table(
        lines,
        ["metric", "value"],
        [
            ["routing events", event["total_routing_events"]],
            ["bytes per KV block", event["bytes_per_block"]],
            ["average selected blocks", event["average_selected_blocks"]],
            ["average skipped blocks", event["average_skipped_blocks"]],
            ["average active KV bytes", event["average_active_kv_bytes"]],
            ["average skipped KV bytes", event["average_skipped_kv_bytes"]],
            [
                "average estimated reduction ratio",
                event["average_estimated_reduction_ratio"],
            ],
            ["cumulative skipped KV bytes", event["cumulative_skipped_kv_bytes"]],
            ["cumulative source", event["cumulative_skipped_kv_source"]],
        ],
    )
    lines.extend(["", "## Global Pool Overhead", ""])
    _append_table(
        lines,
        ["sketch", "dim", "sketch bytes", "full pool bytes", "pool ratio"],
        [
            [
                row["sketch_type"],
                row["sketch_dim"],
                row["global_pool_model"]["sketch_pool_bytes"],
                row["global_pool_model"]["full_kv_pool_bytes"],
                row["global_pool_model"][
                    "sketch_pool_ratio_vs_full_kv_pool"
                ],
            ]
            for row in rows
        ],
    )
    lines.extend(["", "## Conservative Per-Event Model", ""])
    _append_table(
        lines,
        ["sketch", "dim", "avg skipped bytes", "overhead/avg skipped"],
        [
            [
                row["sketch_type"],
                row["sketch_dim"],
                row["average_per_event_model"]["average_skipped_kv_bytes"],
                row["average_per_event_model"][
                    "overhead_vs_avg_skipped_kv"
                ],
            ]
            for row in rows
        ],
    )
    lines.extend(["", "## Cumulative Request Model", ""])
    _append_table(
        lines,
        [
            "sketch",
            "dim",
            "cumulative skipped",
            "overhead/cumulative",
            "net theoretical bytes",
            "class",
        ],
        [
            [
                row["sketch_type"],
                row["sketch_dim"],
                row["cumulative_request_model"][
                    "cumulative_skipped_kv_bytes"
                ],
                row["cumulative_request_model"][
                    "overhead_vs_cumulative_skipped_kv"
                ],
                row["cumulative_request_model"][
                    "net_cumulative_theoretical_bytes"
                ],
                row["cumulative_request_model"]["classification"],
            ]
            for row in rows
        ],
    )
    lines.extend(["", "## Break-Even Model", ""])
    _append_table(
        lines,
        ["sketch", "dim", "events", "event class", "skipped blocks"],
        [
            [
                row["sketch_type"],
                row["sketch_dim"],
                row["break_even_model"]["break_even_events"],
                row["break_even_model"][
                    "break_even_events_classification"
                ],
                row["break_even_model"]["break_even_skipped_blocks"],
            ]
            for row in rows
        ],
    )
    lines.extend([
        "",
        "## Recommendations",
        "",
        report["recommendations"]["recommendation"],
        "",
        "## Caveats",
        "",
        "- This accounting is theoretical only.",
        "- Full KV is still allocated.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency or quality claim follows from this report.",
        "",
        "## Next Steps",
        "",
        "- Validate Phase 8.0 allocator deltas on CUDA.",
        "- Use exact per-event cumulative accounting when uncapped rows exist.",
        "- Keep attention and KV allocation unchanged.",
    ])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        args = _parse_args()
        report = build_report(
            event_estimate_path=args.event_estimate,
            sketch_overhead_path=args.sketch_overhead,
            memory_comparison_path=args.memory_comparison,
        )
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "num_configurations": len(report["accounting_rows"]),
                    "cumulative_source": report["event_estimate_summary"][
                        "cumulative_skipped_kv_source"
                    ],
                    "warnings": report["warnings"],
                    "theoretical_only": True,
                    "measured_runtime_reduction": False,
                    "active_routing": False,
                },
                separators=(",", ":"),
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "theoretical_only": True,
                    "measured_runtime_reduction": False,
                    "active_routing": False,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
