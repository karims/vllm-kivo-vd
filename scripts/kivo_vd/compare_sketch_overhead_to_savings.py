#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare sketch-buffer overhead with theoretical skipped-KV bytes."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PREFERRED_RUNTIME_TYPES = (
    "count_sketch",
    "random_projection",
    "bidiagonal_sign_subsample",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Phase 8.0 sketch-buffer overhead with Phase 7 "
            "theoretical skipped-KV bytes."
        )
    )
    parser.add_argument("--event-estimate", required=True)
    parser.add_argument("--sketch-overhead", required=True)
    parser.add_argument("--memory-comparison")
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase8_1_sketch_overhead_vs_savings.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase8_1_sketch_overhead_vs_savings.md",
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


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def affordability_classification(ratio: float) -> str:
    if ratio < 0:
        raise ValueError("overhead ratio must be non-negative")
    if ratio <= 0.05:
        return "excellent"
    if ratio <= 0.15:
        return "acceptable"
    if ratio <= 0.30:
        return "questionable"
    return "poor"


def compare_overhead_row(
    row: dict[str, Any],
    average_skipped_kv_bytes: float,
) -> dict[str, Any]:
    skipped_bytes = _number(
        average_skipped_kv_bytes,
        "average skipped KV bytes",
        positive=True,
    )
    sketch_bytes = _number(
        row.get("theoretical_sketch_bytes"),
        "theoretical sketch bytes",
    )
    if sketch_bytes < 0:
        raise ValueError("theoretical sketch bytes must be non-negative")
    ratio = sketch_bytes / skipped_bytes
    net_bytes = skipped_bytes - sketch_bytes
    classification = affordability_classification(ratio)
    return {
        "sketch_type": row.get("sketch_type"),
        "sketch_dim": row.get("sketch_dim"),
        "theoretical_sketch_bytes": int(sketch_bytes),
        "sketch_overhead_ratio_vs_full_kv": row.get(
            "sketch_overhead_ratio_vs_full_kv"
        ),
        "measured_allocated_delta_bytes": row.get(
            "measured_allocated_delta_bytes"
        ),
        "overhead_vs_avg_skipped_kv_ratio": ratio,
        "net_theoretical_savings_bytes": net_bytes,
        "net_theoretical_savings_ratio_vs_skipped": net_bytes / skipped_bytes,
        "affordability": classification,
        "overhead_affordable": classification in {"excellent", "acceptable"},
        "experimental_reference": row.get("sketch_type") == "srht",
    }


def summarize_event_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    aggregate = estimate.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("event estimate lacks an aggregate object")
    summary = {
        "bytes_per_block": _number(
            estimate.get("bytes_per_block"),
            "bytes per block",
            positive=True,
        ),
        "average_selected_blocks": aggregate.get("average_selected_blocks"),
        "average_skipped_blocks": aggregate.get("average_skipped_blocks"),
        "average_active_kv_bytes": aggregate.get("average_active_kv_bytes"),
        "average_skipped_kv_bytes": _number(
            aggregate.get("average_skipped_kv_bytes"),
            "average skipped KV bytes",
            positive=True,
        ),
        "average_estimated_reduction_ratio": aggregate.get(
            "average_estimated_reduction_ratio"
        ),
        "routing_event_count": aggregate.get(
            "estimated_routing_events",
            aggregate.get("total_routing_events"),
        ),
    }
    return summary


def _recommendations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [
        row
        for sketch_type in PREFERRED_RUNTIME_TYPES
        for row in rows
        if row["sketch_type"] == sketch_type and row["sketch_dim"] == 32
    ]
    affordable = [row for row in preferred if row["overhead_affordable"]]
    return {
        "preferred_dim32_configs": [
            {
                "sketch_type": row["sketch_type"],
                "sketch_dim": row["sketch_dim"],
                "affordability": row["affordability"],
                "overhead_vs_avg_skipped_kv_ratio": row[
                    "overhead_vs_avg_skipped_kv_ratio"
                ],
            }
            for row in preferred
        ],
        "affordable_preferred_configs": [
            {
                "sketch_type": row["sketch_type"],
                "sketch_dim": row["sketch_dim"],
            }
            for row in affordable
        ],
        "recommendation": (
            "Use CountSketch dim 32 and Random Projection dim 32 as simple "
            "overhead baselines. Keep bidiagonal_sign_subsample dim 32 as an "
            "experimental structured candidate. Do not use this report to "
            "enable active routing."
        ),
        "srht_policy": (
            "SRHT is reference/experimental only when present and is not a "
            "first runtime-overhead recommendation."
        ),
    }


def build_comparison(
    *,
    event_estimate_path: str | Path,
    sketch_overhead_path: str | Path,
    memory_comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    estimate = _load_json(event_estimate_path, "event estimate")
    overhead = _load_json(sketch_overhead_path, "sketch overhead")
    memory_comparison = (
        _load_json(memory_comparison_path, "memory comparison")
        if memory_comparison_path is not None
        else None
    )

    event_summary = summarize_event_estimate(estimate)
    overhead_rows = overhead.get("rows")
    if not isinstance(overhead_rows, list) or not overhead_rows:
        raise ValueError("sketch overhead JSON lacks non-empty rows")
    rows = [
        compare_overhead_row(
            row,
            event_summary["average_skipped_kv_bytes"],
        )
        for row in overhead_rows
        if isinstance(row, dict)
    ]
    if not rows:
        raise ValueError("sketch overhead JSON contains no valid row objects")

    measured_summary = None
    if memory_comparison is not None:
        conclusion = memory_comparison.get("conclusion", {})
        caveats = memory_comparison.get("caveats", {})
        measured_summary = {
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
        "event_estimate_summary": event_summary,
        "sketch_overhead_metadata": {
            "model_kv_metadata": overhead.get("model_kv_metadata"),
            "full_kv_bytes": overhead.get("full_kv_bytes"),
            "buffer_assumption": overhead.get("buffer_assumption"),
            "num_configurations": len(rows),
        },
        "comparison_rows": rows,
        "recommendations": _recommendations(rows),
        "measured_memory_context": measured_summary,
        "scope_warning": (
            "The sketch buffer may cover a configured physical-block pool, "
            "while skipped KV bytes are averaged per routing event. This is a "
            "planning ratio, not an accounting identity."
        ),
        "caveats": {
            "theoretical_only": True,
            "overhead_only": True,
            "replaces_full_kv": False,
            "measured_runtime_reduction": False,
            "active_routing": False,
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


def render_markdown(result: dict[str, Any]) -> str:
    event = result["event_estimate_summary"]
    lines = [
        "# Kivo-VD Phase 8.1 Sketch Overhead Vs Theoretical Savings",
        "",
        "**Status:** Theoretical-only comparison of additional sketch-buffer "
        "overhead with dry-run skipped-KV estimates.",
        "",
        "## Event Estimate Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    event_rows = [
        ("Bytes per KV block", event["bytes_per_block"]),
        ("Average selected blocks", event["average_selected_blocks"]),
        ("Average skipped blocks", event["average_skipped_blocks"]),
        ("Average active KV bytes", event["average_active_kv_bytes"]),
        ("Average skipped KV bytes", event["average_skipped_kv_bytes"]),
        (
            "Average estimated reduction ratio",
            event["average_estimated_reduction_ratio"],
        ),
        ("Routing events", event["routing_event_count"]),
    ]
    lines.extend(
        f"| {name} | `{_format(value)}` |" for name, value in event_rows
    )
    lines.extend([
        "",
        "## Overhead Vs Skipped KV",
        "",
        "| sketch | dim | sketch bytes | overhead/skipped | affordability | "
        "net theoretical bytes |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ])
    for row in result["comparison_rows"]:
        lines.append(
            "| "
            f"`{row['sketch_type']}` | `{row['sketch_dim']}` | "
            f"`{row['theoretical_sketch_bytes']}` | "
            f"`{row['overhead_vs_avg_skipped_kv_ratio']:.6f}` | "
            f"`{row['affordability']}` | "
            f"`{row['net_theoretical_savings_bytes']:.0f}` |"
        )

    lines.extend([
        "",
        "Affordability thresholds are planning heuristics only:",
        "",
        "- 5% or less of skipped KV: excellent;",
        "- above 5% through 15%: acceptable;",
        "- above 15% through 30%: questionable;",
        "- above 30%: poor.",
        "",
        "## Recommendation",
        "",
        result["recommendations"]["recommendation"],
        "",
        result["recommendations"]["srht_policy"],
        "",
        "## Interpretation",
        "",
        result["scope_warning"],
        "",
        "Sketch buffers remain additional memory because full KV is not reduced "
        "or replaced. A positive net theoretical value does not mean the "
        "runtime realized that saving.",
        "",
        "## Caveats",
        "",
        "- This comparison is theoretical only.",
        "- This phase measures overhead only.",
        "- Sketch buffers do not replace full KV.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency or quality claim follows from this report.",
        "",
        "## Next Steps",
        "",
        "- Run Phase 8.0 on CUDA and compare allocator deltas with payload bytes.",
        "- Keep CountSketch/RP dim 32 as simple overhead baselines.",
        "- Keep bidiagonal_sign_subsample dim 32 experimental.",
        "- Do not enable active routing from this result.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        args = _parse_args()
        result = build_comparison(
            event_estimate_path=args.event_estimate,
            sketch_overhead_path=args.sketch_overhead,
            memory_comparison_path=args.memory_comparison,
        )
        _write(
            args.output_json,
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(result))
        print(
            json.dumps(
                {
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "num_configurations": len(result["comparison_rows"]),
                    "recommendations": result["recommendations"],
                    "theoretical_only": True,
                    "overhead_only": True,
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
