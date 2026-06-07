#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare synthetic selected-KV materialization with prior accounting."""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Phase 9.0 selected-KV materialization with Phase 7 "
            "and optional Phase 8 accounting."
        )
    )
    parser.add_argument("--materialization", required=True)
    parser.add_argument("--event-estimate", required=True)
    parser.add_argument("--sketch-accounting")
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/"
            "phase9_1_selected_kv_materialization_comparison.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/"
            "phase9_1_selected_kv_materialization_comparison.md"
        ),
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


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _materialization_summary(
    materialization: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    aggregate = materialization.get("aggregate", {})
    rows = materialization.get("per_event_rows", [])
    device = materialization.get("device", {})
    caveats = materialization.get("caveats", {})
    if not isinstance(aggregate, dict):
        raise ValueError("materialization lacks an aggregate object")
    if not isinstance(rows, list):
        rows = []
    if not isinstance(device, dict):
        device = {}
    if not isinstance(caveats, dict):
        caveats = {}

    valid_rows = [row for row in rows if isinstance(row, dict)]
    full_bytes = [
        value
        for row in valid_rows
        if (
            value := _optional_number(
                row.get("full_considered_kv_bytes")
            )
        )
        is not None
    ]
    allocated_deltas = [
        value
        for row in valid_rows
        if (
            value := _optional_number(
                row.get("cuda_allocated_delta_bytes")
            )
        )
        is not None
    ]
    reserved_deltas = [
        value
        for row in valid_rows
        if (
            value := _optional_number(
                row.get("cuda_reserved_delta_bytes")
            )
        )
        is not None
    ]
    preview_rows_from_output = sum(
        row.get("selected_ids_preview_only") is True
        for row in valid_rows
    )
    preview_rows = int(
        _optional_number(aggregate.get("preview_only_event_count"))
        or preview_rows_from_output
    )
    full_id_rows = int(
        _optional_number(aggregate.get("full_block_ids_exported_count"))
        or 0
    )
    if preview_rows:
        warnings.append(
            f"{preview_rows} materialization rows use preview-only block IDs"
        )
    return {
        "num_events_processed": materialization.get(
            "num_events_processed"
        ),
        "average_selected_blocks": aggregate.get(
            "average_selected_blocks"
        ),
        "average_selected_kv_bytes": aggregate.get(
            "average_selected_kv_bytes"
        ),
        "total_selected_kv_bytes_materialized": aggregate.get(
            "total_selected_kv_bytes_materialized"
        ),
        "average_copy_time_ms": aggregate.get("average_copy_time_ms"),
        "p50_copy_time_ms": aggregate.get("p50_copy_time_ms"),
        "p90_copy_time_ms": aggregate.get("p90_copy_time_ms"),
        "max_copy_time_ms": aggregate.get("max_copy_time_ms"),
        "average_materialization_ratio": aggregate.get(
            "average_materialization_ratio"
        ),
        "average_full_considered_kv_bytes": _mean(full_bytes),
        "cuda_available": device.get("cuda_available"),
        "resolved_device": device.get("resolved"),
        "average_cuda_allocated_delta_bytes": _mean(allocated_deltas),
        "max_cuda_allocated_delta_bytes": (
            max(allocated_deltas) if allocated_deltas else None
        ),
        "average_cuda_reserved_delta_bytes": _mean(reserved_deltas),
        "max_cuda_reserved_delta_bytes": (
            max(reserved_deltas) if reserved_deltas else None
        ),
        "preview_only_event_count": preview_rows,
        "full_block_ids_exported_count": full_id_rows,
        "per_event_rows_truncated": materialization.get(
            "per_event_rows_truncated"
        ),
        "source_caveats": caveats,
    }


def _cumulative_skipped_bytes(
    estimate: dict[str, Any],
    warnings: list[str],
) -> tuple[float | None, str]:
    rows = estimate.get("per_event_estimates", [])
    aggregate = estimate.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    event_count = _optional_number(
        aggregate.get(
            "estimated_routing_events",
            aggregate.get("total_routing_events"),
        )
    )
    if isinstance(rows, list) and rows:
        values = [
            _optional_number(row.get("skipped_kv_bytes"))
            for row in rows
            if isinstance(row, dict)
        ]
        if (
            event_count is not None
            and len(values) == int(event_count)
            and all(value is not None for value in values)
        ):
            return sum(value for value in values if value is not None), (
                "per_event_sum"
            )
        warnings.append(
            "Phase 7 per-event rows are incomplete; cumulative skipped KV "
            "uses aggregate fallback"
        )
    average = _optional_number(aggregate.get("average_skipped_kv_bytes"))
    if average is not None and event_count is not None:
        return average * event_count, "average_times_event_count"
    warnings.append("cumulative skipped KV bytes are unavailable")
    return None, "unavailable"


def _event_estimate_summary(
    estimate: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    aggregate = estimate.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("event estimate lacks an aggregate object")
    cumulative, source = _cumulative_skipped_bytes(estimate, warnings)
    return {
        "bytes_per_block": estimate.get("bytes_per_block"),
        "average_selected_blocks": aggregate.get(
            "average_selected_blocks"
        ),
        "average_skipped_blocks": aggregate.get(
            "average_skipped_blocks"
        ),
        "average_active_kv_bytes": aggregate.get(
            "average_active_kv_bytes"
        ),
        "average_skipped_kv_bytes": aggregate.get(
            "average_skipped_kv_bytes"
        ),
        "cumulative_skipped_kv_bytes": cumulative,
        "cumulative_skipped_kv_source": source,
        "average_estimated_reduction_ratio": aggregate.get(
            "average_estimated_reduction_ratio"
        ),
        "routing_event_count": aggregate.get(
            "estimated_routing_events",
            aggregate.get("total_routing_events"),
        ),
    }


def _sketch_summary(
    accounting: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if accounting is None:
        return None
    rows = accounting.get("accounting_rows", [])
    recommendations = accounting.get("recommendations", {})
    if not isinstance(rows, list):
        rows = []
    if not isinstance(recommendations, dict):
        recommendations = {}
    configs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        global_pool = row.get("global_pool_model", {})
        cumulative = row.get("cumulative_request_model", {})
        break_even = row.get("break_even_model", {})
        if not isinstance(global_pool, dict):
            global_pool = {}
        if not isinstance(cumulative, dict):
            cumulative = {}
        if not isinstance(break_even, dict):
            break_even = {}
        configs.append({
            "sketch_type": row.get("sketch_type"),
            "sketch_dim": row.get("sketch_dim"),
            "sketch_pool_bytes": global_pool.get("sketch_pool_bytes"),
            "cumulative_overhead_ratio": cumulative.get(
                "overhead_vs_cumulative_skipped_kv"
            ),
            "cumulative_classification": cumulative.get("classification"),
            "break_even_events": break_even.get("break_even_events"),
            "break_even_skipped_blocks": break_even.get(
                "break_even_skipped_blocks"
            ),
        })
    return {
        "recommended_configs": recommendations.get("preferred_configs", []),
        "configurations": configs,
    }


def _safe_ratio(
    numerator: Any,
    denominator: Any,
) -> float | None:
    top = _optional_number(numerator)
    bottom = _optional_number(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return top / bottom


def _comparison_metrics(
    materialization: dict[str, Any],
    event: dict[str, Any],
    sketch: dict[str, Any] | None,
) -> dict[str, Any]:
    average_selected = materialization["average_selected_kv_bytes"]
    total_selected = materialization[
        "total_selected_kv_bytes_materialized"
    ]
    copy_ms = _optional_number(materialization["average_copy_time_ms"])
    selected_number = _optional_number(average_selected)
    throughput = (
        selected_number / (copy_ms / 1000.0)
        if selected_number is not None
        and copy_ms is not None
        and copy_ms > 0
        else None
    )
    sketch_rows: list[dict[str, Any]] = []
    if sketch is not None:
        for config in sketch["configurations"]:
            sketch_bytes = _optional_number(config.get("sketch_pool_bytes"))
            combined = (
                selected_number + sketch_bytes
                if selected_number is not None and sketch_bytes is not None
                else None
            )
            sketch_rows.append({
                **config,
                "average_selected_plus_sketch_bytes": combined,
                "selected_plus_sketch_vs_average_skipped_ratio": _safe_ratio(
                    combined,
                    event["average_skipped_kv_bytes"],
                ),
            })
    return {
        "selected_vs_full_considered_ratio": _safe_ratio(
            average_selected,
            materialization["average_full_considered_kv_bytes"],
        ),
        "selected_vs_skipped_ratio": _safe_ratio(
            average_selected,
            event["average_skipped_kv_bytes"],
        ),
        "cumulative_selected_vs_cumulative_skipped_ratio": _safe_ratio(
            total_selected,
            event["cumulative_skipped_kv_bytes"],
        ),
        "rough_copy_throughput_bytes_per_second": throughput,
        "selected_materialization_plus_sketch_overhead": sketch_rows,
    }


def _recommendation(
    materialization: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    ratio = metrics["selected_vs_full_considered_ratio"]
    copy_ms = _optional_number(materialization["average_copy_time_ms"])
    preview_only = materialization["preview_only_event_count"] > 0
    favorable_ratio = ratio is not None and ratio <= 0.50
    copy_time_available = copy_ms is not None
    proceed = favorable_ratio and copy_time_available
    if preview_only:
        message = (
            "Repeat Phase 9.0 with complete selected block IDs before using "
            "ratios for a Phase 9.2 repeated-run conclusion. Preview-only "
            "materialization undercounts payload."
        )
    elif proceed:
        message = (
            "The synthetic materialization ratio is low enough to justify "
            "Phase 9.2 repeated-run validation of copy cost and allocator "
            "variance. Do not enable active attention routing."
        )
    else:
        message = (
            "Keep Phase 9.2 deferred until complete ratios and copy timing "
            "are available or the materialization policy is reduced."
        )
    return {
        "phase9_2_repeated_run_recommended": proceed and not preview_only,
        "ratio_threshold_used": 0.50,
        "copy_time_available": copy_time_available,
        "preview_only_limits_conclusion": preview_only,
        "recommendation": message,
        "active_routing_recommended": False,
    }


def build_comparison(
    *,
    materialization_path: str | Path,
    event_estimate_path: str | Path,
    sketch_accounting_path: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    materialization_input = _load_json(
        materialization_path, "materialization"
    )
    event_input = _load_json(event_estimate_path, "event estimate")
    sketch_input = (
        _load_json(sketch_accounting_path, "sketch accounting")
        if sketch_accounting_path is not None
        else None
    )
    materialization = _materialization_summary(
        materialization_input, warnings
    )
    event = _event_estimate_summary(event_input, warnings)
    sketch = _sketch_summary(sketch_input)
    metrics = _comparison_metrics(materialization, event, sketch)
    return {
        "input_paths": {
            "materialization": str(materialization_path),
            "event_estimate": str(event_estimate_path),
            "sketch_accounting": (
                str(sketch_accounting_path)
                if sketch_accounting_path is not None
                else None
            ),
        },
        "materialization_summary": materialization,
        "event_estimate_summary": event,
        "sketch_accounting_summary": sketch,
        "comparison_metrics": metrics,
        "recommendations": _recommendation(materialization, metrics),
        "warnings": list(dict.fromkeys(warnings)),
        "caveats": {
            "synthetic_kv": True,
            "outside_attention_path": True,
            "full_kv_still_allocated": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "quality_not_measured": True,
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
    materialization = report["materialization_summary"]
    event = report["event_estimate_summary"]
    metrics = report["comparison_metrics"]
    preview_only_count = int(
        materialization.get("preview_only_event_count") or 0
    )
    if preview_only_count:
        export_note = (
            "Preview-only selected IDs undercount copied payload and block a "
            "strong Phase 9.2 conclusion until complete IDs are available."
        )
    else:
        export_note = (
            "Complete selected block IDs were exported; the copied payload "
            "reflects all selected blocks represented by the materialization "
            "events in this run."
        )
    lines = [
        "# Kivo-VD Phase 9.1 Materialization Comparison",
        "",
        "**Status:** Synthetic selected-KV comparison outside the attention "
        "path. Full KV remains allocated, active routing is disabled, and no "
        "runtime memory reduction or quality preservation is claimed.",
        "",
        "## Materialization",
        "",
    ]
    _append_table(
        lines,
        ["metric", "value"],
        [[key, value] for key, value in materialization.items()
         if key != "source_caveats"],
    )
    lines.extend(["", "## Phase 7 Event Estimate", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [[key, value] for key, value in event.items()],
    )
    lines.extend(["", "## Comparison", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [
            [key, value]
            for key, value in metrics.items()
            if key != "selected_materialization_plus_sketch_overhead"
        ],
    )
    sketch_rows = metrics[
        "selected_materialization_plus_sketch_overhead"
    ]
    if sketch_rows:
        lines.extend(["", "## Sketch-Buffer Context", ""])
        _append_table(
            lines,
            [
                "sketch",
                "dim",
                "sketch bytes",
                "selected + sketch",
                "combined / skipped",
                "cumulative class",
                "break-even events",
            ],
            [
                [
                    row["sketch_type"],
                    row["sketch_dim"],
                    row["sketch_pool_bytes"],
                    row["average_selected_plus_sketch_bytes"],
                    row[
                        "selected_plus_sketch_vs_average_skipped_ratio"
                    ],
                    row["cumulative_classification"],
                    row["break_even_events"],
                ]
                for row in sketch_rows
            ],
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        report["recommendations"]["recommendation"],
        "",
        "## Interpretation",
        "",
        "Ratios compare synthetic copied payload with theoretical Phase 7 KV "
        "opportunities. Copy throughput is a synchronized microbenchmark, "
        "not an end-to-end latency result.",
        "",
        export_note,
        "",
        "## Caveats",
        "",
        "- KV tensors are synthetic.",
        "- Materialization occurs outside the attention path.",
        "- Full KV is still allocated.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- Quality is not measured.",
        "- No latency improvement is claimed.",
    ])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend([
        "",
        "## Next Steps",
        "",
        "- Export or otherwise obtain complete selected block IDs.",
        "- Run repeated synthetic CUDA gathers to measure variance.",
        "- Keep real KV and attention behavior unchanged.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        args = _parse_args()
        report = build_comparison(
            materialization_path=args.materialization,
            event_estimate_path=args.event_estimate,
            sketch_accounting_path=args.sketch_accounting,
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
                    "comparison_metrics": report["comparison_metrics"],
                    "recommendations": report["recommendations"],
                    "warnings": report["warnings"],
                    "synthetic_kv": True,
                    "outside_attention_path": True,
                    "full_kv_still_allocated": True,
                    "active_routing": False,
                    "measured_runtime_reduction": False,
                    "quality_not_measured": True,
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
                    "synthetic_kv": True,
                    "outside_attention_path": True,
                    "full_kv_still_allocated": True,
                    "active_routing": False,
                    "measured_runtime_reduction": False,
                    "quality_not_measured": True,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
