#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Estimate theoretical active KV bytes from Kivo dry-run routing events."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROUTING_EVENT_NAMES = {
    "dry_run_routing_decision",
    "dry-run-routing-decision",
}
PER_EVENT_OUTPUT_LIMIT = 100


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate theoretical active KV memory from Kivo dry-run events."
        )
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--memory-baseline")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/kivo_event_memory_estimate.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/kivo_event_memory_estimate.md",
    )
    return parser.parse_args(argv)


def bytes_per_kv_block(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype_bytes: int,
) -> int:
    values = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "dtype_bytes": dtype_bytes,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(
            "KV metadata values must be positive: " + ", ".join(invalid)
        )
    return (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * block_size
        * dtype_bytes
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"memory baseline file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"memory baseline must contain a JSON object: {path}")
    return value


def _read_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"event file is missing: {path}")

    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    with path.open("r", encoding="utf-8") as event_file:
        for line_number, line in enumerate(event_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(
                    f"ignored malformed JSONL row {line_number}: {exc}"
                )
                continue
            if not isinstance(row, dict):
                warnings.append(
                    f"ignored malformed JSONL row {line_number}: not an object"
                )
                continue
            events.append(row)
    return events, warnings


def _find_nested_value(value: Any, keys: tuple[str, ...]) -> Any | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if candidate is not None:
                return candidate
        for child in value.values():
            candidate = _find_nested_value(child, keys)
            if candidate is not None:
                return candidate
    elif isinstance(value, list):
        for child in value:
            candidate = _find_nested_value(child, keys)
            if candidate is not None:
                return candidate
    return None


def _resolve_metadata(
    *,
    model: str,
    num_layers: int | None,
    num_kv_heads: int | None,
    head_dim: int | None,
    block_size: int,
    dtype_bytes: int,
    memory_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    inferred_model = None
    if memory_baseline is not None:
        inferred_model = _find_nested_value(
            memory_baseline, ("model", "model_name")
        )
        num_layers = num_layers or _find_nested_value(
            memory_baseline,
            ("num_layers", "num_hidden_layers", "n_layer"),
        )
        num_kv_heads = num_kv_heads or _find_nested_value(
            memory_baseline,
            ("num_kv_heads", "num_key_value_heads"),
        )
        head_dim = head_dim or _find_nested_value(
            memory_baseline, ("head_dim", "head_size")
        )

    metadata = {
        "model": model or inferred_model,
        "num_layers": _optional_int(num_layers),
        "num_kv_heads": _optional_int(num_kv_heads),
        "head_dim": _optional_int(head_dim),
        "block_size": block_size,
        "dtype_bytes": dtype_bytes,
    }
    missing = [
        name
        for name in ("num_layers", "num_kv_heads", "head_dim")
        if metadata[name] is None
    ]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(
            "missing required KV metadata: "
            + ", ".join(missing)
            + f". Provide {flags}, or a memory baseline containing them."
        )
    return metadata


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_name(event: dict[str, Any]) -> str:
    return str(event.get("event_type", event.get("name", ""))).lower()


def _count_from_event(
    event: dict[str, Any],
    count_keys: tuple[str, ...],
    id_keys: tuple[str, ...],
) -> int | None:
    for key in count_keys:
        value = _optional_int(event.get(key))
        if value is not None:
            return value
    for key in id_keys:
        value = event.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _estimate_event(
    event: dict[str, Any],
    block_bytes: int,
) -> dict[str, Any]:
    selected = _count_from_event(
        event,
        ("selected_block_count", "selected_blocks"),
        ("selected_block_ids",),
    )
    recent = _count_from_event(
        event,
        ("recent_block_count", "recent_blocks"),
        ("recent_block_ids",),
    )
    skipped = _count_from_event(
        event,
        ("skipped_block_count", "skipped_blocks"),
        ("skipped_block_ids",),
    )
    if selected is None or skipped is None:
        raise ValueError(
            "routing event lacks selected/skipped block counts or complete ID lists"
        )
    if selected < 0 or skipped < 0 or (recent is not None and recent < 0):
        raise ValueError("routing event block counts must be non-negative")

    explicit_total = _count_from_event(
        event,
        ("total_considered_blocks", "total_block_count", "num_total_blocks"),
        ("all_block_ids",),
    )
    total = explicit_total if explicit_total is not None else selected + skipped
    if total < selected:
        raise ValueError("routing event total block count is below selected count")

    full_bytes = total * block_bytes
    active_bytes = selected * block_bytes
    skipped_bytes = max(total - selected, 0) * block_bytes
    reduction = skipped_bytes / full_bytes if full_bytes else 0.0
    return {
        "event_id": event.get("event_id"),
        "request_id": event.get("request_id"),
        "source": event.get("source"),
        "selected_blocks": selected,
        "recent_blocks": recent or 0,
        "skipped_blocks": skipped,
        "active_blocks": selected,
        "total_considered_blocks": total,
        "candidate_budget_blocks": event.get("candidate_budget_blocks"),
        "recent_window_blocks": event.get("recent_window_blocks"),
        "full_considered_kv_bytes": full_bytes,
        "active_kv_bytes": active_bytes,
        "skipped_kv_bytes": skipped_bytes,
        "estimated_reduction_ratio": reduction,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def estimate_memory(
    *,
    events_path: str | Path,
    memory_baseline_path: str | Path | None,
    model: str,
    num_layers: int | None,
    num_kv_heads: int | None,
    head_dim: int | None,
    block_size: int,
    dtype_bytes: int,
) -> dict[str, Any]:
    baseline = (
        _read_json(Path(memory_baseline_path))
        if memory_baseline_path is not None
        else None
    )
    metadata = _resolve_metadata(
        model=model,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype_bytes=dtype_bytes,
        memory_baseline=baseline,
    )
    block_bytes = bytes_per_kv_block(
        num_layers=metadata["num_layers"],
        num_kv_heads=metadata["num_kv_heads"],
        head_dim=metadata["head_dim"],
        block_size=metadata["block_size"],
        dtype_bytes=metadata["dtype_bytes"],
    )

    events, warnings = _read_events(Path(events_path))
    routing_events = [
        event for event in events if _event_name(event) in ROUTING_EVENT_NAMES
    ]
    estimates: list[dict[str, Any]] = []
    for index, event in enumerate(routing_events, start=1):
        try:
            estimates.append(_estimate_event(event, block_bytes))
        except ValueError as exc:
            warnings.append(f"ignored routing event {index}: {exc}")

    if not routing_events:
        warnings.append("no dry_run_routing_decision events found")
    if routing_events and not estimates:
        warnings.append("no routing events contained enough block metadata")

    selected = [float(row["selected_blocks"]) for row in estimates]
    recent = [float(row["recent_blocks"]) for row in estimates]
    skipped = [float(row["skipped_blocks"]) for row in estimates]
    active_bytes = [float(row["active_kv_bytes"]) for row in estimates]
    skipped_bytes = [float(row["skipped_kv_bytes"]) for row in estimates]
    reductions = [
        float(row["estimated_reduction_ratio"]) for row in estimates
    ]
    aggregate = {
        "total_routing_events": len(routing_events),
        "estimated_routing_events": len(estimates),
        "average_selected_blocks": _mean(selected),
        "average_recent_blocks": _mean(recent),
        "average_skipped_blocks": _mean(skipped),
        "average_active_kv_bytes": _mean(active_bytes),
        "average_skipped_kv_bytes": _mean(skipped_bytes),
        "average_estimated_reduction_ratio": _mean(reductions),
        "p50_selected_blocks": _percentile(selected, 50),
        "p90_selected_blocks": _percentile(selected, 90),
        "max_selected_blocks": max(selected, default=None),
        "p50_estimated_reduction_ratio": _percentile(reductions, 50),
        "p90_estimated_reduction_ratio": _percentile(reductions, 90),
        "max_estimated_reduction_ratio": max(reductions, default=None),
        "request_ids_seen": sorted({
            str(row["request_id"])
            for row in estimates
            if row["request_id"] is not None
        }),
        "sources_seen": sorted({
            str(row["source"])
            for row in estimates
            if row["source"] is not None
        }),
    }
    if len(estimates) > PER_EVENT_OUTPUT_LIMIT:
        warnings.append(
            f"per-event output capped at {PER_EVENT_OUTPUT_LIMIT} estimates"
        )

    return {
        "input_paths": {
            "events": str(events_path),
            "memory_baseline": (
                str(memory_baseline_path)
                if memory_baseline_path is not None
                else None
            ),
        },
        "model_kv_metadata": metadata,
        "bytes_per_block": block_bytes,
        "aggregate": aggregate,
        "per_event_estimates": estimates[:PER_EVENT_OUTPUT_LIMIT],
        "warnings": warnings,
        "estimated_only": True,
        "measured_runtime_reduction": False,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    metadata = result["model_kv_metadata"]
    aggregate = result["aggregate"]
    metadata_rows = [
        ("Model", metadata["model"]),
        ("Layers", metadata["num_layers"]),
        ("KV heads", metadata["num_kv_heads"]),
        ("Head dimension", metadata["head_dim"]),
        ("Block size", metadata["block_size"]),
        ("Bytes per dtype element", metadata["dtype_bytes"]),
        ("Estimated bytes per KV block", result["bytes_per_block"]),
    ]
    aggregate_rows = [
        ("Routing events", aggregate["total_routing_events"]),
        ("Estimated routing events", aggregate["estimated_routing_events"]),
        ("Average selected blocks", aggregate["average_selected_blocks"]),
        ("Average recent blocks", aggregate["average_recent_blocks"]),
        ("Average skipped blocks", aggregate["average_skipped_blocks"]),
        ("Average active KV bytes", aggregate["average_active_kv_bytes"]),
        ("Average skipped KV bytes", aggregate["average_skipped_kv_bytes"]),
        (
            "Average estimated reduction ratio",
            aggregate["average_estimated_reduction_ratio"],
        ),
        ("P50 selected blocks", aggregate["p50_selected_blocks"]),
        ("P90 selected blocks", aggregate["p90_selected_blocks"]),
        (
            "P90 estimated reduction ratio",
            aggregate["p90_estimated_reduction_ratio"],
        ),
    ]

    lines = [
        "# Kivo-VD Dry-Run Event Memory Estimate",
        "",
        "**Status:** Estimated-only active-KV accounting. This is not measured "
        "runtime memory reduction.",
        "",
        "## Model And KV Metadata",
        "",
        "| field | value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name} | `{_format_value(value)}` |"
        for name, value in metadata_rows
    )
    lines.extend([
        "",
        "## Aggregate Estimate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ])
    lines.extend(
        f"| {name} | `{_format_value(value)}` |"
        for name, value in aggregate_rows
    )
    lines.extend([
        "",
        "## Proven Vs Not Proven",
        "",
        "Proven by this artifact:",
        "",
        "- exported dry-run routing counts can be converted into theoretical KV "
        "byte accounting;",
        "- the accounting uses explicit model/KV dimensions and the K+V factor.",
        "",
        "Not proven by this artifact:",
        "",
        "- measured vLLM runtime memory reduction;",
        "- active KV residency or candidate-routed attention;",
        "- latency improvement or quality preservation.",
        "",
        "## Caveats",
        "",
        "- vLLM still allocates and attends over the normal/full KV cache.",
        "- Selected blocks are treated as the active union; recent blocks are a "
        "diagnostic subset and are not added twice.",
        "- The estimate excludes allocator overhead, fragmentation, block "
        "metadata, sketches, and backend-specific layout padding.",
        "- Phase 7.0 CUDA checkpoints are complementary measured baselines, not "
        "evidence that this theoretical reduction occurred.",
    ])
    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return "\n".join(lines) + "\n"


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        args = _parse_args()
        result = estimate_memory(
            events_path=args.events,
            memory_baseline_path=args.memory_baseline,
            model=args.model,
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            dtype_bytes=args.dtype_bytes,
        )
        _write_text(
            args.output_json,
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
        _write_text(args.output_md, render_markdown(result))
        summary = {
            "output_json": args.output_json,
            "output_md": args.output_md,
            "bytes_per_block": result["bytes_per_block"],
            "aggregate": result["aggregate"],
            "estimated_only": True,
            "measured_runtime_reduction": False,
            "warnings": result["warnings"],
        }
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "estimated_only": True,
                    "measured_runtime_reduction": False,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
