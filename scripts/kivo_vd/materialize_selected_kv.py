#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Materialize synthetic selected KV blocks outside the attention path."""

import argparse
import gc
import json
import math
import statistics
import sys
import time
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
            "Materialize synthetic selected KV blocks from Kivo dry-run "
            "routing events outside attention."
        )
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype-bytes", type=int, choices=[2, 4], default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--num-pool-blocks", type=int)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/"
            "phase9_0_selected_kv_materialization.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/"
            "phase9_0_selected_kv_materialization.md"
        ),
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
    values = (
        num_layers,
        num_kv_heads,
        head_dim,
        block_size,
        dtype_bytes,
    )
    if any(value <= 0 for value in values):
        raise ValueError("KV metadata values must be positive")
    return (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * block_size
        * dtype_bytes
    )


def read_routing_events(
    path: str | Path,
    max_events: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"event file is missing: {input_path}")
    if max_events is not None and max_events <= 0:
        raise ValueError("--max-events must be positive")

    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    with input_path.open("r", encoding="utf-8") as event_file:
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
            name = str(
                row.get("event_type", row.get("name", ""))
            ).lower()
            if name not in ROUTING_EVENT_NAMES:
                continue
            events.append(row)
            if max_events is not None and len(events) >= max_events:
                break
    if not events:
        warnings.append("no dry_run_routing_decision events found")
    return events, warnings


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _block_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    result: list[int] = []
    seen: set[int] = set()
    for item in value:
        block_id = _optional_int(item)
        if block_id is None or block_id < 0:
            return None
        if block_id not in seen:
            seen.add(block_id)
            result.append(block_id)
    return result


def extract_selected_blocks(
    event: dict[str, Any],
) -> tuple[list[int], int | None, bool, str | None]:
    requested_count = _optional_int(event.get("selected_block_count"))
    selected_ids = _block_ids(event.get("selected_block_ids_full"))
    if selected_ids is None:
        selected_ids = _block_ids(event.get("selected_block_ids"))
    preview_only = False
    warning = None
    if selected_ids is None:
        selected_ids = _block_ids(event.get("selected_block_preview"))
        preview_only = selected_ids is not None
    if selected_ids is None:
        return [], requested_count, False, (
            "routing event lacks selected block IDs and preview IDs"
        )
    if requested_count is None:
        requested_count = len(selected_ids)
    if requested_count < len(selected_ids):
        warning = (
            "selected block count is smaller than the available ID list; "
            "using the ID list length"
        )
        requested_count = len(selected_ids)
    elif preview_only and requested_count > len(selected_ids):
        warning = (
            "selected block IDs are preview-only; materialized only the "
            f"{len(selected_ids)} exported IDs out of {requested_count}"
        )
    return selected_ids, requested_count, preview_only, warning


def _skipped_count(event: dict[str, Any]) -> int | None:
    count = _optional_int(event.get("skipped_block_count"))
    if count is not None:
        return count
    ids = _block_ids(event.get("skipped_block_ids"))
    if ids is not None:
        return len(ids)
    return None


def infer_num_pool_blocks(
    selected_ids_by_event: list[list[int]],
    requested: int | None,
    fallback: int = 256,
) -> int:
    if requested is not None and requested <= 0:
        raise ValueError("--num-pool-blocks must be positive")
    max_id = max(
        (block_id for ids in selected_ids_by_event for block_id in ids),
        default=-1,
    )
    if requested is not None:
        if max_id >= requested:
            raise ValueError(
                "--num-pool-blocks is smaller than an exported block ID"
            )
        return requested
    if max_id >= 0:
        return max_id + 1
    return fallback


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def _resolve_dtype(
    torch: Any,
    device: Any,
    dtype_bytes: int,
    warnings: list[str],
) -> Any:
    if dtype_bytes == 4:
        return torch.float32
    dtype = torch.float16
    try:
        probe = torch.empty((1,), device=device, dtype=dtype)
        probe.index_select(
            0, torch.tensor([0], device=device, dtype=torch.long)
        )
    except RuntimeError:
        warnings.append(
            "float16 index_select is unsupported on this device; using "
            "float32 tensors while retaining configured byte accounting"
        )
        dtype = torch.float32
    return dtype


def _sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _cuda_memory(torch: Any, device: Any) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(index, 0)]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        float(row["materialized_selected_block_count"]) for row in rows
    ]
    requested = [
        float(row["requested_selected_block_count"])
        for row in rows
        if row["requested_selected_block_count"] is not None
    ]
    selected_bytes = [float(row["selected_kv_bytes"]) for row in rows]
    copy_times = [float(row["copy_time_ms"]) for row in rows]
    ratios = [
        float(row["materialization_ratio"])
        for row in rows
        if row["materialization_ratio"] is not None
    ]
    return {
        "average_selected_blocks": _mean(selected),
        "average_requested_selected_blocks": _mean(requested),
        "average_materialized_selected_blocks": _mean(selected),
        "average_selected_kv_bytes": _mean(selected_bytes),
        "average_copy_time_ms": _mean(copy_times),
        "p50_copy_time_ms": _percentile(copy_times, 50),
        "p90_copy_time_ms": _percentile(copy_times, 90),
        "max_copy_time_ms": max(copy_times) if copy_times else None,
        "average_materialization_ratio": _mean(ratios),
        "total_selected_kv_bytes_materialized": int(sum(selected_bytes)),
        "full_block_ids_exported_count": sum(
            row["full_block_ids_exported"] is True for row in rows
        ),
        "preview_only_event_count": sum(
            row["selected_ids_preview_only"] is True for row in rows
        ),
    }


def materialize_selected_kv(
    *,
    events_path: str | Path,
    model: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype_bytes: int,
    device_name: str,
    max_events: int | None,
    num_pool_blocks: int | None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for selected-KV materialization"
        ) from exc

    warnings: list[str] = []
    events, event_warnings = read_routing_events(events_path, max_events)
    warnings.extend(event_warnings)
    parsed: list[tuple[dict[str, Any], list[int], int | None, bool]] = []
    for event in events:
        selected_ids, requested_count, preview_only, warning = (
            extract_selected_blocks(event)
        )
        if warning is not None:
            warnings.append(
                f"event {event.get('event_id', 'unknown')}: {warning}"
            )
        if not selected_ids:
            continue
        parsed.append(
            (event, selected_ids, requested_count, preview_only)
        )

    block_bytes = bytes_per_kv_block(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype_bytes=dtype_bytes,
    )
    resolved_pool_blocks = infer_num_pool_blocks(
        [item[1] for item in parsed],
        num_pool_blocks,
    )
    device = _resolve_device(torch, device_name)
    dtype = _resolve_dtype(torch, device, dtype_bytes, warnings)
    rows: list[dict[str, Any]] = []

    if parsed:
        shape = (
            num_layers,
            num_kv_heads,
            resolved_pool_blocks,
            block_size,
            head_dim,
        )
        full_k = torch.empty(shape, dtype=dtype, device=device)
        full_v = torch.empty(shape, dtype=dtype, device=device)
        full_k.zero_()
        full_v.zero_()
        _sync(torch, device)

        for event, selected_ids, requested_count, preview_only in parsed:
            indices = torch.tensor(
                selected_ids,
                device=device,
                dtype=torch.long,
            )
            before_allocated, before_reserved = _cuda_memory(
                torch, device
            )
            _sync(torch, device)
            started = time.perf_counter()
            selected_k = full_k.index_select(2, indices)
            selected_v = full_v.index_select(2, indices)
            _sync(torch, device)
            copy_time_ms = (time.perf_counter() - started) * 1000.0
            after_allocated, after_reserved = _cuda_memory(torch, device)

            materialized_count = len(selected_ids)
            skipped_count = _skipped_count(event)
            full_considered_count = (
                requested_count + skipped_count
                if requested_count is not None
                and skipped_count is not None
                else None
            )
            selected_bytes = materialized_count * block_bytes
            full_considered_bytes = (
                full_considered_count * block_bytes
                if full_considered_count is not None
                else None
            )
            ratio = (
                selected_bytes / full_considered_bytes
                if full_considered_bytes
                else None
            )
            rows.append({
                "event_id": event.get("event_id"),
                "request_id": event.get("request_id"),
                "source": event.get("source"),
                "selected_block_ids": selected_ids,
                "full_block_ids_exported": (
                    event.get("full_block_ids_exported") is True
                    or event.get("selected_block_ids_full") is not None
                ),
                "selected_ids_preview_only": preview_only,
                "requested_selected_block_count": requested_count,
                "materialized_selected_block_count": materialized_count,
                "skipped_block_count": skipped_count,
                "full_considered_block_count": full_considered_count,
                "selected_kv_bytes": selected_bytes,
                "full_considered_kv_bytes": full_considered_bytes,
                "materialization_ratio": ratio,
                "copy_time_ms": copy_time_ms,
                "cuda_allocated_delta_bytes": (
                    after_allocated - before_allocated
                    if after_allocated is not None
                    and before_allocated is not None
                    else None
                ),
                "cuda_reserved_delta_bytes": (
                    after_reserved - before_reserved
                    if after_reserved is not None
                    and before_reserved is not None
                    else None
                ),
            })
            del selected_k, selected_v, indices
        del full_k, full_v
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "input_path": str(events_path),
        "model_kv_metadata": {
            "model": model,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "configured_dtype_bytes": dtype_bytes,
            "effective_tensor_dtype": str(dtype),
            "num_pool_blocks": resolved_pool_blocks,
            "synthetic_k_shape": [
                num_layers,
                num_kv_heads,
                resolved_pool_blocks,
                block_size,
                head_dim,
            ],
        },
        "device": {
            "requested": device_name,
            "resolved": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "bytes_per_block": block_bytes,
        "num_routing_events_read": len(events),
        "num_events_processed": len(rows),
        "aggregate": aggregate_rows(rows),
        "per_event_rows": rows[:PER_EVENT_OUTPUT_LIMIT],
        "per_event_rows_truncated": len(rows) > PER_EVENT_OUTPUT_LIMIT,
        "warnings": list(dict.fromkeys(warnings)),
        "caveats": {
            "synthetic_kv": True,
            "outside_attention_path": True,
            "full_kv_still_allocated": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
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
    metadata = report["model_kv_metadata"]
    aggregate = report["aggregate"]
    lines = [
        "# Kivo-VD Phase 9.0 Selected-KV Materialization",
        "",
        "**Status:** Synthetic KV gather/copy outside the attention path. "
        "Full KV remains allocated, active routing is disabled, and no "
        "runtime memory reduction is claimed.",
        "",
        "## Model And KV Metadata",
        "",
    ]
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in metadata.items()],
    )
    lines.extend(["", "## Aggregate Materialization", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [[key, value] for key, value in aggregate.items()],
    )
    lines.extend(["", "## Per-Event Preview", ""])
    _append_table(
        lines,
        [
            "event",
            "requested",
            "materialized",
            "skipped",
            "selected bytes",
            "ratio",
            "copy ms",
            "preview only",
        ],
        [
            [
                row["event_id"],
                row["requested_selected_block_count"],
                row["materialized_selected_block_count"],
                row["skipped_block_count"],
                row["selected_kv_bytes"],
                row["materialization_ratio"],
                row["copy_time_ms"],
                row["selected_ids_preview_only"],
            ]
            for row in report["per_event_rows"]
        ],
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Copy time measures one synthetic gather per routing event. The "
        "materialization ratio compares materialized preview/full selected "
        "IDs with the event's selected-plus-skipped count when available.",
        "",
        "Preview-only events undercount temporary payload because the current "
        "runtime export caps block-ID previews. They are marked explicitly.",
        "",
        "## Caveats",
        "",
        "- KV tensors are synthetic.",
        "- Materialization occurs outside the attention path.",
        "- Full KV is still allocated.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- Copy timing is not an end-to-end latency claim.",
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
        report = materialize_selected_kv(
            events_path=args.events,
            model=args.model,
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            dtype_bytes=args.dtype_bytes,
            device_name=args.device,
            max_events=args.max_events,
            num_pool_blocks=args.num_pool_blocks,
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
                    "num_events_processed": report[
                        "num_events_processed"
                    ],
                    "average_copy_time_ms": report["aggregate"][
                        "average_copy_time_ms"
                    ],
                    "warnings": report["warnings"],
                    "synthetic_kv": True,
                    "outside_attention_path": True,
                    "full_kv_still_allocated": True,
                    "active_routing": False,
                    "measured_runtime_reduction": False,
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
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
