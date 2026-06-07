#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Measure standalone compact sketch-buffer memory overhead."""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    try:
        values = [int(part) for part in _parse_csv(value)]
    except ValueError as exc:
        raise ValueError("sketch dimensions must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("sketch dimensions must contain positive integers")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure compact per-block Kivo sketch-buffer overhead."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--dtype-bytes", type=int, choices=[2, 4], default=2)
    parser.add_argument("--sketch-dims", default="16,32,64")
    parser.add_argument(
        "--sketch-types",
        default=(
            "count_sketch,random_projection,bidiagonal_sign_subsample"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase8_0_sketch_buffer_overhead.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase8_0_sketch_buffer_overhead.md",
    )
    return parser.parse_args(argv)


def full_kv_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    dtype_bytes: int,
) -> int:
    values = (
        num_layers,
        num_kv_heads,
        head_dim,
        block_size,
        num_blocks,
        dtype_bytes,
    )
    if any(value <= 0 for value in values):
        raise ValueError("full KV dimensions and dtype bytes must be positive")
    return (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * block_size
        * num_blocks
        * dtype_bytes
    )


def sketch_buffer_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    sketch_dim: int,
    num_blocks: int,
    dtype_bytes: int,
) -> int:
    values = (
        num_layers,
        num_kv_heads,
        sketch_dim,
        num_blocks,
        dtype_bytes,
    )
    if any(value <= 0 for value in values):
        raise ValueError("sketch dimensions and dtype bytes must be positive")
    return (
        num_layers
        * num_kv_heads
        * sketch_dim
        * num_blocks
        * dtype_bytes
    )


def sketch_overhead_ratio(sketch_bytes: int, kv_bytes: int) -> float:
    if sketch_bytes < 0 or kv_bytes <= 0:
        raise ValueError("byte counts must be non-negative and full KV positive")
    return sketch_bytes / kv_bytes


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def _resolve_dtype(torch: Any, dtype_bytes: int) -> Any:
    if dtype_bytes == 2:
        return torch.float16
    if dtype_bytes == 4:
        return torch.float32
    raise ValueError("--dtype-bytes must be 2 or 4")


def _cuda_checkpoint(torch: Any, name: str) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "name": name,
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_memory_bytes": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
    }


def _measure_one(
    *,
    torch: Any,
    device: Any,
    dtype: Any,
    sketch_type: str,
    sketch_dim: int,
    num_layers: int,
    num_kv_heads: int,
    num_blocks: int,
    theoretical_bytes: int,
    kv_bytes: int,
) -> dict[str, Any]:
    cuda_measurement = device.type == "cuda"
    if cuda_measurement:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    before = (
        _cuda_checkpoint(torch, "before_allocation")
        if cuda_measurement
        else None
    )
    shape = (num_layers, num_kv_heads, num_blocks, sketch_dim)
    buffer = torch.empty(shape, dtype=dtype, device=device)
    buffer.zero_()
    if cuda_measurement:
        torch.cuda.synchronize()
    tensor_bytes = buffer.numel() * buffer.element_size()
    after = (
        _cuda_checkpoint(torch, "after_allocation")
        if cuda_measurement
        else None
    )

    del buffer
    gc.collect()
    if cuda_measurement:
        torch.cuda.empty_cache()
    cleanup = (
        _cuda_checkpoint(torch, "after_cleanup")
        if cuda_measurement
        else None
    )

    allocated_delta = None
    reserved_delta = None
    if before is not None and after is not None:
        allocated_delta = (
            after["memory_allocated_bytes"]
            - before["memory_allocated_bytes"]
        )
        reserved_delta = (
            after["memory_reserved_bytes"]
            - before["memory_reserved_bytes"]
        )
    return {
        "sketch_type": sketch_type,
        "sketch_dim": sketch_dim,
        "sketch_buffer_shape": list(shape),
        "tensor_element_size_bytes": int(
            torch.empty((), dtype=dtype).element_size()
        ),
        "tensor_payload_bytes": int(tensor_bytes),
        "theoretical_sketch_bytes": theoretical_bytes,
        "sketch_overhead_ratio_vs_full_kv": sketch_overhead_ratio(
            theoretical_bytes, kv_bytes
        ),
        "measured_allocated_delta_bytes": allocated_delta,
        "measured_reserved_delta_bytes": reserved_delta,
        "cuda_memory_checkpoints": [
            checkpoint
            for checkpoint in (before, after, cleanup)
            if checkpoint is not None
        ],
    }


def build_report(
    *,
    model: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    dtype_bytes: int,
    sketch_types: list[str],
    sketch_dims: list[int],
    device_name: str,
    cuda_available: bool,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    kv_bytes = full_kv_bytes(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        dtype_bytes=dtype_bytes,
    )
    ratios = [row["sketch_overhead_ratio_vs_full_kv"] for row in rows]
    recommended = min(
        rows,
        key=lambda row: row["sketch_overhead_ratio_vs_full_kv"],
        default=None,
    )
    return {
        "model_kv_metadata": {
            "model": model,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "dtype_bytes": dtype_bytes,
        },
        "device": device_name,
        "cuda_available": cuda_available,
        "measured_cuda_available": (
            cuda_available and device_name == "cuda"
        ),
        "full_kv_bytes": kv_bytes,
        "sketch_types": sketch_types,
        "sketch_dims": sketch_dims,
        "rows": rows,
        "aggregate": {
            "num_configurations": len(rows),
            "min_overhead_ratio": min(ratios, default=None),
            "max_overhead_ratio": max(ratios, default=None),
            "recommended_small_config": (
                {
                    "sketch_type": recommended["sketch_type"],
                    "sketch_dim": recommended["sketch_dim"],
                    "overhead_ratio": recommended[
                        "sketch_overhead_ratio_vs_full_kv"
                    ],
                    "selection_basis": (
                        "smallest buffer payload only; not retrieval quality"
                    ),
                }
                if recommended is not None
                else None
            ),
        },
        "buffer_assumption": (
            "One compact sketch vector per layer, KV head, and physical block."
        ),
        "overhead_only": True,
        "replaces_full_kv": False,
        "active_routing": False,
        "measured_runtime_reduction": False,
    }


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["model_kv_metadata"]
    lines = [
        "# Kivo-VD Phase 8.0 Sketch-Buffer Overhead",
        "",
        "**Status:** Overhead only. The compact buffers are additional memory "
        "and do not replace full KV.",
        "",
        "## Model And KV Metadata",
        "",
        "| field | value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {key.replace('_', ' ')} | `{_format(value)}` |"
        for key, value in metadata.items()
    )
    lines.extend([
        f"| device | `{report['device']}` |",
        f"| CUDA available | `{_format(report['cuda_available'])}` |",
        f"| full KV bytes | `{report['full_kv_bytes']}` |",
        "",
        "## Sketch Overhead",
        "",
        "| sketch type | dim | shape | theoretical bytes | overhead ratio |",
        "| --- | ---: | --- | ---: | ---: |",
    ])
    for row in report["rows"]:
        lines.append(
            "| "
            f"`{row['sketch_type']}` | `{row['sketch_dim']}` | "
            f"`{row['sketch_buffer_shape']}` | "
            f"`{row['theoretical_sketch_bytes']}` | "
            f"`{row['sketch_overhead_ratio_vs_full_kv']:.6f}` |"
        )

    if report["measured_cuda_available"]:
        lines.extend([
            "",
            "## Measured CUDA Allocation",
            "",
            "| sketch type | dim | allocated delta | reserved delta |",
            "| --- | ---: | ---: | ---: |",
        ])
        for row in report["rows"]:
            lines.append(
                "| "
                f"`{row['sketch_type']}` | `{row['sketch_dim']}` | "
                f"`{_format(row['measured_allocated_delta_bytes'])}` | "
                f"`{_format(row['measured_reserved_delta_bytes'])}` |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Each tensor represents one compact sketch vector per layer, KV head, "
        "and physical block. The ratio compares this additional payload with "
        "theoretical full K+V storage for the same block count.",
        "",
        "Measured CUDA allocator deltas can exceed tensor payload bytes because "
        "of allocator granularity and caching. CPU runs validate shapes and "
        "formula bytes but do not report CUDA deltas.",
        "",
        "## Caveats",
        "",
        "- This is overhead only.",
        "- Sketch buffers do not replace full KV in Phase 8.0.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency or quality claim follows from this allocation test.",
        "- The current assumption is per-block sketches. A future per-token "
        "layout would use a different formula.",
        "",
        "## Next Steps",
        "",
        "- Run the same configurations on RunPod CUDA.",
        "- Compare measured allocator deltas with theoretical tensor payload.",
        "- Keep attention and KV allocation behavior unchanged.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to allocate sketch buffers; install it in the "
            "optional benchmark environment"
        ) from exc

    sketch_types = _parse_csv(args.sketch_types)
    sketch_dims = _parse_int_csv(args.sketch_dims)
    if not sketch_types:
        raise ValueError("--sketch-types must contain at least one type")
    kv_bytes = full_kv_bytes(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        dtype_bytes=args.dtype_bytes,
    )
    device = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype_bytes)
    rows = []
    for sketch_type in sketch_types:
        for sketch_dim in sketch_dims:
            theoretical_bytes = sketch_buffer_bytes(
                num_layers=args.num_layers,
                num_kv_heads=args.num_kv_heads,
                sketch_dim=sketch_dim,
                num_blocks=args.num_blocks,
                dtype_bytes=args.dtype_bytes,
            )
            rows.append(
                _measure_one(
                    torch=torch,
                    device=device,
                    dtype=dtype,
                    sketch_type=sketch_type,
                    sketch_dim=sketch_dim,
                    num_layers=args.num_layers,
                    num_kv_heads=args.num_kv_heads,
                    num_blocks=args.num_blocks,
                    theoretical_bytes=theoretical_bytes,
                    kv_bytes=kv_bytes,
                )
            )

    report = build_report(
        model=args.model,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        dtype_bytes=args.dtype_bytes,
        sketch_types=sketch_types,
        sketch_dims=sketch_dims,
        device_name=device.type,
        cuda_available=bool(torch.cuda.is_available()),
        rows=rows,
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
                "device": device.type,
                "num_configurations": len(rows),
                "min_overhead_ratio": report["aggregate"][
                    "min_overhead_ratio"
                ],
                "max_overhead_ratio": report["aggregate"][
                    "max_overhead_ratio"
                ],
                "overhead_only": True,
                "replaces_full_kv": False,
                "active_routing": False,
                "measured_runtime_reduction": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
