#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm.v1.core.kivo_vd_torch_sketch_backend import make_torch_sketch_backend


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def _dtype_from_string(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype {value!r}; choose one of {sorted(mapping)}")
    return mapping[value]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _time_ms(fn, device: torch.device) -> tuple[float, Any]:
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0, out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline torch benchmark for Kivo-VD sketch backends."
    )
    parser.add_argument(
        "--sketch-types", default="count_sketch,random_projection"
    )
    parser.add_argument("--sketch-dims", default="32,64,128")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--output", default="outputs/kivo_vd/torch_sketch_benchmark.jsonl"
    )
    return parser.parse_args()


def _run_one(
    *,
    sketch_type: str,
    sketch_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    backend = make_torch_sketch_backend(
        sketch_type=sketch_type,
        input_dim=args.head_dim,
        sketch_dim=sketch_dim,
        seed=0,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    keys = torch.randn(
        (args.num_tokens, args.head_dim),
        generator=generator,
        dtype=dtype,
        device="cpu",
    ).to(device)
    queries = torch.randn(
        (args.num_queries, args.head_dim),
        generator=generator,
        dtype=dtype,
        device="cpu",
    ).to(device)

    key_times: list[float] = []
    query_times: list[float] = []
    score_times: list[float] = []
    total_times: list[float] = []
    block_sketches = None

    def key_step():
        key_sketches = backend.sketch_keys(keys)
        return backend.block_sketches_from_key_sketches(
            key_sketches, args.block_size
        )

    def query_step():
        return torch.stack([backend.sketch_query(query) for query in queries])

    def score_step(qs, bs):
        scores = torch.stack([backend.score_blocks(q, bs) for q in qs])
        ranks = torch.stack([backend.rank_blocks(row) for row in scores])
        return scores, ranks

    for _ in range(args.warmup):
        block_sketches = key_step()
        query_sketches = query_step()
        score_step(query_sketches, block_sketches)
    _sync(device)

    for _ in range(args.iters):
        total_start = time.perf_counter()
        key_ms, block_sketches = _time_ms(key_step, device)
        query_ms, query_sketches = _time_ms(query_step, device)
        score_ms, _ = _time_ms(
            lambda: score_step(query_sketches, block_sketches), device
        )
        _sync(device)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        key_times.append(key_ms)
        query_times.append(query_ms)
        score_times.append(score_ms)
        total_times.append(total_ms)

    assert block_sketches is not None
    element_size = torch.empty((), dtype=dtype).element_size()
    full_k_bytes = args.num_tokens * args.head_dim * element_size
    sketch_k_bytes = block_sketches.shape[0] * sketch_dim * element_size

    return {
        "sketch_type": sketch_type,
        "sketch_dim": sketch_dim,
        "num_tokens": args.num_tokens,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "num_blocks": int(block_sketches.shape[0]),
        "num_queries": args.num_queries,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": args.warmup,
        "iters": args.iters,
        "key_sketch_build_time_ms": sum(key_times) / len(key_times),
        "query_sketch_time_ms": sum(query_times) / len(query_times),
        "block_scoring_time_ms": sum(score_times) / len(score_times),
        "total_time_ms": sum(total_times) / len(total_times),
        "full_k_memory_bytes": full_k_bytes,
        "sketch_k_memory_bytes": sketch_k_bytes,
        "sketch_memory_ratio": sketch_k_bytes / full_k_bytes,
    }


def main() -> int:
    args = _parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    device = torch.device(args.device)
    dtype = _dtype_from_string(args.dtype)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with output_path.open("w", encoding="utf-8") as f:
        for sketch_type in _parse_csv(args.sketch_types):
            for sketch_dim in _parse_int_csv(args.sketch_dims):
                row = _run_one(
                    sketch_type=sketch_type,
                    sketch_dim=sketch_dim,
                    args=args,
                    device=device,
                    dtype=dtype,
                )
                rows.append(row)
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sketch_type"]].append(row)

    summary = {
        "output": str(output_path),
        "num_rows": len(rows),
        "avg_total_time_ms_by_type": {
            key: sum(r["total_time_ms"] for r in vals) / len(vals)
            for key, vals in grouped.items()
        },
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
