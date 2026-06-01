#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
from pathlib import Path


def _load_sketch_math_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "vllm" / "v1" / "core" / "kivo_vd_sketch_math.py"
    spec = importlib.util.spec_from_file_location("kivo_vd_sketch_math", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline NumPy-only sketch eval for Kivo-VD."
    )
    parser.add_argument(
        "--sketch-type",
        choices=["random_projection", "count_sketch"],
        default="random_projection",
    )
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mode",
        choices=["gaussian", "clustered", "smooth_sequence", "needle_blocks", "mixed"],
        default="gaussian",
    )
    args = parser.parse_args()

    math = _load_sketch_math_module()

    keys, query = math.generate_synthetic_keys_and_query(
        num_tokens=args.num_tokens,
        input_dim=args.input_dim,
        seed=args.seed,
        mode=args.mode,
        block_size=args.block_size,
    )

    exact_scores = math.compute_exact_scores(query, keys)
    approx_scores = math.compute_sketched_scores(
        query=query,
        keys=keys,
        sketch_type=args.sketch_type,
        sketch_dim=args.sketch_dim,
        seed=args.seed,
    )

    exact_topk = math.topk_indices(exact_scores, args.topk)
    approx_topk = math.topk_indices(approx_scores, args.topk)
    token_recall = math.topk_recall(exact_topk, approx_topk)
    block_recall = math.topk_block_recall(
        exact_token_scores=exact_scores,
        approx_token_scores=approx_scores,
        block_size=args.block_size,
        k=max(1, args.topk // max(args.block_size, 1)),
        mode="max",
    )

    print(
        json.dumps(
            {
                "sketch_type": args.sketch_type,
                "seed": args.seed,
                "input_dim": args.input_dim,
                "sketch_dim": args.sketch_dim,
                "num_tokens": args.num_tokens,
                "block_size": args.block_size,
                "topk": args.topk,
                "mode": args.mode,
                "token_topk_recall": token_recall,
                "block_topk_recall": block_recall,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
