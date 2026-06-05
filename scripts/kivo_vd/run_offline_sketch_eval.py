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
        choices=[
            "random_projection",
            "count_sketch",
            "srht",
            "bidiagonal_sign",
            "bidiagonal_sign_subsample",
            "tridiagonal_sign",
        ],
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
    token_score_corr = math.pearson_correlation(exact_scores, approx_scores)

    block_k = max(1, args.topk // max(args.block_size, 1))
    exact_block_scores = math.block_scores_from_token_scores(
        exact_scores,
        args.block_size,
        mode="max",
    )
    approx_block_scores = math.block_scores_from_token_scores(
        approx_scores,
        args.block_size,
        mode="max",
    )
    block_score_corr = math.pearson_correlation(exact_block_scores, approx_block_scores)
    exact_top_blocks = math.topk_indices(exact_block_scores, block_k)
    approx_top_blocks = math.topk_indices(approx_block_scores, block_k)
    approx_block_ranking = math.topk_indices(approx_block_scores, approx_block_scores.shape[0])
    block_recall_2x = math.recall_at_budget(exact_top_blocks, approx_block_ranking, block_k * 2)
    block_recall_4x = math.recall_at_budget(exact_top_blocks, approx_block_ranking, block_k * 4)
    block_mrr = math.mean_reciprocal_rank(exact_top_blocks, approx_block_ranking)

    block_recall = math.topk_block_recall(
        exact_token_scores=exact_scores,
        approx_token_scores=approx_scores,
        block_size=args.block_size,
        k=block_k,
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
                "block_recall_at_2x_budget": block_recall_2x,
                "block_recall_at_4x_budget": block_recall_4x,
                "block_mrr": block_mrr,
                "token_score_correlation": token_score_corr,
                "block_score_correlation": block_score_corr,
                "exact_top_block_ids": exact_top_blocks.tolist(),
                "approx_top_block_ids": approx_top_blocks.tolist(),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
