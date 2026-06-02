#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_sketch_math_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "vllm" / "v1" / "core" / "kivo_vd_sketch_math.py"
    spec = importlib.util.spec_from_file_location("kivo_vd_sketch_math", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _srht_sketch_dim_is_valid(input_dim: int, sketch_dim: int) -> bool:
    padded_dim = 1 << (input_dim - 1).bit_length()
    return sketch_dim <= padded_dim


def _run_one(
    math: Any,
    sketch_type: str,
    sketch_dim: int,
    num_tokens: int,
    input_dim: int,
    block_size: int,
    topk_blocks: int,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    keys, query = math.generate_synthetic_keys_and_query(
        num_tokens=num_tokens,
        input_dim=input_dim,
        seed=seed,
        mode=mode,
        block_size=block_size,
    )

    t0 = time.perf_counter()
    exact_scores = math.compute_exact_scores(query, keys)
    approx_scores = math.compute_sketched_scores(
        query=query,
        keys=keys,
        sketch_type=sketch_type,
        sketch_dim=sketch_dim,
        seed=seed,
    )
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    token_k = min(num_tokens, max(1, topk_blocks * block_size))
    exact_topk_tokens = math.topk_indices(exact_scores, token_k)
    approx_topk_tokens = math.topk_indices(approx_scores, token_k)
    token_recall = math.topk_recall(exact_topk_tokens, approx_topk_tokens)
    token_score_corr = math.pearson_correlation(exact_scores, approx_scores)

    exact_block_scores = math.block_scores_from_token_scores(
        exact_scores, block_size, mode="max"
    )
    approx_block_scores = math.block_scores_from_token_scores(
        approx_scores, block_size, mode="max"
    )
    block_score_corr = math.pearson_correlation(exact_block_scores, approx_block_scores)
    exact_top_blocks = math.topk_indices(exact_block_scores, topk_blocks)
    approx_top_blocks = math.topk_indices(approx_block_scores, topk_blocks)
    approx_block_ranking = math.topk_indices(approx_block_scores, approx_block_scores.shape[0])
    block_recall = math.topk_recall(exact_top_blocks, approx_top_blocks)
    block_recall_2x = math.recall_at_budget(
        exact_top_blocks,
        approx_block_ranking,
        topk_blocks * 2,
    )
    block_recall_4x = math.recall_at_budget(
        exact_top_blocks,
        approx_block_ranking,
        topk_blocks * 4,
    )
    block_mrr = math.mean_reciprocal_rank(exact_top_blocks, approx_block_ranking)

    return {
        "sketch_type": sketch_type,
        "sketch_dim": sketch_dim,
        "num_tokens": num_tokens,
        "input_dim": input_dim,
        "block_size": block_size,
        "topk_blocks": topk_blocks,
        "seed": seed,
        "mode": mode,
        "token_topk_recall": float(token_recall),
        "block_topk_recall": float(block_recall),
        "block_recall_at_2x_budget": float(block_recall_2x),
        "block_recall_at_4x_budget": float(block_recall_4x),
        "block_mrr": float(block_mrr),
        "token_score_correlation": float(token_score_corr),
        "block_score_correlation": float(block_score_corr),
        "exact_top_block_ids": exact_top_blocks.tolist(),
        "approx_top_block_ids": approx_top_blocks.tolist(),
        "runtime_ms": float(runtime_ms),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Kivo-VD sketch sweep.")
    parser.add_argument(
        "--output",
        default="outputs/kivo_vd/sketch_sweep.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny sweep for smoke testing.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed override.")
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="Single num_tokens override.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=128,
        help="Input hidden dimension.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size for block-level recall.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    math = _load_sketch_math_module()

    if args.quick:
        sketch_types = ["random_projection", "count_sketch", "srht"]
        sketch_dims = [16, 64]
        num_tokens_list = [1024]
        topk_blocks_list = [4, 8]
        seeds = [0]
        modes = ["gaussian", "clustered", "smooth_sequence", "needle_blocks", "mixed"]
    else:
        sketch_types = ["random_projection", "count_sketch", "srht"]
        sketch_dims = [16, 32, 64, 128]
        num_tokens_list = [1024, 4096, 8192]
        topk_blocks_list = [4, 8, 16]
        seeds = [0, 1, 2]
        modes = ["gaussian", "clustered", "smooth_sequence", "needle_blocks", "mixed"]

    if args.seed is not None:
        seeds = [args.seed]
    if args.num_tokens is not None:
        num_tokens_list = [args.num_tokens]

    rows = []
    for mode in modes:
        for sketch_type in sketch_types:
            for sketch_dim in sketch_dims:
                for num_tokens in num_tokens_list:
                    for topk_blocks in topk_blocks_list:
                        for seed in seeds:
                            if sketch_type == "srht" and not (
                                _srht_sketch_dim_is_valid(
                                    input_dim=args.input_dim,
                                    sketch_dim=sketch_dim,
                                )
                            ):
                                continue
                            rows.append(
                                _run_one(
                                    math=math,
                                    sketch_type=sketch_type,
                                    sketch_dim=sketch_dim,
                                    num_tokens=num_tokens,
                                    input_dim=args.input_dim,
                                    block_size=args.block_size,
                                    topk_blocks=topk_blocks,
                                    seed=seed,
                                    mode=mode,
                                )
                            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    grouped: defaultdict[tuple[str, str, int, int], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = (row["mode"], row["sketch_type"], row["sketch_dim"], row["topk_blocks"])
        grouped[key].append(
            {
                "block_topk_recall": row["block_topk_recall"],
                "block_recall_at_2x_budget": row["block_recall_at_2x_budget"],
                "block_recall_at_4x_budget": row["block_recall_at_4x_budget"],
                "block_mrr": row["block_mrr"],
            }
        )

    summary = []
    for (mode, sketch_type, sketch_dim, topk_blocks), vals in sorted(grouped.items()):
        n = float(len(vals))
        summary.append(
            {
                "mode": mode,
                "sketch_type": sketch_type,
                "sketch_dim": sketch_dim,
                "topk_blocks": topk_blocks,
                "avg_block_topk_recall": float(
                    sum(v["block_topk_recall"] for v in vals) / n
                ),
                "avg_block_recall_at_2x_budget": float(
                    sum(v["block_recall_at_2x_budget"] for v in vals) / n
                ),
                "avg_block_recall_at_4x_budget": float(
                    sum(v["block_recall_at_4x_budget"] for v in vals) / n
                ),
                "avg_block_mrr": float(sum(v["block_mrr"] for v in vals) / n),
            }
        )

    print(json.dumps({"output": str(output_path), "num_runs": len(rows)}))
    for row in summary:
        print(json.dumps(row, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
