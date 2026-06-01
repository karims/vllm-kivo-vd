#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_sketch_math_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "vllm" / "v1" / "core" / "kivo_vd_sketch_math.py"
    spec = importlib.util.spec_from_file_location("kivo_vd_sketch_math", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_gpt2_head_qk(
    model: Any,
    input_ids: Any,
    layer: int,
    head: int,
) -> tuple[np.ndarray, np.ndarray]:
    with __import__("torch").no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    if layer < 0 or layer >= len(model.transformer.h):
        raise ValueError(f"Layer {layer} out of range [0, {len(model.transformer.h) - 1}]")

    h_in = hidden_states[layer]  # input to selected transformer block
    attn = model.transformer.h[layer].attn
    qkv = attn.c_attn(h_in)  # [1, seq, 3*hidden]

    q, k, _v = qkv.chunk(3, dim=-1)
    q = q[0]
    k = k[0]

    num_heads = getattr(attn, "num_heads", model.config.n_head)
    head_dim = q.shape[-1] // num_heads
    if head < 0 or head >= num_heads:
        raise ValueError(f"Head {head} out of range [0, {num_heads - 1}]")

    q = q.view(q.shape[0], num_heads, head_dim)
    k = k.view(k.shape[0], num_heads, head_dim)

    q_head = q[:, head, :]
    k_head = k[:, head, :]

    seq_len = q_head.shape[0]
    if seq_len < 2:
        raise ValueError("Need at least 2 tokens to evaluate final-token query vs previous keys.")

    # Final-token query against all previous keys.
    query = q_head[seq_len - 1].detach().cpu().numpy().astype(np.float64)
    keys = k_head[: seq_len - 1].detach().cpu().numpy().astype(np.float64)
    return query, keys


def _default_prompt() -> str:
    return (
        "In a quiet library, a researcher compares notes from several experiments "
        "on memory-efficient attention. She observes that exact top-k agreement "
        "can be noisy, but candidate retrieval quality may still be strong when "
        "evaluated with larger budgets and reranking."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional HF Q/K offline sketch eval.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default=_default_prompt())
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument(
        "--sketch-type",
        choices=["random_projection", "count_sketch"],
        default="random_projection",
    )
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--topk-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    math = _load_sketch_math_module()

    # Optional heavy deps imported only for this script path.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "This optional script requires torch and transformers. "
            "Install them in your environment first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(args.device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)

    query, keys = _extract_gpt2_head_qk(
        model=model,
        input_ids=input_ids,
        layer=args.layer,
        head=args.head,
    )

    exact_scores = math.compute_exact_scores(query, keys)
    approx_scores = math.compute_sketched_scores(
        query=query,
        keys=keys,
        sketch_type=args.sketch_type,
        sketch_dim=args.sketch_dim,
        seed=args.seed,
    )

    token_topk = min(keys.shape[0], max(1, args.topk_blocks * args.block_size))
    exact_top_tokens = math.topk_indices(exact_scores, token_topk)
    approx_top_tokens = math.topk_indices(approx_scores, token_topk)
    token_recall = math.topk_recall(exact_top_tokens, approx_top_tokens)
    token_score_corr = math.pearson_correlation(exact_scores, approx_scores)

    exact_block_scores = math.block_scores_from_token_scores(
        exact_scores, args.block_size, mode="max"
    )
    approx_block_scores = math.block_scores_from_token_scores(
        approx_scores, args.block_size, mode="max"
    )
    block_score_corr = math.pearson_correlation(exact_block_scores, approx_block_scores)
    exact_top_blocks = math.topk_indices(exact_block_scores, args.topk_blocks)
    approx_top_blocks = math.topk_indices(approx_block_scores, args.topk_blocks)
    approx_block_ranking = math.topk_indices(
        approx_block_scores, approx_block_scores.shape[0]
    )
    block_topk_recall = math.topk_recall(exact_top_blocks, approx_top_blocks)
    block_recall_2x = math.recall_at_budget(
        exact_top_blocks, approx_block_ranking, args.topk_blocks * 2
    )
    block_recall_4x = math.recall_at_budget(
        exact_top_blocks, approx_block_ranking, args.topk_blocks * 4
    )
    block_mrr = math.mean_reciprocal_rank(exact_top_blocks, approx_block_ranking)

    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "prompt_num_tokens": int(input_ids.shape[1]),
                "layer": args.layer,
                "head": args.head,
                "sketch_type": args.sketch_type,
                "sketch_dim": args.sketch_dim,
                "block_size": args.block_size,
                "topk_blocks": args.topk_blocks,
                "token_topk_recall": float(token_recall),
                "block_topk_recall": float(block_topk_recall),
                "block_recall_at_2x_budget": float(block_recall_2x),
                "block_recall_at_4x_budget": float(block_recall_4x),
                "block_mrr": float(block_mrr),
                "token_score_correlation": float(token_score_corr),
                "block_score_correlation": float(block_score_corr),
                "exact_top_block_ids": exact_top_blocks.tolist(),
                "approx_top_block_ids": approx_top_blocks.tolist(),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
