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
    query_position: int,
) -> tuple[np.ndarray, np.ndarray, int]:
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
        raise ValueError("Need at least 2 tokens to evaluate query vs prior keys.")
    if query_position < 0 or query_position >= seq_len:
        raise ValueError(
            f"query_position={query_position} out of range [0, {seq_len - 1}]"
        )
    if query_position == 0:
        raise ValueError("query_position must be >= 1 to have at least one key.")

    # Causal: use keys up to (but excluding) query position.
    query = q_head[query_position].detach().cpu().numpy().astype(np.float64)
    keys = k_head[:query_position].detach().cpu().numpy().astype(np.float64)
    return query, keys, query_position


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
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--truncate-side",
        choices=["left", "right"],
        default="right",
    )
    parser.add_argument("--query-position", default="last")
    parser.add_argument("--sweep-query-positions", action="store_true")
    return parser.parse_args()


def _resolve_model_max_context_tokens(model: Any, tokenizer: Any) -> int | None:
    n_positions = getattr(getattr(model, "config", None), "n_positions", None)
    if isinstance(n_positions, int) and n_positions > 0:
        return n_positions

    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 1_000_000:
        return tokenizer_max
    return None


def _truncate_input_ids(
    input_ids: Any, max_tokens: int | None, truncate_side: str
) -> tuple[Any, bool]:
    if max_tokens is None:
        return input_ids, False
    if max_tokens < 2:
        raise ValueError("max_tokens must be >= 2 when set.")
    seq_len = int(input_ids.shape[1])
    if seq_len <= max_tokens:
        return input_ids, False
    if truncate_side == "left":
        return input_ids[:, seq_len - max_tokens :], True
    return input_ids[:, :max_tokens], True


def _resolve_query_position(query_position: str, seq_len: int) -> int:
    if seq_len < 2:
        raise ValueError("Need at least 2 tokens to resolve query position.")
    if query_position == "last":
        return seq_len - 1
    try:
        idx = int(query_position)
    except ValueError as exc:
        raise ValueError("query-position must be 'last' or an integer string.") from exc
    if idx < 0:
        idx = seq_len + idx
    if idx < 1 or idx >= seq_len:
        raise ValueError(
            f"Resolved query position {idx} out of valid range [1, {seq_len - 1}]"
        )
    return idx


def _sweep_query_positions(seq_len: int) -> list[int]:
    if seq_len < 2:
        raise ValueError("Need at least 2 tokens for query position sweep.")
    candidates = [
        max(1, int(seq_len * 0.25)),
        max(1, int(seq_len * 0.50)),
        max(1, int(seq_len * 0.75)),
        seq_len - 1,
    ]
    out: list[int] = []
    for c in candidates:
        if c >= seq_len:
            c = seq_len - 1
        if c not in out:
            out.append(c)
    return out


def _evaluate_at_query_position(
    *,
    math: Any,
    model: Any,
    input_ids: Any,
    layer: int,
    head: int,
    query_position: int,
    sketch_type: str,
    sketch_dim: int,
    seed: int,
    block_size: int,
    topk_blocks: int,
) -> dict[str, Any]:
    query, keys, resolved_position = _extract_gpt2_head_qk(
        model=model,
        input_ids=input_ids,
        layer=layer,
        head=head,
        query_position=query_position,
    )

    exact_scores = math.compute_exact_scores(query, keys)
    approx_scores = math.compute_sketched_scores(
        query=query,
        keys=keys,
        sketch_type=sketch_type,
        sketch_dim=sketch_dim,
        seed=seed,
    )

    token_topk = min(keys.shape[0], max(1, topk_blocks * block_size))
    exact_top_tokens = math.topk_indices(exact_scores, token_topk)
    approx_top_tokens = math.topk_indices(approx_scores, token_topk)
    token_recall = math.topk_recall(exact_top_tokens, approx_top_tokens)
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
    approx_block_ranking = math.topk_indices(
        approx_block_scores, approx_block_scores.shape[0]
    )
    block_topk_recall = math.topk_recall(exact_top_blocks, approx_top_blocks)
    block_recall_2x = math.recall_at_budget(
        exact_top_blocks, approx_block_ranking, topk_blocks * 2
    )
    block_recall_4x = math.recall_at_budget(
        exact_top_blocks, approx_block_ranking, topk_blocks * 4
    )
    block_mrr = math.mean_reciprocal_rank(exact_top_blocks, approx_block_ranking)

    return {
        "query_position": int(resolved_position),
        "num_keys_used": int(keys.shape[0]),
        "token_topk_recall": float(token_recall),
        "block_topk_recall": float(block_topk_recall),
        "block_recall_at_2x_budget": float(block_recall_2x),
        "block_recall_at_4x_budget": float(block_recall_4x),
        "block_mrr": float(block_mrr),
        "token_score_correlation": float(token_score_corr),
        "block_score_correlation": float(block_score_corr),
        "exact_top_block_ids": exact_top_blocks.tolist(),
        "approx_top_block_ids": approx_top_blocks.tolist(),
    }


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
    original_prompt_num_tokens = int(input_ids.shape[1])

    model_context = _resolve_model_max_context_tokens(model, tokenizer)
    if model_context is not None and args.max_tokens is not None:
        effective_max_tokens = min(model_context, args.max_tokens)
    else:
        effective_max_tokens = model_context if model_context is not None else args.max_tokens

    input_ids, truncated = _truncate_input_ids(
        input_ids=input_ids,
        max_tokens=effective_max_tokens,
        truncate_side=args.truncate_side,
    )
    prompt_num_tokens = int(input_ids.shape[1])

    if args.sweep_query_positions:
        query_positions = _sweep_query_positions(prompt_num_tokens)
    else:
        query_positions = [_resolve_query_position(args.query_position, prompt_num_tokens)]

    for query_position in query_positions:
        metrics = _evaluate_at_query_position(
            math=math,
            model=model,
            input_ids=input_ids,
            layer=args.layer,
            head=args.head,
            query_position=query_position,
            sketch_type=args.sketch_type,
            sketch_dim=args.sketch_dim,
            seed=args.seed,
            block_size=args.block_size,
            topk_blocks=args.topk_blocks,
        )
        payload = {
            "model_name": args.model_name,
            "original_prompt_num_tokens": original_prompt_num_tokens,
            "prompt_num_tokens": prompt_num_tokens,
            "truncated": bool(truncated),
            "max_context_tokens": (
                int(effective_max_tokens)
                if effective_max_tokens is not None
                else None
            ),
            "layer": args.layer,
            "head": args.head,
            "sketch_type": args.sketch_type,
            "sketch_dim": args.sketch_dim,
            "block_size": args.block_size,
            "topk_blocks": args.topk_blocks,
        }
        payload.update(metrics)
        print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
