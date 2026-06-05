#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class QKExtractionResult(NamedTuple):
    query: np.ndarray
    keys: np.ndarray
    query_position: int
    metadata: dict[str, Any]


def _load_sketch_math_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "vllm" / "v1" / "core" / "kivo_vd_sketch_math.py"
    spec = importlib.util.spec_from_file_location("kivo_vd_sketch_math", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_transformer_layers(model: Any) -> Any:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        if hasattr(decoder, "layers"):
            return decoder.layers
    raise ValueError(
        "Unable to find transformer layers. Expected model.transformer.h, "
        "model.model.layers, or model.model.decoder.layers."
    )


def _get_layer_attention(layer_module: Any) -> Any:
    for name in ("attn", "self_attn", "attention"):
        if hasattr(layer_module, name):
            return getattr(layer_module, name)
    raise ValueError(
        "Unable to find attention module on layer. Expected one of: "
        "attn, self_attn, attention."
    )


def _inspected_attention_attrs(attn: Any) -> list[str]:
    return sorted(
        name
        for name in dir(attn)
        if not name.startswith("_")
        and any(token in name for token in ("attn", "proj", "head"))
    )


def _detect_extraction_mode(attn: Any, extraction_mode: str) -> str:
    if extraction_mode != "auto":
        return extraction_mode
    if hasattr(attn, "c_attn"):
        return "gpt2_fused_c_attn"
    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
        return "separate_qk_proj"
    raise ValueError(
        "Unsupported attention module for Q/K extraction. Inspected attrs: "
        f"{_inspected_attention_attrs(attn)}"
    )


def _map_query_head_to_kv_head(
    query_head: int,
    num_query_heads: int,
    num_key_value_heads: int,
) -> int:
    if num_query_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("Head counts must be positive.")
    if query_head < 0 or query_head >= num_query_heads:
        raise ValueError(
            f"Head {query_head} out of range [0, {num_query_heads - 1}]"
        )
    if num_query_heads == num_key_value_heads:
        return query_head
    if num_query_heads % num_key_value_heads != 0:
        raise ValueError(
            "Cannot map query head to KV head because num_query_heads is not "
            "divisible by num_key_value_heads."
        )
    group_size = num_query_heads // num_key_value_heads
    return query_head // group_size


def _validate_query_position(
    query_position: int,
    seq_len: int,
) -> None:
    if seq_len < 2:
        raise ValueError("Need at least 2 tokens to evaluate query vs prior keys.")
    if query_position < 0 or query_position >= seq_len:
        raise ValueError(
            f"query_position={query_position} out of range [0, {seq_len - 1}]"
        )
    if query_position == 0:
        raise ValueError("query_position must be >= 1 to have at least one key.")


def _get_num_query_heads(model: Any, attn: Any) -> int:
    for obj, name in (
        (attn, "num_heads"),
        (attn, "num_attention_heads"),
        (getattr(model, "config", None), "num_attention_heads"),
        (getattr(model, "config", None), "n_head"),
    ):
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to determine number of query heads.")


def _get_num_key_value_heads(model: Any, attn: Any, num_query_heads: int) -> int:
    for obj, name in (
        (attn, "num_key_value_heads"),
        (getattr(model, "config", None), "num_key_value_heads"),
        (getattr(model, "config", None), "num_key_value_heads_per_layer"),
    ):
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return num_query_heads


def _extract_gpt2_head_qk(
    model: Any,
    input_ids: Any,
    layer: int,
    head: int,
    query_position: int,
    extraction_mode: str,
) -> QKExtractionResult:
    with __import__("torch").no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    layers = _get_transformer_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"Layer {layer} out of range [0, {len(layers) - 1}]")

    h_in = hidden_states[layer]  # input to selected transformer block
    attn = _get_layer_attention(layers[layer])
    resolved_mode = _detect_extraction_mode(attn, extraction_mode)
    if resolved_mode != "gpt2_fused_c_attn":
        raise ValueError(
            "Requested GPT-2 fused extraction but resolved mode was "
            f"{resolved_mode!r}."
        )
    qkv = attn.c_attn(h_in)  # [1, seq, 3*hidden]

    q, k, _v = qkv.chunk(3, dim=-1)
    q = q[0]
    k = k[0]

    num_heads = _get_num_query_heads(model, attn)
    head_dim = q.shape[-1] // num_heads
    if head < 0 or head >= num_heads:
        raise ValueError(f"Head {head} out of range [0, {num_heads - 1}]")

    q = q.view(q.shape[0], num_heads, head_dim)
    k = k.view(k.shape[0], num_heads, head_dim)

    q_head = q[:, head, :]
    k_head = k[:, head, :]

    seq_len = q_head.shape[0]
    _validate_query_position(query_position, seq_len)

    # Causal: use keys up to (but excluding) query position.
    query = q_head[query_position].detach().float().cpu().numpy().astype(np.float64)
    keys = k_head[:query_position].detach().float().cpu().numpy().astype(np.float64)
    return QKExtractionResult(
        query=query,
        keys=keys,
        query_position=query_position,
        metadata={
            "extraction_mode": resolved_mode,
            "qk_space": "gpt2_projection",
            "num_query_heads": num_heads,
            "num_key_value_heads": num_heads,
            "selected_query_head": head,
            "selected_kv_head": head,
        },
    )


def _extract_separate_qk_proj(
    model: Any,
    input_ids: Any,
    layer: int,
    head: int,
    query_position: int,
    extraction_mode: str,
) -> QKExtractionResult:
    with __import__("torch").no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    layers = _get_transformer_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"Layer {layer} out of range [0, {len(layers) - 1}]")

    h_in = hidden_states[layer]
    attn = _get_layer_attention(layers[layer])
    resolved_mode = _detect_extraction_mode(attn, extraction_mode)
    if resolved_mode != "separate_qk_proj":
        raise ValueError(
            "Requested separate q/k extraction but resolved mode was "
            f"{resolved_mode!r}."
        )

    q = attn.q_proj(h_in)[0]
    k = attn.k_proj(h_in)[0]
    num_query_heads = _get_num_query_heads(model, attn)
    num_kv_heads = _get_num_key_value_heads(model, attn, num_query_heads)
    selected_kv_head = _map_query_head_to_kv_head(
        head, num_query_heads, num_kv_heads
    )

    q_head_dim = q.shape[-1] // num_query_heads
    k_head_dim = k.shape[-1] // num_kv_heads
    if q_head_dim != k_head_dim:
        raise ValueError(
            f"Q/K head dim mismatch: q={q_head_dim}, k={k_head_dim}."
        )

    q = q.view(q.shape[0], num_query_heads, q_head_dim)
    k = k.view(k.shape[0], num_kv_heads, k_head_dim)
    q_head = q[:, head, :]
    k_head = k[:, selected_kv_head, :]
    seq_len = q_head.shape[0]
    _validate_query_position(query_position, seq_len)

    query = q_head[query_position].detach().float().cpu().numpy().astype(np.float64)
    keys = k_head[:query_position].detach().float().cpu().numpy().astype(np.float64)
    return QKExtractionResult(
        query=query,
        keys=keys,
        query_position=query_position,
        metadata={
            "extraction_mode": resolved_mode,
            "qk_space": "pre_rope_projection",
            "num_query_heads": num_query_heads,
            "num_key_value_heads": num_kv_heads,
            "selected_query_head": head,
            "selected_kv_head": selected_kv_head,
        },
    )


def _extract_head_qk(
    model: Any,
    input_ids: Any,
    layer: int,
    head: int,
    query_position: int,
    extraction_mode: str,
) -> QKExtractionResult:
    layers = _get_transformer_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"Layer {layer} out of range [0, {len(layers) - 1}]")
    attn = _get_layer_attention(layers[layer])
    resolved_mode = _detect_extraction_mode(attn, extraction_mode)
    if resolved_mode == "gpt2_fused_c_attn":
        return _extract_gpt2_head_qk(
            model, input_ids, layer, head, query_position, resolved_mode
        )
    if resolved_mode == "separate_qk_proj":
        return _extract_separate_qk_proj(
            model, input_ids, layer, head, query_position, resolved_mode
        )
    raise ValueError(f"Unsupported extraction mode: {resolved_mode}")


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
    parser.add_argument(
        "--extraction-mode",
        choices=["auto", "gpt2_fused_c_attn", "separate_qk_proj"],
        default="auto",
    )
    parser.add_argument(
        "--include-ranked-blocks",
        action="store_true",
        help="Include full approximate block ranking in JSON output.",
    )
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


def _sketch_compression_metadata(
    *,
    head_dim: int,
    sketch_type: str,
    sketch_dim: int,
) -> dict[str, Any]:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive.")
    if sketch_dim <= 0:
        raise ValueError("sketch_dim must be positive.")
    effective_sketch_dim = min(sketch_dim, head_dim)
    return {
        "head_dim": int(head_dim),
        "effective_input_dim": int(head_dim),
        "effective_sketch_dim": int(effective_sketch_dim),
        "sketch_compression_ratio": float(effective_sketch_dim / head_dim),
        "is_full_dimensional_sketch": bool(effective_sketch_dim >= head_dim),
    }


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
    include_ranked_blocks: bool = False,
    extraction_mode: str = "auto",
) -> dict[str, Any]:
    extraction = _extract_head_qk(
        model=model,
        input_ids=input_ids,
        layer=layer,
        head=head,
        query_position=query_position,
        extraction_mode=extraction_mode,
    )
    query = extraction.query
    keys = extraction.keys
    resolved_position = extraction.query_position
    head_dim = int(query.shape[0])

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

    out = {
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
        **_sketch_compression_metadata(
            head_dim=head_dim,
            sketch_type=sketch_type,
            sketch_dim=sketch_dim,
        ),
        **extraction.metadata,
    }
    if include_ranked_blocks:
        out["approx_ranked_block_ids"] = approx_block_ranking.tolist()
    return out


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
        effective_max_tokens = (
            model_context if model_context is not None else args.max_tokens
        )

    input_ids, truncated = _truncate_input_ids(
        input_ids=input_ids,
        max_tokens=effective_max_tokens,
        truncate_side=args.truncate_side,
    )
    prompt_num_tokens = int(input_ids.shape[1])

    if args.sweep_query_positions:
        query_positions = _sweep_query_positions(prompt_num_tokens)
    else:
        query_positions = [
            _resolve_query_position(args.query_position, prompt_num_tokens)
        ]

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
            include_ranked_blocks=args.include_ranked_blocks,
            extraction_mode=args.extraction_mode,
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
