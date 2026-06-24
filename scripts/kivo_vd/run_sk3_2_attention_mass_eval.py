#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Offline evaluator for true attention mass vs sketch-based block scoring."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kivo_vd.run_hf_qk_sketch_eval import (  # noqa: E402
    _extract_head_qk,
    _get_num_query_heads,
    _get_transformer_layers,
    _resolve_model_max_context_tokens,
    _resolve_query_position,
    _truncate_input_ids,
)
from vllm.v1.worker.kivo_kv_sketch_runtime import (  # noqa: E402
    CountSketchKVSketchBackend,
    KivoKVSketchBackend,
    RandomProjectionKVSketchBackend,
)

BACKEND_MEAN_KEY = "mean_key"
BACKEND_RANDOM_PROJECTION = "random_projection"
BACKEND_COUNTSKETCH = "countsketch"
BACKEND_ALL = "all"
BACKEND_CHOICES = [
    BACKEND_COUNTSKETCH,
    BACKEND_RANDOM_PROJECTION,
    BACKEND_MEAN_KEY,
    BACKEND_ALL,
]


def build_prompt(prompt_kind: str) -> str:
    factual_recall = (
        "Project Orion launched in 2018. The team moved the project to Toronto "
        "in 2020. The lead researcher is Maya Chen. The archive was updated in "
        "2021 with new references.\nQuestion: Who leads Project Orion and when "
        "did it move to Toronto?"
    )
    code_context = (
        "Function compute_total iterates over invoice items, adds tax_rate, and "
        "returns the result in cents as an integer for downstream API callers. "
        "The integer return avoids floating-point drift in billing paths.\n"
        "Question: What does compute_total return and why are cents used?"
    )
    synthetic_long = " ".join(
        [
            "Early note: the recovery code is ALDER-42 and the fallback city is Halifax.",
            "Background text about long-context evaluation and retrieval robustness."
            * 40,
            "Final question: what is the recovery code and fallback city?",
        ]
    )
    prompt_map = {
        "factual_recall": factual_recall,
        "code_context": code_context,
        "synthetic_long": synthetic_long,
    }
    try:
        return prompt_map[prompt_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt kind: {prompt_kind}") from exc


def compute_true_attention_weights(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    if query.ndim != 1:
        raise ValueError("query must be 1D.")
    if keys.ndim != 2:
        raise ValueError("keys must be 2D [num_tokens, head_dim].")
    if keys.shape[1] != query.shape[0]:
        raise ValueError("query/key head_dim mismatch.")
    logits = torch.matmul(keys, query) / math.sqrt(float(query.shape[0]))
    return torch.softmax(logits, dim=0)


def aggregate_block_values(values: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if values.ndim != 1:
        raise ValueError("values must be 1D.")
    if values.numel() == 0:
        return torch.empty(0, dtype=values.dtype)
    blocks = []
    for start in range(0, int(values.numel()), block_size):
        blocks.append(values[start : start + block_size].sum())
    return torch.stack(blocks)


def aggregate_block_vectors(keys: torch.Tensor, block_size: int) -> list[torch.Tensor]:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if keys.ndim != 2:
        raise ValueError("keys must be 2D [num_tokens, head_dim].")
    blocks = []
    for start in range(0, int(keys.shape[0]), block_size):
        block = keys[start : start + block_size]
        blocks.append(block)
    return blocks


def mean_key_block_scores(query: torch.Tensor, keys: torch.Tensor, block_size: int) -> torch.Tensor:
    scores = []
    for block in aggregate_block_vectors(keys, block_size):
        block_mean = block.mean(dim=0)
        scores.append(torch.dot(query, block_mean))
    if not scores:
        return torch.empty(0, dtype=query.dtype)
    return torch.stack(scores)


def _build_sketch_backend(
    backend: str,
    *,
    sketch_dim: int,
    seed: int,
) -> KivoKVSketchBackend:
    if backend == BACKEND_RANDOM_PROJECTION:
        return RandomProjectionKVSketchBackend(sketch_dim=sketch_dim, seed=seed)
    if backend == BACKEND_COUNTSKETCH:
        return CountSketchKVSketchBackend(sketch_dim=sketch_dim, seed=seed)
    raise ValueError(f"Unsupported sketch backend: {backend}")


def sketch_block_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    block_size: int,
    backend: str,
    sketch_dim: int,
    seed: int,
) -> torch.Tensor:
    sketch_backend = _build_sketch_backend(backend, sketch_dim=sketch_dim, seed=seed)
    query_result = sketch_backend.build_block_sketch(
        block_id=-1,
        block_tensor=query,
        kv_kind="query",
    )
    if not query_result.success or query_result.record is None:
        raise RuntimeError(
            f"Failed to sketch query for backend {backend}: "
            f"{query_result.blocker_reason}"
        )
    query_sketch = query_result.record.sketch.float()
    block_scores = []
    for block_idx, block in enumerate(aggregate_block_vectors(keys, block_size)):
        token_sketches = []
        for token_idx, token in enumerate(block):
            result = sketch_backend.build_block_sketch(
                block_id=block_idx * block_size + token_idx,
                block_tensor=token,
                kv_kind="key",
            )
            if not result.success or result.record is None:
                raise RuntimeError(
                    f"Failed to sketch key token for backend {backend}: "
                    f"{result.blocker_reason}"
                )
            token_sketches.append(result.record.sketch.float())
        block_sketch_mean = torch.stack(token_sketches).mean(dim=0)
        block_scores.append(torch.dot(query_sketch, block_sketch_mean))
    if not block_scores:
        return torch.empty(0, dtype=query.dtype)
    return torch.stack(block_scores)


def topk_indices_desc(scores: torch.Tensor, topk: int) -> list[int]:
    if scores.numel() == 0:
        return []
    k = max(1, min(int(topk), int(scores.numel())))
    return (
        torch.topk(scores, k=k, largest=True, sorted=True).indices.tolist()
    )


def attention_mass_recovered(true_mass: torch.Tensor, selected_block_ids: list[int]) -> float:
    if true_mass.numel() == 0 or not selected_block_ids:
        return 0.0
    valid_ids = [idx for idx in selected_block_ids if 0 <= idx < int(true_mass.numel())]
    if not valid_ids:
        return 0.0
    return float(true_mass[valid_ids].sum().item())


def topk_overlap(predicted: list[int], truth: list[int]) -> int:
    return len(set(predicted).intersection(truth))


def _rank_positions_desc(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0 + 1.0
        for pos in range(index, next_index):
            ranks[indexed[pos][0]] = average_rank
        index = next_index
    return ranks


def spearman_rank_correlation(true_values: torch.Tensor, approx_values: torch.Tensor) -> float | None:
    if true_values.numel() != approx_values.numel():
        raise ValueError("true_values and approx_values must have same length.")
    if true_values.numel() < 2:
        return None
    true_ranks = torch.tensor(_rank_positions_desc(true_values.tolist()), dtype=torch.float64)
    approx_ranks = torch.tensor(_rank_positions_desc(approx_values.tolist()), dtype=torch.float64)
    true_centered = true_ranks - true_ranks.mean()
    approx_centered = approx_ranks - approx_ranks.mean()
    denom = torch.linalg.vector_norm(true_centered) * torch.linalg.vector_norm(approx_centered)
    if float(denom.item()) == 0.0:
        return None
    return float(torch.dot(true_centered, approx_centered).item() / denom.item())


def evaluate_backend_scores(
    *,
    backend: str,
    query: torch.Tensor,
    keys: torch.Tensor,
    block_size: int,
    sketch_dim: int,
    seed: int,
    topks: list[int],
) -> dict[str, Any]:
    true_attention = compute_true_attention_weights(query, keys)
    true_mass = aggregate_block_values(true_attention, block_size)
    warnings: list[str] = []

    if backend == BACKEND_MEAN_KEY:
        scores = mean_key_block_scores(query, keys, block_size)
    elif backend in {BACKEND_RANDOM_PROJECTION, BACKEND_COUNTSKETCH}:
        scores = sketch_block_scores(
            query,
            keys,
            block_size=block_size,
            backend=backend,
            sketch_dim=sketch_dim,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    top_blocks_by_true_mass = topk_indices_desc(true_mass, max(topks))
    top_blocks_by_score = topk_indices_desc(scores, max(topks))
    topk_metrics: dict[str, Any] = {}
    for topk in topks:
        selected = topk_indices_desc(scores, topk)
        true_topk = topk_indices_desc(true_mass, topk)
        topk_metrics[str(topk)] = {
            "attention_mass_recovered_by_score_topk": attention_mass_recovered(
                true_mass, selected
            ),
            "topk_overlap_with_true_topk": topk_overlap(selected, true_topk),
            "true_topk_mass": attention_mass_recovered(true_mass, true_topk),
        }

    spearman = spearman_rank_correlation(true_mass, scores)
    if spearman is None:
        warnings.append("rank_correlation_spearman unavailable_or_degenerate")

    return {
        "score_by_block": scores.tolist(),
        "true_mass_by_block": true_mass.tolist(),
        "top_blocks_by_true_mass": top_blocks_by_true_mass,
        "top_blocks_by_score": top_blocks_by_score,
        "topk_metrics": topk_metrics,
        "rank_correlation_spearman": spearman,
        "warnings": warnings,
    }


def build_output_payload(
    *,
    model: str,
    prompt_kind: str,
    token_count: int,
    block_size: int,
    num_blocks: int,
    layer: int,
    head: int,
    query_position: int,
    backend_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt_kind": prompt_kind,
        "token_count": int(token_count),
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "layer": int(layer),
        "head": int(head),
        "query_position": int(query_position),
        "backend_results": backend_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline attention-mass evaluator for sketch scoring."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt-kind",
        choices=["factual_recall", "code_context", "synthetic_long"],
        default="factual_recall",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--layer", default="last")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--query-position", default="last")
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--sketch-seed", type=int, default=123)
    parser.add_argument(
        "--topk",
        type=int,
        action="append",
        default=None,
        help="Repeatable top-k block budget; defaults to 4,8,16.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_ALL,
    )
    parser.add_argument("--output", default="/tmp/sk3_2_attention_mass_eval.json")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--extraction-mode",
        choices=["auto", "gpt2_fused_c_attn", "separate_qk_proj"],
        default="auto",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _resolve_layer_index(model: Any, layer_arg: str) -> int:
    layers = _get_transformer_layers(model)
    if layer_arg == "last":
        return len(layers) - 1
    layer_index = int(layer_arg)
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f"Layer {layer_index} out of range [0, {len(layers) - 1}]")
    return layer_index


def _resolve_topks(topks: list[int] | None) -> list[int]:
    resolved = topks if topks else [4, 8, 16]
    cleaned = sorted({max(1, int(value)) for value in resolved})
    return cleaned


def _resolve_backends(backend: str) -> list[str]:
    if backend == BACKEND_ALL:
        return [BACKEND_MEAN_KEY, BACKEND_RANDOM_PROJECTION, BACKEND_COUNTSKETCH]
    return [backend]


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "This offline evaluator requires transformers to run the HF model path."
        ) from exc

    load_kwargs: dict[str, Any] = {}
    if args.local_files_only:
        load_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model = model.to(args.device)
    model.eval()
    return model, tokenizer


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    topks = _resolve_topks(args.topk)
    model, tokenizer = _load_model_and_tokenizer(args)
    prompt = build_prompt(args.prompt_kind)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    model_context = _resolve_model_max_context_tokens(model, tokenizer)
    effective_max_tokens = args.max_model_len
    if model_context is not None:
        effective_max_tokens = min(effective_max_tokens, int(model_context))
    input_ids, _ = _truncate_input_ids(
        input_ids=input_ids,
        max_tokens=effective_max_tokens,
        truncate_side="right",
    )
    token_count = int(input_ids.shape[1])
    layer_index = _resolve_layer_index(model, args.layer)
    layers = _get_transformer_layers(model)
    attn = getattr(layers[layer_index], "self_attn", None) or getattr(
        layers[layer_index], "attn", None
    )
    num_query_heads = _get_num_query_heads(model, attn)
    if args.head < 0 or args.head >= num_query_heads:
        raise ValueError(f"Head {args.head} out of range [0, {num_query_heads - 1}]")
    query_position = _resolve_query_position(args.query_position, token_count)
    extraction = _extract_head_qk(
        model=model,
        input_ids=input_ids,
        layer=layer_index,
        head=args.head,
        query_position=query_position,
        extraction_mode=args.extraction_mode,
    )
    query = torch.from_numpy(extraction.query).to(torch.float32)
    keys = torch.from_numpy(extraction.keys).to(torch.float32)
    num_blocks = (int(keys.shape[0]) + args.block_size - 1) // args.block_size

    backend_results: dict[str, Any] = {}
    for backend_name in _resolve_backends(args.backend):
        backend_results[backend_name] = evaluate_backend_scores(
            backend=backend_name,
            query=query,
            keys=keys,
            block_size=args.block_size,
            sketch_dim=args.sketch_dim,
            seed=args.sketch_seed,
            topks=topks,
        )

    payload = build_output_payload(
        model=args.model,
        prompt_kind=args.prompt_kind,
        token_count=token_count,
        block_size=args.block_size,
        num_blocks=num_blocks,
        layer=layer_index,
        head=args.head,
        query_position=extraction.query_position,
        backend_results=backend_results,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
