#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Offline evaluator for true attention mass vs approximate block scores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kivo_vd.run_hf_qk_sketch_eval import (  # noqa: E402
    _extract_head_qk,
    _get_layer_attention,
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
BACKEND_MAX_TOKEN_SCORE_EXACT = "max_token_score_exact"
BACKEND_ALL = "all"
BACKEND_CHOICES = [
    BACKEND_COUNTSKETCH,
    BACKEND_RANDOM_PROJECTION,
    BACKEND_MEAN_KEY,
    BACKEND_MAX_TOKEN_SCORE_EXACT,
    BACKEND_ALL,
]
DEFAULT_TOPKS = [4, 8, 16, 32]


class EvalPoint(NamedTuple):
    layer: int
    head: int
    query_position: int
    query: torch.Tensor
    keys: torch.Tensor


def estimate_prompt_token_count(prompt: str) -> int:
    return len(prompt.split())


def _repeat_distractor(base: str, repeats: int) -> str:
    return " ".join([base] * repeats)


def build_prompt(prompt_kind: str) -> str:
    factual_recall = (
        "Project Orion launched in 2018. The team moved the project to Toronto "
        "in 2020. The lead researcher is Maya Chen. The archive was updated in "
        "2021 with new references.\nQuestion: Who leads Project Orion and when "
        "did it move to Toronto?"
    )
    factual_recall_long = "\n".join(
        [
            "Fact: Project Orion was led by Maya Chen and moved to Toronto in 2020.",
            _repeat_distractor(
                (
                    "Distractor memo about compliance reviews, archival scans, "
                    "meeting notes, procurement tickets, and instrumentation "
                    "updates for the long-context benchmark."
                ),
                140,
            ),
            "Question: Who led Project Orion and when did it move to Toronto?",
        ]
    )
    code_context = (
        "Function compute_total iterates over invoice items, adds tax_rate, and "
        "returns the result in cents as an integer for downstream API callers. "
        "The integer return avoids floating-point drift in billing paths.\n"
        "Question: What does compute_total return and why are cents used?"
    )
    code_context_long = "\n".join(
        [
            (
                "Code fact: compute_total returns invoice total including tax "
                "in cents as an integer for API callers."
            ),
            _repeat_distractor(
                (
                    "Additional repository context covers serializers, retry "
                    "logic, HTTP handlers, schema migrations, deployment notes, "
                    "unit tests, and comments unrelated to the key billing fact."
                ),
                140,
            ),
            (
                "Question: What does compute_total return and why is the value "
                "represented in cents?"
            ),
        ]
    )
    synthetic_long = "\n".join(
        [
            "Early note: the recovery code is ALDER-42 and the fallback city is Halifax.",
            _repeat_distractor(
                (
                    "Background text about long-context evaluation, distractor "
                    "entities, routing trivia, timestamp logs, and references "
                    "that should not change the answer-bearing evidence."
                ),
                145,
            ),
            "Final question: what is the recovery code and fallback city?",
        ]
    )
    prompt_map = {
        "factual_recall": factual_recall,
        "factual_recall_long": factual_recall_long,
        "code_context": code_context,
        "code_context_long": code_context_long,
        "synthetic_long": synthetic_long,
    }
    try:
        return prompt_map[prompt_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt kind: {prompt_kind}") from exc


def compute_true_attention_weights(
    query: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
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
        blocks.append(keys[start : start + block_size])
    return blocks


def mean_key_block_scores(
    query: torch.Tensor, keys: torch.Tensor, block_size: int
) -> torch.Tensor:
    scores = []
    for block in aggregate_block_vectors(keys, block_size):
        scores.append(torch.dot(query, block.mean(dim=0)))
    if not scores:
        return torch.empty(0, dtype=query.dtype)
    return torch.stack(scores)


def max_token_score_exact(
    query: torch.Tensor, keys: torch.Tensor, block_size: int
) -> torch.Tensor:
    scores = []
    for block in aggregate_block_vectors(keys, block_size):
        scores.append(torch.max(torch.matmul(block, query)))
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
    return torch.topk(scores, k=k, largest=True, sorted=True).indices.tolist()


def attention_mass_recovered(
    true_mass: torch.Tensor, selected_block_ids: list[int]
) -> float:
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


def spearman_rank_correlation(
    true_values: torch.Tensor, approx_values: torch.Tensor
) -> float | None:
    if true_values.numel() != approx_values.numel():
        raise ValueError("true_values and approx_values must have same length.")
    if true_values.numel() < 2:
        return None
    true_ranks = torch.tensor(
        _rank_positions_desc(true_values.tolist()), dtype=torch.float64
    )
    approx_ranks = torch.tensor(
        _rank_positions_desc(approx_values.tolist()), dtype=torch.float64
    )
    true_centered = true_ranks - true_ranks.mean()
    approx_centered = approx_ranks - approx_ranks.mean()
    denom = torch.linalg.vector_norm(true_centered) * torch.linalg.vector_norm(
        approx_centered
    )
    if float(denom.item()) == 0.0:
        return None
    return float(torch.dot(true_centered, approx_centered).item() / denom.item())


def _pad_and_mean(vectors: list[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        return torch.empty(0, dtype=torch.float32)
    max_len = max(int(vector.numel()) for vector in vectors)
    if max_len == 0:
        return torch.empty(0, dtype=torch.float32)
    numerator = torch.zeros(max_len, dtype=torch.float32)
    counts = torch.zeros(max_len, dtype=torch.float32)
    for vector in vectors:
        length = int(vector.numel())
        if length <= 0:
            continue
        numerator[:length] += vector.to(torch.float32)
        counts[:length] += 1.0
    valid = counts > 0
    output = torch.zeros(max_len, dtype=torch.float32)
    output[valid] = numerator[valid] / counts[valid]
    return output


def _backend_scores_for_eval_point(
    *,
    backend: str,
    query: torch.Tensor,
    keys: torch.Tensor,
    block_size: int,
    sketch_dim: int,
    seed: int,
) -> torch.Tensor:
    if backend == BACKEND_MEAN_KEY:
        return mean_key_block_scores(query, keys, block_size)
    if backend == BACKEND_MAX_TOKEN_SCORE_EXACT:
        return max_token_score_exact(query, keys, block_size)
    if backend in {BACKEND_RANDOM_PROJECTION, BACKEND_COUNTSKETCH}:
        return sketch_block_scores(
            query,
            keys,
            block_size=block_size,
            backend=backend,
            sketch_dim=sketch_dim,
            seed=seed,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def evaluate_backend_scores(
    *,
    backend: str,
    eval_points: list[EvalPoint],
    block_size: int,
    sketch_dim: int,
    seed: int,
    topks: list[int],
) -> dict[str, Any]:
    if not eval_points:
        raise ValueError("eval_points must not be empty.")
    true_mass_vectors: list[torch.Tensor] = []
    score_vectors: list[torch.Tensor] = []
    warnings: list[str] = []

    for eval_point in eval_points:
        true_attention = compute_true_attention_weights(eval_point.query, eval_point.keys)
        true_mass_vectors.append(aggregate_block_values(true_attention, block_size))
        score_vectors.append(
            _backend_scores_for_eval_point(
                backend=backend,
                query=eval_point.query,
                keys=eval_point.keys,
                block_size=block_size,
                sketch_dim=sketch_dim,
                seed=seed,
            )
        )

    true_mass = _pad_and_mean(true_mass_vectors)
    scores = _pad_and_mean(score_vectors)
    num_blocks = int(true_mass.numel())
    max_topk = max(topks)
    top_blocks_by_true_mass = topk_indices_desc(true_mass, max_topk)
    top_blocks_by_score = topk_indices_desc(scores, max_topk)
    topk_metrics: dict[str, Any] = {}
    for topk in topks:
        selected = topk_indices_desc(scores, topk)
        true_topk = topk_indices_desc(true_mass, topk)
        oracle_mass = attention_mass_recovered(true_mass, true_topk)
        recovered = attention_mass_recovered(true_mass, selected)
        topk_metrics[str(topk)] = {
            "attention_mass_recovered_by_score_topk": recovered,
            "oracle_true_topk_mass": oracle_mass,
            "recovery_fraction_of_oracle": (
                float(recovered / oracle_mass) if oracle_mass > 0 else None
            ),
            "topk_overlap_with_true_topk": topk_overlap(selected, true_topk),
            "retained_block_fraction": (
                float(len(selected) / num_blocks) if num_blocks > 0 else None
            ),
        }

    spearman = spearman_rank_correlation(true_mass, scores)
    if spearman is None:
        warnings.append("rank_correlation_spearman unavailable_or_degenerate")
    return {
        "score_by_block": scores.tolist(),
        "true_mass_by_block": true_mass.tolist(),
        "top_blocks_by_true_mass": top_blocks_by_true_mass,
        "top_blocks_by_score": top_blocks_by_score,
        "rank_correlation_spearman": spearman,
        "num_blocks": num_blocks,
        "topk_metrics": topk_metrics,
        "warnings": warnings,
    }


def build_output_payload(
    *,
    model: str,
    prompt_kind: str,
    token_count: int,
    block_size: int,
    num_blocks: int,
    selected_layers: list[int],
    selected_heads: list[int],
    selected_query_positions: list[int],
    backend_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt_kind": prompt_kind,
        "token_count": int(token_count),
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "selected_layers": [int(x) for x in selected_layers],
        "selected_heads": [int(x) for x in selected_heads],
        "selected_query_positions": [int(x) for x in selected_query_positions],
        "aggregation": "mean",
        "layer": (int(selected_layers[0]) if len(selected_layers) == 1 else None),
        "head": (int(selected_heads[0]) if len(selected_heads) == 1 else None),
        "query_position": (
            int(selected_query_positions[0])
            if len(selected_query_positions) == 1
            else None
        ),
        "backend_results": backend_results,
    }


def _parse_csv_ints(spec: str) -> list[int]:
    values = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"Expected at least one integer in spec: {spec!r}")
    return values


def resolve_layer_selection(layer_spec: str, num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if layer_spec == "last":
        return [num_layers - 1]
    if layer_spec == "last4":
        return list(range(max(0, num_layers - 4), num_layers))
    if layer_spec == "all":
        return list(range(num_layers))
    layers = _parse_csv_ints(layer_spec)
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {num_layers - 1}]")
    return layers


def resolve_head_selection(head_spec: str, num_heads: int) -> list[int]:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive.")
    if head_spec == "all":
        return list(range(num_heads))
    heads = _parse_csv_ints(head_spec)
    for head in heads:
        if head < 0 or head >= num_heads:
            raise ValueError(f"Head {head} out of range [0, {num_heads - 1}]")
    return heads


def resolve_query_position_selection(query_spec: str, seq_len: int) -> list[int]:
    if seq_len < 2:
        raise ValueError("Need at least 2 tokens to resolve query positions.")
    if query_spec == "last":
        return [seq_len - 1]
    if query_spec == "last4":
        start = max(1, seq_len - 4)
        return list(range(start, seq_len))
    return [
        _resolve_query_position(item.strip(), seq_len)
        for item in query_spec.split(",")
        if item.strip()
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline attention-mass evaluator for sketch scoring."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt-kind",
        choices=[
            "factual_recall",
            "factual_recall_long",
            "code_context",
            "code_context_long",
            "synthetic_long",
        ],
        default="factual_recall_long",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--layer", default="last")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--query-position", default="last")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--heads", default=None)
    parser.add_argument("--query-positions", default=None)
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--sketch-seed", type=int, default=123)
    parser.add_argument(
        "--topk",
        type=int,
        action="append",
        default=None,
        help="Repeatable top-k block budget; defaults to 4,8,16,32.",
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


def _resolve_topks(topks: list[int] | None) -> list[int]:
    resolved = topks if topks else DEFAULT_TOPKS
    return sorted({max(1, int(value)) for value in resolved})


def _resolve_backends(backend: str) -> list[str]:
    if backend == BACKEND_ALL:
        return [
            BACKEND_MEAN_KEY,
            BACKEND_RANDOM_PROJECTION,
            BACKEND_COUNTSKETCH,
            BACKEND_MAX_TOKEN_SCORE_EXACT,
        ]
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


def collect_eval_points(
    *,
    model: Any,
    input_ids: Any,
    selected_layers: list[int],
    selected_heads: list[int],
    selected_query_positions: list[int],
    extraction_mode: str,
) -> list[EvalPoint]:
    eval_points: list[EvalPoint] = []
    for layer in selected_layers:
        for head in selected_heads:
            for query_position in selected_query_positions:
                extraction = _extract_head_qk(
                    model=model,
                    input_ids=input_ids,
                    layer=layer,
                    head=head,
                    query_position=query_position,
                    extraction_mode=extraction_mode,
                )
                eval_points.append(
                    EvalPoint(
                        layer=layer,
                        head=head,
                        query_position=extraction.query_position,
                        query=torch.from_numpy(extraction.query).to(torch.float32),
                        keys=torch.from_numpy(extraction.keys).to(torch.float32),
                    )
                )
    return eval_points


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    topks = _resolve_topks(args.topk)
    layer_spec = args.layers if args.layers is not None else args.layer
    head_spec = args.heads if args.heads is not None else str(args.head)
    query_spec = (
        args.query_positions
        if args.query_positions is not None
        else args.query_position
    )
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

    layers = _get_transformer_layers(model)
    selected_layers = resolve_layer_selection(layer_spec, len(layers))
    first_attn = _get_layer_attention(layers[selected_layers[0]])
    num_heads = _get_num_query_heads(model, first_attn)
    selected_heads = resolve_head_selection(head_spec, num_heads)
    selected_query_positions = resolve_query_position_selection(query_spec, token_count)

    eval_points = collect_eval_points(
        model=model,
        input_ids=input_ids,
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        selected_query_positions=selected_query_positions,
        extraction_mode=args.extraction_mode,
    )
    backend_results: dict[str, Any] = {}
    for backend_name in _resolve_backends(args.backend):
        backend_results[backend_name] = evaluate_backend_scores(
            backend=backend_name,
            eval_points=eval_points,
            block_size=args.block_size,
            sketch_dim=args.sketch_dim,
            seed=args.sketch_seed,
            topks=topks,
        )

    num_blocks = max(result["num_blocks"] for result in backend_results.values())
    payload = build_output_payload(
        model=args.model,
        prompt_kind=args.prompt_kind,
        token_count=token_count,
        block_size=args.block_size,
        num_blocks=num_blocks,
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        selected_query_positions=selected_query_positions,
        backend_results=backend_results,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
