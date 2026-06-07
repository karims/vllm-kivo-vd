#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare full and selected-KV reference attention outside vLLM."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full and selected-KV attention on synthetic Q/K/V "
            "tensors outside vLLM."
        )
    )
    parser.add_argument("--num-query-heads", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--selected-blocks")
    parser.add_argument(
        "--selection-policy",
        choices=["recent", "first", "random", "oracle_topk"],
        default="recent",
    )
    parser.add_argument("--candidate-budget-blocks", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/"
            "phase10_0_selected_attention_equivalence.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/"
            "phase10_0_selected_attention_equivalence.md"
        ),
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def expand_kv_heads(
    tensor: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:
    num_kv_heads = tensor.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num query heads must be divisible by num KV heads"
        )
    repeats = num_query_heads // num_kv_heads
    if repeats == 1:
        return tensor
    return tensor.repeat_interleave(repeats, dim=1)


def scaled_dot_product_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("Q/K/V tensors must have shape [batch, heads, tokens, dim]")
    if keys.shape != values.shape:
        raise ValueError("K and V tensors must have identical shapes")
    if query.shape[0] != keys.shape[0]:
        raise ValueError("Q/K/V batch dimensions must match")
    if query.shape[1] != keys.shape[1]:
        raise ValueError("Q/K/V head dimensions must match")
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")

    scores = torch.matmul(query, keys.transpose(-2, -1))
    scores = scores / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    output = torch.matmul(weights, values)
    return output, weights


def gather_selected_blocks(
    tensor: torch.Tensor,
    selected_block_ids: list[int],
    block_size: int,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("KV tensor must have shape [batch, heads, tokens, dim]")
    if block_size <= 0 or tensor.shape[2] % block_size != 0:
        raise ValueError("KV token length must be divisible by block size")
    num_blocks = tensor.shape[2] // block_size
    if not selected_block_ids:
        raise ValueError("at least one selected block is required")
    if any(block_id < 0 or block_id >= num_blocks
           for block_id in selected_block_ids):
        raise ValueError("selected block ID is outside the KV block range")

    blocks = tensor.reshape(
        tensor.shape[0],
        tensor.shape[1],
        num_blocks,
        block_size,
        tensor.shape[3],
    )
    indices = torch.tensor(
        selected_block_ids,
        dtype=torch.long,
        device=tensor.device,
    )
    selected = blocks.index_select(2, indices)
    return selected.reshape(
        tensor.shape[0],
        tensor.shape[1],
        len(selected_block_ids) * block_size,
        tensor.shape[3],
    )


def attention_mass_by_block(
    attention_weights: torch.Tensor,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    if attention_weights.shape[-1] != num_blocks * block_size:
        raise ValueError("attention weights do not match the block layout")
    block_weights = attention_weights.reshape(
        *attention_weights.shape[:-1],
        num_blocks,
        block_size,
    ).sum(dim=-1)
    reduce_dims = tuple(range(block_weights.ndim - 1))
    return block_weights.mean(dim=reduce_dims)


def parse_selected_blocks(value: str | None, num_blocks: int) -> list[int] | None:
    if value is None:
        return None
    selected: list[int] = []
    seen: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        block_id = int(part)
        if block_id < 0 or block_id >= num_blocks:
            raise ValueError(
                f"selected block ID {block_id} is outside [0, {num_blocks})"
            )
        if block_id not in seen:
            selected.append(block_id)
            seen.add(block_id)
    if not selected:
        raise ValueError("--selected-blocks must contain at least one block ID")
    return selected


def select_block_ids(
    *,
    policy: str,
    num_blocks: int,
    candidate_budget_blocks: int,
    seed: int,
    block_attention_mass: torch.Tensor | None = None,
) -> list[int]:
    if candidate_budget_blocks <= 0:
        raise ValueError("--candidate-budget-blocks must be positive")
    budget = min(candidate_budget_blocks, num_blocks)
    if policy == "recent":
        return list(range(num_blocks - budget, num_blocks))
    if policy == "first":
        return list(range(budget))
    if policy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return sorted(
            torch.randperm(num_blocks, generator=generator)[:budget].tolist()
        )
    if policy == "oracle_topk":
        if block_attention_mass is None:
            raise ValueError("oracle_topk requires full attention weights")
        return sorted(
            torch.topk(block_attention_mass, k=budget).indices.tolist()
        )
    raise ValueError(f"unsupported selection policy: {policy}")


def captured_attention_mass(
    block_attention_mass: torch.Tensor,
    selected_block_ids: list[int],
) -> float:
    indices = torch.tensor(
        selected_block_ids,
        dtype=torch.long,
        device=block_attention_mass.device,
    )
    return float(block_attention_mass.index_select(0, indices).sum().item())


def calculate_metrics(
    full_output: torch.Tensor,
    selected_output: torch.Tensor,
) -> dict[str, float]:
    full = full_output.float().reshape(-1)
    selected = selected_output.float().reshape(-1)
    if full.shape != selected.shape:
        raise ValueError("full and selected outputs must have identical shapes")
    eps = torch.finfo(torch.float32).eps
    cosine = torch.nn.functional.cosine_similarity(
        full.unsqueeze(0),
        selected.unsqueeze(0),
        dim=1,
        eps=eps,
    )
    difference = selected - full
    return {
        "cosine_similarity": float(cosine.item()),
        "relative_l2_error": float(
            (torch.linalg.vector_norm(difference)
             / torch.linalg.vector_norm(full).clamp_min(eps)).item()
        ),
        "max_absolute_error": float(difference.abs().max().item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "full_output_norm": float(torch.linalg.vector_norm(full).item()),
        "selected_output_norm": float(
            torch.linalg.vector_norm(selected).item()
        ),
    }


def validate_config(args: argparse.Namespace) -> None:
    positive_values = {
        "--num-query-heads": args.num_query_heads,
        "--num-kv-heads": args.num_kv_heads,
        "--head-dim": args.head_dim,
        "--block-size": args.block_size,
        "--num-blocks": args.num_blocks,
        "--query-len": args.query_len,
        "--candidate-budget-blocks": args.candidate_budget_blocks,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError(
            "--num-query-heads must be divisible by --num-kv-heads"
        )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    validate_config(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    kv_length = args.num_blocks * args.block_size

    query = torch.randn(
        (1, args.num_query_heads, args.query_len, args.head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    keys = torch.randn(
        (1, args.num_kv_heads, kv_length, args.head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    values = torch.randn(
        (1, args.num_kv_heads, kv_length, args.head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    expanded_keys = expand_kv_heads(keys, args.num_query_heads)
    expanded_values = expand_kv_heads(values, args.num_query_heads)

    full_output, full_weights = scaled_dot_product_attention(
        query,
        expanded_keys,
        expanded_values,
    )
    block_mass = attention_mass_by_block(
        full_weights,
        args.num_blocks,
        args.block_size,
    )
    explicit_ids = parse_selected_blocks(
        args.selected_blocks,
        args.num_blocks,
    )
    if explicit_ids is not None:
        selected_ids = explicit_ids
        selection_source = "explicit"
    else:
        selected_ids = select_block_ids(
            policy=args.selection_policy,
            num_blocks=args.num_blocks,
            candidate_budget_blocks=args.candidate_budget_blocks,
            seed=args.seed,
            block_attention_mass=block_mass,
        )
        selection_source = args.selection_policy

    selected_keys = gather_selected_blocks(
        expanded_keys,
        selected_ids,
        args.block_size,
    )
    selected_values = gather_selected_blocks(
        expanded_values,
        selected_ids,
        args.block_size,
    )
    selected_output, _ = scaled_dot_product_attention(
        query,
        selected_keys,
        selected_values,
    )
    metrics = calculate_metrics(full_output, selected_output)
    metrics["attention_mass_captured"] = captured_attention_mass(
        block_mass,
        selected_ids,
    )
    selected_tokens = len(selected_ids) * args.block_size

    return {
        "config": {
            "num_query_heads": args.num_query_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "query_len": args.query_len,
            "kv_length": kv_length,
            "selection_policy": args.selection_policy,
            "selection_source": selection_source,
            "candidate_budget_blocks": args.candidate_budget_blocks,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
        },
        "selected_block_ids": selected_ids,
        "selected_block_count": len(selected_ids),
        "selected_token_count": selected_tokens,
        "selected_block_ratio": len(selected_ids) / args.num_blocks,
        "selected_token_ratio": selected_tokens / kv_length,
        "full_output_shape": list(full_output.shape),
        "selected_output_shape": list(selected_output.shape),
        "metrics": metrics,
        "caveats": {
            "synthetic_qkv": True,
            "outside_vllm": True,
            "outside_attention_kernel": True,
            "no_real_model_quality": True,
            "measured_runtime_reduction": False,
            "active_routing": False,
        },
    }


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _append_table(
    lines: list[str],
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append(
            "| " + " | ".join(f"`{_format(value)}`" for value in row) + " |"
        )


def render_markdown(report: dict[str, Any]) -> str:
    config = report["config"]
    metrics = report["metrics"]
    lines = [
        "# Kivo-VD Phase 10.0 Selected-Attention Equivalence",
        "",
        "**Status:** Standalone synthetic PyTorch reference experiment "
        "outside vLLM and outside production attention kernels.",
        "",
        "## Configuration",
        "",
    ]
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in config.items()],
    )
    lines.extend(["", "## Selected Blocks", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [
            ["selected_block_ids", report["selected_block_ids"]],
            ["selected_block_count", report["selected_block_count"]],
            ["selected_token_count", report["selected_token_count"]],
            ["selected_block_ratio", report["selected_block_ratio"]],
            ["selected_token_ratio", report["selected_token_ratio"]],
        ],
    )
    lines.extend(["", "## Metrics", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [[key, value] for key, value in metrics.items()],
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Oracle top-k is an undeployable upper-bound diagnostic based on full "
        "attention mass. Recent, first, and random are sanity policies rather "
        "than proposed runtime selectors.",
        "",
        "High oracle similarity with weak heuristic results indicates a "
        "selection-policy problem. Weak oracle similarity indicates that the "
        "candidate budget or selected-attention approximation itself may be "
        "insufficient for this synthetic case.",
        "",
        "## Caveats",
        "",
        "- Q/K/V tensors are synthetic.",
        "- The experiment runs outside vLLM.",
        "- The experiment runs outside production attention kernels.",
        "- No real model quality is measured.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency improvement is claimed.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        report = run_experiment(args)
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "selection_source": report["config"]["selection_source"],
                    "selected_block_count": report["selected_block_count"],
                    "selected_block_ratio": report["selected_block_ratio"],
                    "cosine_similarity": report["metrics"][
                        "cosine_similarity"
                    ],
                    "relative_l2_error": report["metrics"][
                        "relative_l2_error"
                    ],
                    "attention_mass_captured": report["metrics"][
                        "attention_mass_captured"
                    ],
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "synthetic_qkv": True,
                    "outside_vllm": True,
                    "active_routing": False,
                    "measured_runtime_reduction": False,
                    "no_real_model_quality": True,
                },
                separators=(",", ":"),
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
