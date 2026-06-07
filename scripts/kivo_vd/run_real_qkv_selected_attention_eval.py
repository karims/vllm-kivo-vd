#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare full and selected attention on real GPT-2 Q/K/V projections."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def default_prompt() -> str:
    filler = (
        "Transformer inference stores key and value vectors for prior tokens. "
        "Candidate block retrieval may reduce the active working set, but "
        "correctness must be evaluated before changing attention behavior. "
    )
    return (
        "The secret retrieval key is BLUE ORCHID. "
        + filler * 12
        + "What is the secret retrieval key?"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full and selected attention on real GPT-2 Q/K/V "
            "projections outside vLLM."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-file")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--candidate-budget-blocks", type=int, default=16)
    parser.add_argument(
        "--selection-policy",
        choices=["recent", "first", "random", "oracle_topk"],
        default="recent",
    )
    parser.add_argument("--selected-blocks")
    parser.add_argument("--max-length", type=int, default=768)
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
            "phase10_1_real_qkv_selected_attention_eval.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/"
            "phase10_1_real_qkv_selected_attention_eval.md"
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


def read_prompts(
    prompt: str | None,
    prompts_file: str | None,
) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.append(prompt)
    if prompts_file:
        path = Path(prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"prompts file is missing: {path}")
        prompts.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        prompts.append(default_prompt())
    return prompts


def split_gpt2_fused_qkv(
    qkv: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if qkv.ndim != 3:
        raise ValueError("fused QKV must have shape [batch, tokens, 3*hidden]")
    if qkv.shape[-1] % 3 != 0:
        raise ValueError("fused QKV width must be divisible by three")
    hidden_size = qkv.shape[-1] // 3
    if hidden_size % num_heads != 0:
        raise ValueError("hidden size must be divisible by number of heads")
    head_dim = hidden_size // num_heads
    query, keys, values = qkv.split(hidden_size, dim=-1)

    def reshape(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            num_heads,
            head_dim,
        ).permute(0, 2, 1, 3).contiguous()

    return reshape(query), reshape(keys), reshape(values)


def extract_gpt2_qkv(
    model: Any,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    layers = model.transformer.h
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(
            f"layer {layer_idx} is outside [0, {len(layers)})"
        )
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    if outputs.hidden_states is None:
        raise RuntimeError("model did not return hidden states")
    block = layers[layer_idx]
    attention_input = block.ln_1(outputs.hidden_states[layer_idx])
    fused_qkv = block.attn.c_attn(attention_input)
    num_heads = int(model.config.n_head)
    query, keys, values = split_gpt2_fused_qkv(
        fused_qkv,
        num_heads,
    )
    return query, keys, values, {
        "num_heads": num_heads,
        "head_dim": query.shape[-1],
        "hidden_size": query.shape[-1] * num_heads,
    }


def last_query_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("Q/K/V must have shape [batch, heads, tokens, dim]")
    if keys.shape != values.shape:
        raise ValueError("K and V shapes must match")
    if query.shape[:2] != keys.shape[:2]:
        raise ValueError("Q/K/V batch and head dimensions must match")
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    last_query = query[:, :, -1:, :]
    scores = torch.matmul(last_query, keys.transpose(-2, -1))
    scores = scores / math.sqrt(query.shape[-1])
    probabilities = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    output = torch.matmul(probabilities, values)
    return output, probabilities


def num_blocks_for_tokens(token_count: int, block_size: int) -> int:
    if token_count <= 0 or block_size <= 0:
        raise ValueError("token count and block size must be positive")
    return math.ceil(token_count / block_size)


def block_attention_mass(
    probabilities: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError(
            "attention probabilities must have shape "
            "[batch, heads, queries, tokens]"
        )
    token_count = probabilities.shape[-1]
    num_blocks = num_blocks_for_tokens(token_count, block_size)
    masses = []
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = min(start + block_size, token_count)
        masses.append(probabilities[..., start:end].sum(dim=-1).mean())
    return torch.stack(masses)


def gather_selected_blocks(
    tensor: torch.Tensor,
    selected_block_ids: list[int],
    block_size: int,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("KV tensor must have shape [batch, heads, tokens, dim]")
    num_blocks = num_blocks_for_tokens(tensor.shape[2], block_size)
    if not selected_block_ids:
        raise ValueError("at least one selected block is required")
    if any(
        block_id < 0 or block_id >= num_blocks
        for block_id in selected_block_ids
    ):
        raise ValueError("selected block ID is outside the KV block range")
    pieces = [
        tensor[
            :,
            :,
            block_id * block_size:
            min((block_id + 1) * block_size, tensor.shape[2]),
            :,
        ]
        for block_id in selected_block_ids
    ]
    return torch.cat(pieces, dim=2)


def parse_selected_blocks(
    value: str | None,
    num_blocks: int,
) -> list[int] | None:
    if value is None:
        return None
    result: list[int] = []
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
            result.append(block_id)
            seen.add(block_id)
    if not result:
        raise ValueError("--selected-blocks must include at least one ID")
    return result


def select_block_ids(
    *,
    policy: str,
    num_blocks: int,
    candidate_budget_blocks: int,
    seed: int,
    masses: torch.Tensor | None = None,
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
        if masses is None:
            raise ValueError("oracle_topk requires full attention mass")
        return sorted(torch.topk(masses, k=budget).indices.tolist())
    raise ValueError(f"unsupported selection policy: {policy}")


def captured_attention_mass(
    masses: torch.Tensor,
    selected_block_ids: list[int],
) -> float:
    indices = torch.tensor(
        selected_block_ids,
        device=masses.device,
        dtype=torch.long,
    )
    return float(masses.index_select(0, indices).sum().item())


def calculate_metrics(
    full_output: torch.Tensor,
    selected_output: torch.Tensor,
) -> dict[str, float]:
    full = full_output.float().reshape(-1)
    selected = selected_output.float().reshape(-1)
    if full.shape != selected.shape:
        raise ValueError("full and selected outputs must have identical shapes")
    eps = torch.finfo(torch.float32).eps
    difference = selected - full
    cosine = torch.nn.functional.cosine_similarity(
        full.unsqueeze(0),
        selected.unsqueeze(0),
        dim=1,
        eps=eps,
    )
    return {
        "cosine_similarity": float(cosine.item()),
        "relative_l2_error": float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(full).clamp_min(eps)
            ).item()
        ),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "max_absolute_error": float(difference.abs().max().item()),
        "full_output_norm": float(torch.linalg.vector_norm(full).item()),
        "selected_output_norm": float(
            torch.linalg.vector_norm(selected).item()
        ),
    }


def load_hf_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install it in the optional HF "
            "environment"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    if not (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
        and hasattr(model.config, "n_head")
    ):
        raise ValueError(
            "Phase 10.1 currently supports GPT-2-style models only"
        )
    return tokenizer, model


def evaluate_prompt(
    *,
    prompt: str,
    prompt_index: int,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    token_count = int(input_ids.shape[1])
    if token_count <= 0:
        raise ValueError("prompt tokenized to zero tokens")
    query, keys, values, projection_metadata = extract_gpt2_qkv(
        model,
        input_ids,
        args.layer_idx,
    )
    full_output, full_probabilities = last_query_attention(
        query,
        keys,
        values,
    )
    masses = block_attention_mass(full_probabilities, args.block_size)
    num_blocks = int(masses.shape[0])
    explicit = parse_selected_blocks(args.selected_blocks, num_blocks)
    if explicit is not None:
        selected_ids = explicit
        selection_source = "explicit"
    else:
        selected_ids = select_block_ids(
            policy=args.selection_policy,
            num_blocks=num_blocks,
            candidate_budget_blocks=args.candidate_budget_blocks,
            seed=args.seed + prompt_index,
            masses=masses,
        )
        selection_source = args.selection_policy
    selected_keys = gather_selected_blocks(
        keys,
        selected_ids,
        args.block_size,
    )
    selected_values = gather_selected_blocks(
        values,
        selected_ids,
        args.block_size,
    )
    selected_query = query[:, :, -1:, :]
    selected_output, _ = last_query_attention(
        selected_query,
        selected_keys,
        selected_values,
    )
    selected_token_count = int(selected_keys.shape[2])
    metrics = calculate_metrics(full_output, selected_output)
    metrics["attention_mass_captured"] = captured_attention_mass(
        masses,
        selected_ids,
    )
    return {
        "prompt_index": prompt_index,
        "prompt_preview": prompt[:120],
        "token_length": token_count,
        "layer_index": args.layer_idx,
        "block_count": num_blocks,
        "block_size": args.block_size,
        "selection_source": selection_source,
        "selected_block_ids": selected_ids,
        "selected_block_count": len(selected_ids),
        "selected_token_count": selected_token_count,
        "selected_block_ratio": len(selected_ids) / num_blocks,
        "selected_token_ratio": selected_token_count / token_count,
        "projection_metadata": projection_metadata,
        "full_output_shape": list(full_output.shape),
        "selected_output_shape": list(selected_output.shape),
        **metrics,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("at least one prompt row is required")
    return {
        "num_prompts": len(rows),
        "average_cosine_similarity": sum(
            row["cosine_similarity"] for row in rows
        ) / len(rows),
        "average_relative_l2_error": sum(
            row["relative_l2_error"] for row in rows
        ) / len(rows),
        "average_attention_mass_captured": sum(
            row["attention_mass_captured"] for row in rows
        ) / len(rows),
        "min_cosine_similarity": min(
            row["cosine_similarity"] for row in rows
        ),
        "max_relative_l2_error": max(
            row["relative_l2_error"] for row in rows
        ),
    }


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "--block-size": args.block_size,
        "--candidate-budget-blocks": args.candidate_budget_blocks,
        "--max-length": args.max_length,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.layer_idx < 0:
        raise ValueError("--layer-idx must be non-negative")


def build_report(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_rows(rows)
    return {
        "config": config,
        "aggregate": aggregate,
        "per_prompt": rows,
        # Retain the original Phase 10.1 names for existing readers.
        "aggregate_metrics": aggregate,
        "per_prompt_rows": rows,
        "caveats": {
            "real_model_qkv": True,
            "outside_vllm": True,
            "outside_attention_kernel": True,
            "no_generation_quality": True,
            "no_logits_eval": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
        },
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    prompts = read_prompts(args.prompt, args.prompts_file)
    tokenizer, model = load_hf_model(args.model, device, dtype)
    rows = [
        evaluate_prompt(
            prompt=prompt,
            prompt_index=index,
            tokenizer=tokenizer,
            model=model,
            args=args,
            device=device,
        )
        for index, prompt in enumerate(prompts)
    ]
    return build_report(
        config={
            "model": args.model,
            "layer_index": args.layer_idx,
            "block_size": args.block_size,
            "candidate_budget_blocks": args.candidate_budget_blocks,
            "selection_policy": args.selection_policy,
            "explicit_selected_blocks": args.selected_blocks,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "query_position": "last",
            "qkv_space": "gpt2_projection_after_ln_1",
        },
        rows=rows,
    )


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
    aggregate = report.get("aggregate", report.get("aggregate_metrics"))
    per_prompt = report.get("per_prompt", report.get("per_prompt_rows"))
    if not isinstance(aggregate, dict) or not isinstance(per_prompt, list):
        raise ValueError("report lacks aggregate or per_prompt results")
    lines = [
        "# Kivo-VD Phase 10.1 Real-QKV Selected-Attention Evaluation",
        "",
        "**Status:** Real GPT-2 projected Q/K/V, evaluated with standalone "
        "PyTorch attention outside vLLM and production attention kernels.",
        "",
        "## Configuration",
        "",
    ]
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in report["config"].items()],
    )
    lines.extend(["", "## Aggregate Metrics", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [
            [key, value]
            for key, value in aggregate.items()
        ],
    )
    lines.extend(["", "## Per-Prompt Results", ""])
    _append_table(
        lines,
        [
            "prompt",
            "tokens",
            "blocks",
            "selected",
            "block ratio",
            "mass",
            "cosine",
            "relative L2",
            "mean abs",
            "max abs",
        ],
        [
            [
                row["prompt_index"],
                row["token_length"],
                row["block_count"],
                row["selected_block_count"],
                row["selected_block_ratio"],
                row["attention_mass_captured"],
                row["cosine_similarity"],
                row["relative_l2_error"],
                row["mean_absolute_error"],
                row["max_absolute_error"],
            ]
            for row in per_prompt
        ],
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Oracle top-k uses full attention probabilities and is therefore an "
        "undeployable upper-bound diagnostic. Weak oracle results would make "
        "selected attention risky even under best-case block selection. "
        "Strong oracle results with weak heuristic policies would instead "
        "point to candidate selection as the limiting problem.",
        "",
        "These output-vector comparisons are not generation or logits "
        "quality measurements and do not establish end-to-end model behavior.",
        "",
        "## Caveats",
        "",
        "- Q/K/V projections come from a real GPT-2-style model.",
        "- Evaluation runs outside vLLM.",
        "- Evaluation runs outside production attention kernels.",
        "- No generation quality is measured.",
        "- No logits evaluation is performed.",
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
        report = run_evaluation(args)
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    **report["aggregate"],
                    "model": args.model,
                    "selection_policy": args.selection_policy,
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "real_model_qkv": True,
                    "outside_vllm": True,
                    "no_generation_quality": True,
                    "active_routing": False,
                    "measured_runtime_reduction": False,
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
