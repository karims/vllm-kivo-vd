#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Measure next-token logit sensitivity to one selected-attention patch."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

SELECTION_POLICIES = {
    "recent",
    "oracle_topk",
    "query_key_block_score",
    "count_sketch",
    "random_projection",
    "bidiagonal_sign_subsample",
}


def _load_selected_attention_helpers() -> Any:
    module_path = (
        Path(__file__).resolve().parent
        / "run_real_qkv_selected_attention_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_real_qkv_selected_attention_eval",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Phase 10 helpers: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch one GPT-2 layer's last-token attention output and compare "
            "next-token logits outside vLLM."
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
        choices=sorted(SELECTION_POLICIES),
        default="query_key_block_score",
    )
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument(
        "--block-score-reduction",
        choices=["max", "mean", "logsumexp"],
        default="max",
    )
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
        default="outputs/kivo_vd/phase11_0_logit_sensitivity.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase11_0_logit_sensitivity.md",
    )
    return parser.parse_args(argv)


def built_in_prompts() -> list[str]:
    fillers = {
        "retrieval": (
            "A retrieval system compares memory blocks before exact "
            "reranking. "
        ),
        "systems": (
            "A systems engineer checks scheduler traces, allocator state, "
            "cache residency, and reproducible diagnostics. "
        ),
        "code": (
            "A Python function validates input, transforms records, handles "
            "errors, and returns deterministic output. "
        ),
        "failure": (
            "Later paragraphs contain distractors about weather, gardens, "
            "books, transit, and office supplies. "
        ),
        "context": (
            "Long-context attention balances local continuity with retrieval "
            "of information introduced earlier. "
        ),
    }
    return [
        (
            "The secret retrieval key is BLUE ORCHID. "
            + fillers["retrieval"] * 22
            + "What is the secret retrieval key?"
        ),
        (
            "The first diagnostic step is CHECK CUDA AVAILABILITY. "
            + fillers["systems"] * 22
            + "What is the first diagnostic step?"
        ),
        (
            "The function should return the sentinel value 731. "
            + fillers["code"] * 22
            + "What sentinel value should the function return?"
        ),
        (
            "Important early token: AMBER COMPASS. "
            + fillers["failure"] * 26
            + "Which important token appeared near the beginning?"
        ),
        (
            "The central principle is exact reranking after candidate search. "
            + fillers["context"] * 24
            + "What is the central principle?"
        ),
    ]


def read_prompts(
    prompt: str | None,
    prompts_file: str | None,
) -> list[str]:
    prompts = [prompt] if prompt else []
    if prompts_file:
        path = Path(prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"prompts file is missing: {path}")
        prompts.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts or built_in_prompts()


def topk_overlap(
    baseline_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    k: int,
) -> int:
    if baseline_logits.shape != patched_logits.shape:
        raise ValueError("baseline and patched logits must have equal shapes")
    if baseline_logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if k <= 0:
        raise ValueError("k must be positive")
    effective_k = min(k, baseline_logits.numel())
    baseline_ids = set(
        torch.topk(baseline_logits, effective_k).indices.tolist()
    )
    patched_ids = set(torch.topk(patched_logits, effective_k).indices.tolist())
    return len(baseline_ids & patched_ids)


def kl_divergence_from_logits(
    baseline_logits: torch.Tensor,
    patched_logits: torch.Tensor,
) -> float:
    if baseline_logits.shape != patched_logits.shape:
        raise ValueError("baseline and patched logits must have equal shapes")
    baseline_log_probs = torch.log_softmax(baseline_logits.float(), dim=-1)
    patched_log_probs = torch.log_softmax(patched_logits.float(), dim=-1)
    baseline_probs = baseline_log_probs.exp()
    divergence = torch.sum(
        baseline_probs * (baseline_log_probs - patched_log_probs)
    )
    return float(divergence.clamp_min(0).item())


def compare_logits(
    baseline_logits: torch.Tensor,
    patched_logits: torch.Tensor,
) -> dict[str, Any]:
    if baseline_logits.shape != patched_logits.shape:
        raise ValueError("baseline and patched logits must have equal shapes")
    baseline = baseline_logits.float().reshape(-1)
    patched = patched_logits.float().reshape(-1)
    eps = torch.finfo(torch.float32).eps
    difference = patched - baseline
    cosine = torch.nn.functional.cosine_similarity(
        baseline.unsqueeze(0),
        patched.unsqueeze(0),
        dim=1,
        eps=eps,
    )
    baseline_probs = torch.softmax(baseline, dim=-1)
    patched_probs = torch.softmax(patched, dim=-1)
    baseline_top = int(torch.argmax(baseline).item())
    patched_top = int(torch.argmax(patched).item())
    return {
        "logits_cosine_similarity": float(cosine.item()),
        "logits_relative_l2_error": float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(baseline).clamp_min(eps)
            ).item()
        ),
        "kl_divergence": kl_divergence_from_logits(baseline, patched),
        "top1_token_match": baseline_top == patched_top,
        "top5_overlap_count": topk_overlap(baseline, patched, 5),
        "top10_overlap_count": topk_overlap(baseline, patched, 10),
        "baseline_top_token_id": baseline_top,
        "patched_top_token_id": patched_top,
        "baseline_top_token_probability": float(
            baseline_probs[baseline_top].item()
        ),
        "patched_top_token_probability": float(
            patched_probs[patched_top].item()
        ),
        "baseline_top_token_probability_after_patch": float(
            patched_probs[baseline_top].item()
        ),
        "baseline_top_token_probability_delta": float(
            abs(
                baseline_probs[baseline_top].item()
                - patched_probs[baseline_top].item()
            )
        ),
    }


def _merge_attention_heads(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 4:
        raise ValueError("attention output must be [batch, heads, tokens, dim]")
    return output.permute(0, 2, 1, 3).contiguous().reshape(
        output.shape[0],
        output.shape[2],
        output.shape[1] * output.shape[3],
    )


def _initial_hidden_states(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    position_ids = torch.arange(
        input_ids.shape[1],
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    token_embeddings = model.transformer.wte(input_ids)
    position_embeddings = model.transformer.wpe(position_ids)
    return model.transformer.drop(token_embeddings + position_embeddings)


def _block_hidden_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        hidden_states = output[0]
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states
    raise TypeError("GPT-2 block returned an unsupported output shape")


def patched_next_token_logits(
    *,
    model: Any,
    input_ids: torch.Tensor,
    layer_idx: int,
    block_size: int,
    candidate_budget_blocks: int,
    selection_policy: str,
    sketch_dim: int,
    block_score_reduction: str,
    seed: int,
    helpers: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    layers = model.transformer.h
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer {layer_idx} is outside [0, {len(layers)})")

    hidden_states = _initial_hidden_states(model, input_ids)
    attention_metrics: dict[str, Any] | None = None
    with torch.no_grad():
        for current_idx, block in enumerate(layers):
            if current_idx != layer_idx:
                hidden_states = _block_hidden_output(
                    block(hidden_states, use_cache=False)
                )
                continue

            normal_block_output = _block_hidden_output(
                block(hidden_states, use_cache=False)
            )
            residual_last = hidden_states[:, -1:, :]
            attention_input = block.ln_1(hidden_states)
            fused_qkv = block.attn.c_attn(attention_input)
            query, keys, values = helpers.split_gpt2_fused_qkv(
                fused_qkv,
                int(model.config.n_head),
            )
            full_attention, probabilities = helpers.last_query_attention(
                query,
                keys,
                values,
            )
            masses = helpers.block_attention_mass(probabilities, block_size)
            selected_ids, selector_info = (
                helpers.select_block_ids_for_policy(
                    policy=selection_policy,
                    num_blocks=int(masses.shape[0]),
                    candidate_budget_blocks=candidate_budget_blocks,
                    seed=seed,
                    masses=masses,
                    query=query,
                    keys=keys,
                    block_size=block_size,
                    sketch_dim=sketch_dim,
                    block_score_reduction=block_score_reduction,
                )
            )
            selected_keys = helpers.gather_selected_blocks(
                keys, selected_ids, block_size
            )
            selected_values = helpers.gather_selected_blocks(
                values, selected_ids, block_size
            )
            selected_attention, _ = helpers.last_query_attention(
                query[:, :, -1:, :],
                selected_keys,
                selected_values,
            )
            projected_attention = block.attn.c_proj(
                _merge_attention_heads(selected_attention)
            )
            projected_attention = block.attn.resid_dropout(
                projected_attention
            )
            patched_last = residual_last + projected_attention
            patched_last = patched_last + block.mlp(
                block.ln_2(patched_last)
            )
            hidden_states = normal_block_output.clone()
            hidden_states[:, -1:, :] = patched_last
            attention_metrics = {
                **helpers.calculate_metrics(
                    full_attention,
                    selected_attention,
                ),
                "selected_block_ids": selected_ids,
                "selected_block_count": len(selected_ids),
                "block_count": int(masses.shape[0]),
                "selected_block_ratio": len(selected_ids) / masses.shape[0],
                "attention_mass_captured": (
                    helpers.captured_attention_mass(masses, selected_ids)
                ),
                **selector_info,
            }

        hidden_states = model.transformer.ln_f(hidden_states)
        logits = model.lm_head(hidden_states[:, -1, :])
    if attention_metrics is None:
        raise RuntimeError("selected layer patch was not applied")
    return logits, attention_metrics


def load_hf_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Phase 11.0"
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
        raise ValueError("Phase 11.0 currently supports GPT-2-style models")
    return tokenizer, model


def evaluate_prompt(
    *,
    prompt: str,
    prompt_index: int,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
    helpers: Any,
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        baseline_logits = model(
            input_ids=input_ids,
            use_cache=False,
        ).logits[:, -1, :]
    patched_logits, attention = patched_next_token_logits(
        model=model,
        input_ids=input_ids,
        layer_idx=args.layer_idx,
        block_size=args.block_size,
        candidate_budget_blocks=args.candidate_budget_blocks,
        selection_policy=args.selection_policy,
        sketch_dim=args.sketch_dim,
        block_score_reduction=args.block_score_reduction,
        seed=args.seed + prompt_index,
        helpers=helpers,
    )
    metrics = compare_logits(baseline_logits[0], patched_logits[0])
    baseline_token_id = metrics["baseline_top_token_id"]
    patched_token_id = metrics["patched_top_token_id"]
    return {
        "prompt_index": prompt_index,
        "prompt_preview": prompt[:120],
        "token_length": int(input_ids.shape[1]),
        "layer_index": args.layer_idx,
        "policy": args.selection_policy,
        "selected_blocks": attention["selected_block_ids"],
        "selected_block_count": attention["selected_block_count"],
        "selected_block_ratio": attention["selected_block_ratio"],
        "attention_mass_captured": attention["attention_mass_captured"],
        "attention_output_cosine_similarity": attention[
            "cosine_similarity"
        ],
        "attention_output_relative_l2_error": attention[
            "relative_l2_error"
        ],
        "baseline_top_token": tokenizer.decode([baseline_token_id]),
        "patched_top_token": tokenizer.decode([patched_token_id]),
        **metrics,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one prompt row is required")

    def average(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    return {
        "num_prompts": len(rows),
        "average_logits_cosine_similarity": average(
            "logits_cosine_similarity"
        ),
        "average_logits_relative_l2_error": average(
            "logits_relative_l2_error"
        ),
        "average_kl_divergence": average("kl_divergence"),
        "top1_match_rate": sum(
            bool(row["top1_token_match"]) for row in rows
        ) / len(rows),
        "average_top5_overlap": average("top5_overlap_count"),
        "average_top10_overlap": average("top10_overlap_count"),
        "average_attention_output_cosine": average(
            "attention_output_cosine_similarity"
        ),
        "average_attention_output_relative_l2": average(
            "attention_output_relative_l2_error"
        ),
    }


def build_report(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "config": config,
        "aggregate": aggregate_rows(rows),
        "per_prompt": rows,
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "single_layer_patch_only": True,
            "no_measured_runtime_reduction": True,
            "no_latency_claim": True,
            "generation_quality_not_fully_measured": True,
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
    lines = [
        "# Kivo-VD Phase 11.0 Logit Sensitivity",
        "",
        "**Status:** Single-layer, last-token selected-attention patch on a "
        "real GPT-2 model outside vLLM.",
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
        [[key, value] for key, value in report["aggregate"].items()],
    )
    lines.extend(["", "## Per-Prompt Results", ""])
    _append_table(
        lines,
        [
            "prompt",
            "tokens",
            "selected",
            "attention cosine",
            "attention rel L2",
            "logits cosine",
            "logits rel L2",
            "KL",
            "top-1 match",
            "top-5 overlap",
            "top-10 overlap",
            "baseline top",
            "patched top",
        ],
        [
            [
                row["prompt_index"],
                row["token_length"],
                row["selected_block_count"],
                row["attention_output_cosine_similarity"],
                row["attention_output_relative_l2_error"],
                row["logits_cosine_similarity"],
                row["logits_relative_l2_error"],
                row["kl_divergence"],
                row["top1_token_match"],
                row["top5_overlap_count"],
                row["top10_overlap_count"],
                row["baseline_top_token"],
                row["patched_top_token"],
            ]
            for row in report["per_prompt"]
        ],
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "High top-1 agreement and low KL for `query_key_block_score` would "
        "support broader standalone quality tests. If oracle remains stable "
        "but the deployable selector does not, selection is still the "
        "bottleneck. If oracle itself changes logits substantially, selected "
        "attention may be too risky at that layer and budget.",
        "",
        "## Caveats",
        "",
        "- Evaluation runs outside vLLM.",
        "- No vLLM integration is implemented or authorized.",
        "- Only one layer and the last-token attention output are patched.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency claim is made.",
        "- Full generation quality is not measured.",
        "",
        "## Next Steps",
        "",
        "Compare query-key and oracle policies across layers and practical "
        "budgets before considering multi-layer or generation-level tests.",
    ])
    return "\n".join(lines) + "\n"


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--block-size": args.block_size,
        "--candidate-budget-blocks": args.candidate_budget_blocks,
        "--sketch-dim": args.sketch_dim,
        "--max-length": args.max_length,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.layer_idx < 0:
        raise ValueError("--layer-idx must be non-negative")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    helpers = _load_selected_attention_helpers()
    torch.manual_seed(args.seed)
    device = helpers.resolve_device(args.device)
    dtype = helpers.resolve_dtype(args.dtype)
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
            helpers=helpers,
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
            "sketch_dim": args.sketch_dim,
            "block_score_reduction": args.block_score_reduction,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "patch_scope": "single_layer_last_token_attention_output",
        },
        rows=rows,
    )


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
                    "layer_index": args.layer_idx,
                    "selection_policy": args.selection_policy,
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    **report["caveats"],
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
