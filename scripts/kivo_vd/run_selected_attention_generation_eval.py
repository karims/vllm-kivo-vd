#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare baseline and selected-attention greedy generation outside vLLM."""

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


def _load_phase11() -> Any:
    module_path = (
        Path(__file__).resolve().parent
        / "run_selected_attention_logit_sensitivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_selected_attention_logit_sensitivity",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Phase 11.0 helpers: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and single-layer selected-attention greedy "
            "generation on GPT-2 outside vLLM."
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
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--teacher-forced-context", action="store_true")
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
        default="outputs/kivo_vd/phase11_2_generation_eval.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase11_2_generation_eval.md",
    )
    return parser.parse_args(argv)


def prefix_match_length(
    baseline_ids: list[int],
    patched_ids: list[int],
) -> int:
    length = 0
    for baseline_id, patched_id in zip(baseline_ids, patched_ids):
        if baseline_id != patched_id:
            break
        length += 1
    return length


def edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1]
                + (left_value != right_value),
            ))
        previous = current
    return previous[-1]


def normalized_edit_distance(
    baseline_ids: list[int],
    patched_ids: list[int],
) -> float:
    denominator = max(len(baseline_ids), len(patched_ids), 1)
    return edit_distance(baseline_ids, patched_ids) / denominator


def compare_token_sequences(
    baseline_ids: list[int],
    patched_ids: list[int],
) -> dict[str, Any]:
    overlap_length = min(len(baseline_ids), len(patched_ids))
    matched = sum(
        baseline_ids[index] == patched_ids[index]
        for index in range(overlap_length)
    )
    denominator = max(len(baseline_ids), len(patched_ids), 1)
    prefix_length = prefix_match_length(baseline_ids, patched_ids)
    first_mismatch = (
        prefix_length
        if prefix_length < max(len(baseline_ids), len(patched_ids))
        else None
    )
    return {
        "exact_token_sequence_match": baseline_ids == patched_ids,
        "prefix_match_length": prefix_length,
        "token_match_rate": matched / denominator,
        "first_mismatch_index": first_mismatch,
        "normalized_edit_distance": normalized_edit_distance(
            baseline_ids,
            patched_ids,
        ),
    }


def generate_sequences(
    *,
    model: Any,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    phase11: Any,
    helpers: Any,
) -> dict[str, Any]:
    baseline_context = input_ids.clone()
    patched_context = input_ids.clone()
    baseline_generated: list[int] = []
    patched_generated: list[int] = []
    step_kl: list[float] = []
    step_top1: list[float] = []
    selected_ratios: list[float] = []

    for step in range(args.max_new_tokens):
        if baseline_context.shape[1] >= args.max_length:
            break
        with torch.no_grad():
            baseline_logits = model(
                input_ids=baseline_context,
                use_cache=False,
            ).logits[:, -1, :]
        patched_input = (
            baseline_context
            if args.teacher_forced_context
            else patched_context
        )
        if patched_input.shape[1] >= args.max_length:
            break
        patched_logits, attention = phase11.patched_next_token_logits(
            model=model,
            input_ids=patched_input,
            layer_idx=args.layer_idx,
            block_size=args.block_size,
            candidate_budget_blocks=args.candidate_budget_blocks,
            selection_policy=args.selection_policy,
            sketch_dim=args.sketch_dim,
            block_score_reduction=args.block_score_reduction,
            seed=args.seed + step,
            helpers=helpers,
        )
        baseline_next = int(torch.argmax(baseline_logits[0]).item())
        patched_next = int(torch.argmax(patched_logits[0]).item())
        baseline_generated.append(baseline_next)
        patched_generated.append(patched_next)
        step_kl.append(
            phase11.kl_divergence_from_logits(
                baseline_logits[0],
                patched_logits[0],
            )
        )
        step_top1.append(float(baseline_next == patched_next))
        selected_ratios.append(float(attention["selected_block_ratio"]))

        baseline_token = torch.tensor(
            [[baseline_next]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        patched_token = torch.tensor(
            [[patched_next]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        baseline_context = torch.cat(
            [baseline_context, baseline_token],
            dim=1,
        )
        if args.teacher_forced_context:
            patched_context = baseline_context
        else:
            patched_context = torch.cat(
                [patched_context, patched_token],
                dim=1,
            )

    return {
        "baseline_generated_token_ids": baseline_generated,
        "patched_generated_token_ids": patched_generated,
        "per_step_kl_divergence": step_kl,
        "per_step_top1_match": step_top1,
        "per_step_selected_block_ratio": selected_ratios,
    }


def evaluate_prompt(
    *,
    prompt: str,
    prompt_index: int,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
    phase11: Any,
    helpers: Any,
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    generation = generate_sequences(
        model=model,
        input_ids=input_ids,
        args=args,
        phase11=phase11,
        helpers=helpers,
    )
    baseline_ids = generation["baseline_generated_token_ids"]
    patched_ids = generation["patched_generated_token_ids"]
    sequence_metrics = compare_token_sequences(baseline_ids, patched_ids)
    step_kl = generation["per_step_kl_divergence"]
    step_top1 = generation["per_step_top1_match"]
    selected_ratios = generation["per_step_selected_block_ratio"]
    exact = sequence_metrics["exact_token_sequence_match"]
    return {
        "prompt_index": prompt_index,
        "prompt_preview": prompt[:120],
        "prompt_token_length": int(input_ids.shape[1]),
        "num_generated_tokens": len(baseline_ids),
        "baseline_generated_token_ids": baseline_ids,
        "patched_generated_token_ids": patched_ids,
        "baseline_generated_text": tokenizer.decode(baseline_ids),
        "patched_generated_text": tokenizer.decode(patched_ids),
        **sequence_metrics,
        "average_per_step_kl_divergence": (
            sum(step_kl) / len(step_kl) if step_kl else 0.0
        ),
        "average_per_step_top1_match": (
            sum(step_top1) / len(step_top1) if step_top1 else 0.0
        ),
        "average_selected_block_ratio": (
            sum(selected_ratios) / len(selected_ratios)
            if selected_ratios
            else 0.0
        ),
        "final_generated_text_comparison_note": (
            "exact token sequence match"
            if exact
            else (
                "teacher-forced logits comparison diverged"
                if args.teacher_forced_context
                else "free-running generation contexts diverged"
            )
        ),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one prompt row is required")

    def average(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    return {
        "num_prompts": len(rows),
        "exact_sequence_match_rate": sum(
            bool(row["exact_token_sequence_match"]) for row in rows
        ) / len(rows),
        "average_token_match_rate": average("token_match_rate"),
        "average_prefix_match_length": average("prefix_match_length"),
        "average_normalized_edit_distance": average(
            "normalized_edit_distance"
        ),
        "average_per_step_kl_divergence": average(
            "average_per_step_kl_divergence"
        ),
        "average_per_step_top1_match_rate": average(
            "average_per_step_top1_match"
        ),
        "average_selected_block_ratio": average(
            "average_selected_block_ratio"
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
            "greedy_generation_only": True,
            "no_measured_runtime_reduction": True,
            "no_latency_claim": True,
            "generation_quality_probe_only": True,
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
        "# Kivo-VD Phase 11.2 Generation Evaluation",
        "",
        "**Status:** Single-layer selected-attention greedy-generation probe "
        "on GPT-2 outside vLLM.",
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
    lines.extend(["", "## Per-Prompt Metrics", ""])
    _append_table(
        lines,
        [
            "prompt",
            "prompt tokens",
            "generated",
            "exact",
            "prefix",
            "token match",
            "edit distance",
            "avg KL",
            "step top-1",
            "selected ratio",
        ],
        [
            [
                row["prompt_index"],
                row["prompt_token_length"],
                row["num_generated_tokens"],
                row["exact_token_sequence_match"],
                row["prefix_match_length"],
                row["token_match_rate"],
                row["normalized_edit_distance"],
                row["average_per_step_kl_divergence"],
                row["average_per_step_top1_match"],
                row["average_selected_block_ratio"],
            ]
            for row in report["per_prompt"]
        ],
    )
    lines.extend(["", "## Generation Examples", ""])
    _append_table(
        lines,
        ["prompt", "baseline", "patched", "note"],
        [
            [
                row["prompt_index"],
                row["baseline_generated_text"],
                row["patched_generated_text"],
                row["final_generated_text_comparison_note"],
            ]
            for row in report["per_prompt"]
        ],
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "High exact-match and token-match rates with low divergence support "
        "broader offline tests. A stable oracle with an unstable deployable "
        "selector identifies selector quality as the bottleneck. A divergent "
        "oracle indicates that selected attention may be risky at the tested "
        "layer and budget.",
        "",
        "## Caveats",
        "",
        "- Evaluation runs outside vLLM.",
        "- No vLLM integration is implemented or authorized.",
        "- Only one layer's last-token attention output is patched.",
        "- Generation uses greedy decoding only.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency claim is made.",
        "- This is a generation-quality probe, not a preservation claim.",
        "",
        "## Next Steps",
        "",
        "If practical-budget query-key and oracle runs remain stable, compare "
        "layers and longer prompts before considering multi-layer patches.",
    ])
    return "\n".join(lines) + "\n"


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--block-size": args.block_size,
        "--candidate-budget-blocks": args.candidate_budget_blocks,
        "--sketch-dim": args.sketch_dim,
        "--max-length": args.max_length,
        "--max-new-tokens": args.max_new_tokens,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.layer_idx < 0:
        raise ValueError("--layer-idx must be non-negative")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    phase11 = _load_phase11()
    helpers = phase11._load_selected_attention_helpers()
    torch.manual_seed(args.seed)
    device = helpers.resolve_device(args.device)
    dtype = helpers.resolve_dtype(args.dtype)
    prompts = phase11.read_prompts(args.prompt, args.prompts_file)
    tokenizer, model = phase11.load_hf_model(args.model, device, dtype)
    rows = [
        evaluate_prompt(
            prompt=prompt,
            prompt_index=index,
            tokenizer=tokenizer,
            model=model,
            args=args,
            device=device,
            phase11=phase11,
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
            "max_new_tokens": args.max_new_tokens,
            "teacher_forced_context": bool(args.teacher_forced_context),
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "patch_scope": "single_layer_last_token_each_decode_step",
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
                    "teacher_forced_context": bool(
                        args.teacher_forced_context
                    ),
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
