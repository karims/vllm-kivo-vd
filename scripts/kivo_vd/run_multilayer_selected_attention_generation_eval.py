#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Evaluate multi-layer selected-attention greedy generation outside vLLM."""

import argparse
import importlib.util
import json
from collections import defaultdict
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


def _load_script(filename: str, module_name: str) -> Any:
    module_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load helper script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_phase11() -> Any:
    return _load_script(
        "run_selected_attention_logit_sensitivity.py",
        "run_selected_attention_logit_sensitivity",
    )


def _load_phase11_generation() -> Any:
    return _load_script(
        "run_selected_attention_generation_eval.py",
        "run_selected_attention_generation_eval",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and multi-layer selected-attention greedy "
            "generation on GPT-2 outside vLLM."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-file")
    parser.add_argument("--layers", default="5,8")
    parser.add_argument("--budgets", default="8")
    parser.add_argument("--layer-budget-map")
    parser.add_argument("--block-size", type=int, default=16)
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
        default=(
            "outputs/kivo_vd/"
            "phase11_3_multilayer_generation_eval.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/"
            "phase11_3_multilayer_generation_eval.md"
        ),
    )
    return parser.parse_args(argv)


def parse_int_csv(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("comma-separated integer list must not be empty")
    return result


def parse_layer_budget_map(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "layer budget entries must use layer:budget syntax"
            )
        layer_text, budget_text = item.split(":", 1)
        layer = int(layer_text)
        budget = int(budget_text)
        if layer < 0 or budget <= 0:
            raise ValueError("layers must be non-negative and budgets positive")
        if layer in result:
            raise ValueError(f"duplicate layer in budget map: {layer}")
        result[layer] = budget
    if not result:
        raise ValueError("layer budget map must not be empty")
    return dict(sorted(result.items()))


def resolve_layer_budget_map(
    *,
    layers_value: str,
    budgets_value: str,
    map_value: str | None,
) -> dict[int, int]:
    if map_value:
        return parse_layer_budget_map(map_value)
    layers = parse_int_csv(layers_value)
    budgets = parse_int_csv(budgets_value)
    if any(layer < 0 for layer in layers):
        raise ValueError("layers must be non-negative")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must be positive")
    if len(set(layers)) != len(layers):
        raise ValueError("layers must not contain duplicates")
    if len(budgets) == 1:
        budgets = budgets * len(layers)
    elif len(budgets) != len(layers):
        raise ValueError(
            "budgets must contain one value or match the number of layers"
        )
    return dict(sorted(zip(layers, budgets)))


def patched_multilayer_next_token_logits(
    *,
    model: Any,
    input_ids: torch.Tensor,
    layer_budget_map: dict[int, int],
    block_size: int,
    selection_policy: str,
    sketch_dim: int,
    block_score_reduction: str,
    seed: int,
    phase11: Any,
    helpers: Any,
) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
    layers = model.transformer.h
    invalid = [
        layer for layer in layer_budget_map
        if layer < 0 or layer >= len(layers)
    ]
    if invalid:
        raise ValueError(
            f"layers {invalid} are outside [0, {len(layers)})"
        )
    hidden_states = phase11._initial_hidden_states(model, input_ids)
    per_layer: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for layer_idx, block in enumerate(layers):
            budget = layer_budget_map.get(layer_idx)
            if budget is None:
                hidden_states = phase11._block_hidden_output(
                    block(hidden_states, use_cache=False)
                )
                continue

            normal_output = phase11._block_hidden_output(
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
                    candidate_budget_blocks=budget,
                    seed=seed + layer_idx,
                    masses=masses,
                    query=query,
                    keys=keys,
                    block_size=block_size,
                    sketch_dim=sketch_dim,
                    block_score_reduction=block_score_reduction,
                )
            )
            selected_keys = helpers.gather_selected_blocks(
                keys,
                selected_ids,
                block_size,
            )
            selected_values = helpers.gather_selected_blocks(
                values,
                selected_ids,
                block_size,
            )
            selected_attention, _ = helpers.last_query_attention(
                query[:, :, -1:, :],
                selected_keys,
                selected_values,
            )
            projected = block.attn.c_proj(
                phase11._merge_attention_heads(selected_attention)
            )
            projected = block.attn.resid_dropout(projected)
            patched_last = residual_last + projected
            patched_last = patched_last + block.mlp(
                block.ln_2(patched_last)
            )
            hidden_states = normal_output.clone()
            hidden_states[:, -1:, :] = patched_last
            per_layer[layer_idx] = {
                "budget": budget,
                "selected_block_ids": selected_ids,
                "selected_block_count": len(selected_ids),
                "block_count": int(masses.shape[0]),
                "selected_block_ratio": len(selected_ids) / masses.shape[0],
                "attention_mass_captured": (
                    helpers.captured_attention_mass(masses, selected_ids)
                ),
                **helpers.calculate_metrics(
                    full_attention,
                    selected_attention,
                ),
                **selector_info,
            }

        hidden_states = model.transformer.ln_f(hidden_states)
        logits = model.lm_head(hidden_states[:, -1, :])
    return logits, per_layer


def generate_sequences(
    *,
    model: Any,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    layer_budget_map: dict[int, int],
    phase11: Any,
    helpers: Any,
) -> dict[str, Any]:
    baseline_context = input_ids.clone()
    patched_context = input_ids.clone()
    baseline_generated: list[int] = []
    patched_generated: list[int] = []
    step_kl: list[float] = []
    step_top1: list[float] = []
    layer_ratios: dict[int, list[float]] = defaultdict(list)
    layer_counts: dict[int, list[int]] = defaultdict(list)

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
        patched_logits, layer_metrics = (
            patched_multilayer_next_token_logits(
                model=model,
                input_ids=patched_input,
                layer_budget_map=layer_budget_map,
                block_size=args.block_size,
                selection_policy=args.selection_policy,
                sketch_dim=args.sketch_dim,
                block_score_reduction=args.block_score_reduction,
                seed=args.seed + step,
                phase11=phase11,
                helpers=helpers,
            )
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
        for layer, metrics in layer_metrics.items():
            layer_ratios[layer].append(metrics["selected_block_ratio"])
            layer_counts[layer].append(metrics["selected_block_count"])

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

    per_layer = {
        str(layer): {
            "budget": layer_budget_map[layer],
            "average_selected_block_ratio": (
                sum(layer_ratios[layer]) / len(layer_ratios[layer])
                if layer_ratios[layer]
                else 0.0
            ),
            "average_selected_blocks": (
                sum(layer_counts[layer]) / len(layer_counts[layer])
                if layer_counts[layer]
                else 0.0
            ),
        }
        for layer in layer_budget_map
    }
    return {
        "baseline_generated_token_ids": baseline_generated,
        "patched_generated_token_ids": patched_generated,
        "per_step_kl_divergence": step_kl,
        "per_step_top1_match": step_top1,
        "per_layer_selected_block_summary": per_layer,
    }


def evaluate_prompt(
    *,
    prompt: str,
    prompt_index: int,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    layer_budget_map: dict[int, int],
    device: torch.device,
    phase11: Any,
    generation_helpers: Any,
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
        layer_budget_map=layer_budget_map,
        phase11=phase11,
        helpers=helpers,
    )
    baseline_ids = generation["baseline_generated_token_ids"]
    patched_ids = generation["patched_generated_token_ids"]
    comparison = generation_helpers.compare_token_sequences(
        baseline_ids,
        patched_ids,
    )
    step_kl = generation["per_step_kl_divergence"]
    step_top1 = generation["per_step_top1_match"]
    exact = comparison["exact_token_sequence_match"]
    return {
        "prompt_index": prompt_index,
        "prompt_preview": prompt[:120],
        "prompt_token_length": int(input_ids.shape[1]),
        "num_generated_tokens": len(baseline_ids),
        "baseline_generated_token_ids": baseline_ids,
        "patched_generated_token_ids": patched_ids,
        "baseline_generated_text": tokenizer.decode(baseline_ids),
        "patched_generated_text": tokenizer.decode(patched_ids),
        **comparison,
        "average_per_step_kl_divergence": (
            sum(step_kl) / len(step_kl) if step_kl else 0.0
        ),
        "average_per_step_top1_match": (
            sum(step_top1) / len(step_top1) if step_top1 else 0.0
        ),
        "per_layer_selected_block_summary": generation[
            "per_layer_selected_block_summary"
        ],
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


def aggregate_rows(
    rows: list[dict[str, Any]],
    layer_budget_map: dict[int, int],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one prompt row is required")

    def average(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    per_layer = {}
    for layer, budget in layer_budget_map.items():
        summaries = [
            row["per_layer_selected_block_summary"][str(layer)]
            for row in rows
        ]
        per_layer[str(layer)] = {
            "budget": budget,
            "average_selected_block_ratio": sum(
                item["average_selected_block_ratio"] for item in summaries
            ) / len(summaries),
            "average_selected_blocks": sum(
                item["average_selected_blocks"] for item in summaries
            ) / len(summaries),
        }
    average_ratio = sum(
        values["average_selected_block_ratio"]
        for values in per_layer.values()
    ) / len(per_layer)
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
        "per_layer_selected_block_summary": per_layer,
        "average_selected_block_ratio_across_patched_layers": average_ratio,
    }


def build_report(
    *,
    config: dict[str, Any],
    layer_budget_map: dict[int, int],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "config": config,
        "layer_budget_map": {
            str(layer): budget for layer, budget in layer_budget_map.items()
        },
        "aggregate": aggregate_rows(rows, layer_budget_map),
        "per_prompt": rows,
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "multilayer_patch": True,
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
        "# Kivo-VD Phase 11.3 Multi-Layer Generation Evaluation",
        "",
        "**Status:** Multi-layer selected-attention greedy-generation probe "
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
    lines.extend(["", "## Layer-Budget Map", ""])
    _append_table(
        lines,
        ["layer", "budget"],
        [
            [layer, budget]
            for layer, budget in report["layer_budget_map"].items()
        ],
    )
    aggregate = report["aggregate"]
    scalar_aggregate = {
        key: value
        for key, value in aggregate.items()
        if key != "per_layer_selected_block_summary"
    }
    lines.extend(["", "## Aggregate Metrics", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [[key, value] for key, value in scalar_aggregate.items()],
    )
    lines.extend(["", "## Per-Layer Selected Blocks", ""])
    _append_table(
        lines,
        ["layer", "budget", "avg selected blocks", "avg selected ratio"],
        [
            [
                layer,
                values["budget"],
                values["average_selected_blocks"],
                values["average_selected_block_ratio"],
            ]
            for layer, values in aggregate[
                "per_layer_selected_block_summary"
            ].items()
        ],
    )
    lines.extend(["", "## Per-Prompt Metrics", ""])
    _append_table(
        lines,
        [
            "prompt",
            "tokens",
            "generated",
            "exact",
            "prefix",
            "token match",
            "edit distance",
            "avg KL",
            "step top-1",
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
        "If layers 5 and 8 remain stable at budget 8, add layer 11. If that "
        "remains stable, test the adaptive map `0:12,5:8,8:8,11:8`. A stable "
        "oracle with a divergent query-key run identifies selector quality as "
        "the bottleneck. Oracle divergence means the budget is too aggressive.",
        "",
        "## Caveats",
        "",
        "- Evaluation runs outside vLLM.",
        "- No vLLM integration is implemented or authorized.",
        "- Multiple layers are patched only in this standalone experiment.",
        "- Generation uses greedy decoding only.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency claim is made.",
        "- This is a generation-quality probe, not a preservation claim.",
        "",
        "## Next Steps",
        "",
        "Stable adaptive-map results would support a Phase 11.4 offline "
        "generation sweep and readiness gate. vLLM integration remains out "
        "of scope.",
    ])
    return "\n".join(lines) + "\n"


def validate_args(
    args: argparse.Namespace,
    layer_budget_map: dict[int, int],
) -> None:
    positive = {
        "--block-size": args.block_size,
        "--sketch-dim": args.sketch_dim,
        "--max-length": args.max_length,
        "--max-new-tokens": args.max_new_tokens,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not layer_budget_map:
        raise ValueError("at least one patched layer is required")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    layer_budget_map = resolve_layer_budget_map(
        layers_value=args.layers,
        budgets_value=args.budgets,
        map_value=args.layer_budget_map,
    )
    validate_args(args, layer_budget_map)
    phase11 = _load_phase11()
    generation_helpers = _load_phase11_generation()
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
            layer_budget_map=layer_budget_map,
            device=device,
            phase11=phase11,
            generation_helpers=generation_helpers,
            helpers=helpers,
        )
        for index, prompt in enumerate(prompts)
    ]
    return build_report(
        config={
            "model": args.model,
            "layers": list(layer_budget_map),
            "budgets": list(layer_budget_map.values()),
            "block_size": args.block_size,
            "selection_policy": args.selection_policy,
            "sketch_dim": args.sketch_dim,
            "block_score_reduction": args.block_score_reduction,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "teacher_forced_context": bool(args.teacher_forced_context),
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "patch_scope": "multilayer_last_token_each_decode_step",
        },
        layer_budget_map=layer_budget_map,
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
                    **{
                        key: value
                        for key, value in report["aggregate"].items()
                        if key != "per_layer_selected_block_summary"
                    },
                    "model": args.model,
                    "selection_policy": args.selection_policy,
                    "layer_budget_map": report["layer_budget_map"],
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
