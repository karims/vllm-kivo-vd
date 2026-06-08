#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sweep ratio-scaled long-context adaptive budgets outside vLLM."""

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RATIO_POLICIES = (
    "balanced=0:0.60,5:0.45,8:0.45,11:0.60;"
    "safer=0:0.70,5:0.55,8:0.55,11:0.70;"
    "aggressive=0:0.50,5:0.40,8:0.40,11:0.50"
)
DEFAULT_TARGET_LENGTHS = "768,960"
DEFAULT_MODEL_POLICIES = "query_key_block_score,oracle_topk"
DEFAULT_MAX_NEW_TOKENS = "16,32"
METRIC_FIELDS = (
    "exact_sequence_match_rate",
    "average_token_match_rate",
    "average_prefix_match_length",
    "average_normalized_edit_distance",
    "average_per_step_kl_divergence",
    "average_per_step_top1_match_rate",
    "average_selected_block_ratio_across_patched_layers",
    "estimated_active_block_reduction_ratio",
)


def _load_script(filename: str, module_name: str) -> Any:
    module_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load helper script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_adaptive_module() -> Any:
    return _load_script(
        "run_adaptive_multilayer_generation_sweep.py",
        "run_adaptive_multilayer_generation_sweep",
    )


def _load_long_context_module() -> Any:
    return _load_script(
        "run_long_context_adaptive_generation_sweep.py",
        "run_long_context_adaptive_generation_sweep",
    )


def _load_multilayer_module() -> Any:
    return _load_script(
        "run_multilayer_selected_attention_generation_eval.py",
        "run_multilayer_selected_attention_generation_eval",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ratio/context-scaled layer-budget maps and evaluate "
            "long-context selected-attention generation outside vLLM."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--target-token-lengths", default=DEFAULT_TARGET_LENGTHS)
    parser.add_argument("--num-prompts-per-length", type=int, default=2)
    parser.add_argument("--ratio-policies", default=DEFAULT_RATIO_POLICIES)
    parser.add_argument("--min-budget", type=int, default=8)
    parser.add_argument("--max-budget", type=int)
    parser.add_argument(
        "--budget-rounding",
        choices=["floor", "ceil", "round"],
        default="ceil",
    )
    parser.add_argument("--policies", default=DEFAULT_MODEL_POLICIES)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--max-new-tokens-values",
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument(
        "--block-score-reduction",
        choices=["max", "mean", "logsumexp"],
        default="max",
    )
    parser.add_argument("--max-length", type=int, default=1024)
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
        "--output-dir",
        default="outputs/kivo_vd/phase11_6_ratio_scaled_long_context_sweep",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def parse_ratio_policies(value: str) -> dict[str, dict[int, float]]:
    policies: dict[str, dict[int, float]] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("ratio policies must use name=layer:ratio syntax")
        name, spec = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("ratio policy name must not be empty")
        ratios: dict[int, float] = {}
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError("ratio entries must use layer:ratio syntax")
            layer_text, ratio_text = entry.split(":", 1)
            layer = int(layer_text)
            ratio = float(ratio_text)
            if layer < 0:
                raise ValueError("ratio policy layers must be non-negative")
            if ratio <= 0:
                raise ValueError("ratio policy ratios must be positive")
            if layer in ratios:
                raise ValueError(f"duplicate layer in ratio policy: {layer}")
            ratios[layer] = ratio
        if not ratios:
            raise ValueError(f"ratio policy is empty: {name}")
        if name in policies:
            raise ValueError(f"duplicate ratio policy: {name}")
        policies[name] = dict(sorted(ratios.items()))
    if not policies:
        raise ValueError("--ratio-policies must not be empty")
    return policies


def format_ratio_policy(value: dict[int, float]) -> str:
    return ",".join(
        f"{layer}:{ratio:.6g}" for layer, ratio in sorted(value.items())
    )


def _round_budget(value: float, mode: str) -> int:
    if mode == "floor":
        return math.floor(value)
    if mode == "round":
        return round(value)
    return math.ceil(value)


def derive_layer_budget_map(
    *,
    ratios: dict[int, float],
    num_blocks: int,
    min_budget: int,
    max_budget: int | None,
    rounding: str,
) -> dict[int, int]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if min_budget <= 0:
        raise ValueError("min_budget must be positive")
    result = {}
    for layer, ratio in sorted(ratios.items()):
        budget = _round_budget(num_blocks * ratio, rounding)
        budget = max(min_budget, budget)
        if max_budget is not None:
            budget = min(max_budget, budget)
        result[layer] = min(num_blocks, budget)
    return result


def estimate_context_blocks(
    *,
    average_prompt_tokens: float,
    block_size: int,
) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(1, math.ceil(average_prompt_tokens / block_size))


def failure_flags(row: dict[str, Any]) -> list[str]:
    flags = []
    if float(row["exact_sequence_match_rate"]) < 1.0:
        flags.append("exact_sequence_match_below_1")
    if float(row["average_token_match_rate"]) < 0.99:
        flags.append("token_match_below_0.99")
    if float(row["average_normalized_edit_distance"]) > 0.0:
        flags.append("normalized_edit_distance_above_0")
    if float(row["average_per_step_kl_divergence"]) > 0.01:
        flags.append("average_kl_above_0.01")
    if float(row["average_per_step_top1_match_rate"]) < 1.0:
        flags.append("per_step_top1_below_1")
    if (
        float(row["average_selected_block_ratio_across_patched_layers"])
        > 0.85
    ):
        flags.append("selected_ratio_above_0.85")
    if float(row["estimated_active_block_reduction_ratio"]) < 0.20:
        flags.append("estimated_reduction_below_0.20")
    actual = float(row["average_actual_prompt_tokens"])
    target = float(row["target_token_length"])
    if actual < target * 0.95:
        flags.append("actual_prompt_length_too_short")
    return flags


def _is_passing(row: dict[str, Any]) -> bool:
    if row.get("status") != "succeeded":
        return False
    required = (
        "policy",
        "exact_sequence_match_rate",
        "average_token_match_rate",
        "average_normalized_edit_distance",
        "average_per_step_kl_divergence",
    )
    if any(field not in row for field in required):
        return False
    return all([
        row["policy"] == "query_key_block_score",
        row["exact_sequence_match_rate"] >= 1.0,
        row["average_token_match_rate"] >= 1.0,
        row["average_normalized_edit_distance"] <= 0.0,
        row["average_per_step_kl_divergence"] <= 0.01,
    ])


def best_deployable_tradeoff(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [row for row in rows if _is_passing(row)]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            -float(row["estimated_active_block_reduction_ratio"]),
            float(row["average_per_step_kl_divergence"]),
        ),
    )


def safest_passing_config(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [row for row in rows if _is_passing(row)]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: float(row["average_per_step_kl_divergence"]),
    )


def _average_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "succeeded":
            groups[tuple(row[field] for field in group_fields)].append(row)
    summaries = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = dict(zip(group_fields, key))
        summary["count"] = len(values)
        for field in METRIC_FIELDS:
            summary[field] = sum(float(row[field]) for row in values) / len(
                values
            )
        summary["average_actual_prompt_tokens"] = sum(
            float(row["average_actual_prompt_tokens"]) for row in values
        ) / len(values)
        summary["estimated_context_blocks"] = sum(
            int(row["estimated_context_blocks"]) for row in values
        ) / len(values)
        summaries.append(summary)
    return summaries


def calculate_oracle_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "succeeded"]
    keys = ("ratio_policy_name", "target_token_length", "max_new_tokens")
    indexed = {
        (row["policy"], *(row[key] for key in keys)): row
        for row in successful
    }
    combinations = sorted({tuple(row[key] for key in keys) for row in successful})
    policies = sorted({
        row["policy"] for row in successful if row["policy"] != "oracle_topk"
    })
    gaps = []
    for combo in combinations:
        oracle = indexed.get(("oracle_topk", *combo))
        if oracle is None:
            continue
        for policy in policies:
            candidate = indexed.get((policy, *combo))
            if candidate is None:
                continue
            gaps.append({
                "policy": policy,
                **dict(zip(keys, combo)),
                "query_minus_oracle_kl": (
                    candidate["average_per_step_kl_divergence"]
                    - oracle["average_per_step_kl_divergence"]
                ),
                "oracle_minus_query_exact_match": (
                    oracle["exact_sequence_match_rate"]
                    - candidate["exact_sequence_match_rate"]
                ),
                "oracle_minus_query_token_match": (
                    oracle["average_token_match_rate"]
                    - candidate["average_token_match_rate"]
                ),
                "query_minus_oracle_edit_distance": (
                    candidate["average_normalized_edit_distance"]
                    - oracle["average_normalized_edit_distance"]
                ),
            })
    return gaps


def build_summary(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    derived_maps: list[dict[str, Any]],
) -> dict[str, Any]:
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    failed = [row for row in rows if row.get("status") == "failed"]
    planned = [row for row in rows if row.get("status") == "planned"]
    best = best_deployable_tradeoff(rows)
    return {
        "status": "planned" if planned and not succeeded and not failed else (
            "complete" if not failed else "partial"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "counts": {
            "total": len(rows),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "planned": len(planned),
        },
        "derived_maps": derived_maps,
        "per_ratio_policy": _average_rows(rows, ("ratio_policy_name",)),
        "per_model_policy": _average_rows(rows, ("policy",)),
        "per_target_token_length": _average_rows(
            rows, ("target_token_length",)
        ),
        "per_max_new_tokens": _average_rows(rows, ("max_new_tokens",)),
        "policy_ratio_length_token": _average_rows(
            rows,
            (
                "policy",
                "ratio_policy_name",
                "target_token_length",
                "max_new_tokens",
            ),
        ),
        "selected_ratio_reduction": _average_rows(
            rows,
            ("policy", "ratio_policy_name", "target_token_length"),
        ),
        "oracle_gaps": calculate_oracle_gaps(rows),
        "best_deployable_tradeoff": best,
        "safest_passing_deployable_config": safest_passing_config(rows),
        "readiness": {
            "phase11_7_ready": best is not None,
            "phase12_ready": False,
            "recommended_next_step": (
                "Use the best passing ratio policy as input to broader "
                "prompt/model coverage; vLLM remains out of scope."
                if best is not None
                else "Refine ratio policies before broader evaluation."
            ),
        },
        "worst_cases": {
            "exact_sequence_match_ascending": sorted(
                succeeded,
                key=lambda row: row["exact_sequence_match_rate"],
            )[:5],
            "token_match_ascending": sorted(
                succeeded,
                key=lambda row: row["average_token_match_rate"],
            )[:5],
            "edit_distance_descending": sorted(
                succeeded,
                key=lambda row: row["average_normalized_edit_distance"],
                reverse=True,
            )[:5],
            "kl_descending": sorted(
                succeeded,
                key=lambda row: row["average_per_step_kl_divergence"],
                reverse=True,
            )[:5],
        },
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "greedy_generation_only": True,
            "synthetic_long_prompts": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "latency_claim": False,
            "generation_quality_probe_only": True,
            "gpt2_context_limit_applies": True,
        },
    }


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _table(
    lines: list[str],
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")


def _metric_table(
    lines: list[str],
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    _table(
        lines,
        [
            *fields,
            "count",
            "actual tokens",
            "blocks",
            "exact",
            "token match",
            "edit",
            "KL",
            "selected ratio",
            "estimated reduction",
        ],
        [
            [
                *(row[field] for field in fields),
                row["count"],
                row["average_actual_prompt_tokens"],
                row["estimated_context_blocks"],
                row["exact_sequence_match_rate"],
                row["average_token_match_rate"],
                row["average_normalized_edit_distance"],
                row["average_per_step_kl_divergence"],
                row[
                    "average_selected_block_ratio_across_patched_layers"
                ],
                row["estimated_active_block_reduction_ratio"],
            ]
            for row in rows
        ],
    )


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 11.6 Ratio-Scaled Long-Context Sweep",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        "This is a standalone HuggingFace/PyTorch generation-quality probe.",
        "",
        "## Configuration",
        "",
    ]
    _table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in summary["config"].items()],
    )
    lines.extend(["", "## Derived Maps", ""])
    _table(
        lines,
        [
            "ratio policy",
            "target",
            "actual tokens",
            "blocks",
            "ratios",
            "derived map",
        ],
        [
            [
                row["ratio_policy_name"],
                row["target_token_length"],
                row["average_actual_prompt_tokens"],
                row["estimated_context_blocks"],
                row["ratio_policy_spec"],
                row["derived_layer_budget_map"],
            ]
            for row in summary["derived_maps"]
        ],
    )
    lines.extend(["", "## High-Level Summary", ""])
    _table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in summary["counts"].items()],
    )
    sections = [
        ("Per-Ratio-Policy Results", "per_ratio_policy", ("ratio_policy_name",)),
        ("Per-Model-Policy Results", "per_model_policy", ("policy",)),
        (
            "Per-Target-Length Results",
            "per_target_token_length",
            ("target_token_length",),
        ),
        (
            "Per-Max-New-Tokens Results",
            "per_max_new_tokens",
            ("max_new_tokens",),
        ),
        (
            "Selected Ratio And Estimated Reduction",
            "selected_ratio_reduction",
            ("policy", "ratio_policy_name", "target_token_length"),
        ),
        (
            "Policy / Ratio / Length / Token Results",
            "policy_ratio_length_token",
            (
                "policy",
                "ratio_policy_name",
                "target_token_length",
                "max_new_tokens",
            ),
        ),
    ]
    for title, key, fields in sections:
        lines.extend(["", f"## {title}", ""])
        _metric_table(lines, summary[key], fields)
    lines.extend(["", "## Oracle Gaps", ""])
    _table(
        lines,
        [
            "policy",
            "ratio",
            "target",
            "tokens",
            "KL gap",
            "exact gap",
            "token gap",
            "edit gap",
        ],
        [
            [
                row["policy"],
                row["ratio_policy_name"],
                row["target_token_length"],
                row["max_new_tokens"],
                row["query_minus_oracle_kl"],
                row["oracle_minus_query_exact_match"],
                row["oracle_minus_query_token_match"],
                row["query_minus_oracle_edit_distance"],
            ]
            for row in summary["oracle_gaps"]
        ],
    )
    lines.extend(["", "## Worst Cases", ""])
    for name, rows in summary["worst_cases"].items():
        lines.extend([f"### {name.replace('_', ' ').title()}", ""])
        _table(
            lines,
            [
                "policy",
                "ratio",
                "target",
                "tokens",
                "exact",
                "token match",
                "edit",
                "KL",
            ],
            [
                [
                    row["policy"],
                    row["ratio_policy_name"],
                    row["target_token_length"],
                    row["max_new_tokens"],
                    row["exact_sequence_match_rate"],
                    row["average_token_match_rate"],
                    row["average_normalized_edit_distance"],
                    row["average_per_step_kl_divergence"],
                ]
                for row in rows
            ],
        )
        lines.append("")
    for title, key in (
        ("Best Deployable Tradeoff", "best_deployable_tradeoff"),
        ("Safest Passing Deployable Config",
         "safest_passing_deployable_config"),
    ):
        lines.extend(["", f"## {title}", ""])
        value = summary[key]
        if value is None:
            lines.append("No passing non-oracle configuration was found.")
        else:
            _table(lines, ["field", "value"], [[k, v] for k, v in value.items()])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Ratio policies derive layer budgets from estimated context blocks.",
        "- `phase11_7_ready` only means at least one query-key ratio policy "
        "passed this standalone probe.",
        "- `phase12_ready` remains `false` by design.",
        "",
        "## Caveats",
        "",
        "- This experiment runs outside vLLM.",
        "- No vLLM integration or active routing is implemented.",
        "- Generation uses greedy decoding only.",
        "- Prompts are synthetic long prompts.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency claim is made.",
        "- This is a generation-quality probe, not a preservation claim.",
        "- GPT-2's context limit applies.",
        "",
        "## Recommended Next Step",
        "",
        summary["readiness"]["recommended_next_step"],
    ])
    return "\n".join(lines) + "\n"


def _derive_maps(
    *,
    ratio_policies: dict[str, dict[int, float]],
    target_lengths: list[int],
    prompt_groups: dict[int, list[dict[str, Any]]] | None,
    block_size: int,
    min_budget: int,
    max_budget: int | None,
    rounding: str,
) -> list[dict[str, Any]]:
    derived = []
    for target in target_lengths:
        if prompt_groups and target in prompt_groups:
            lengths = [
                row["actual_prompt_token_length"]
                for row in prompt_groups[target]
            ]
            average_tokens = sum(lengths) / len(lengths)
        else:
            average_tokens = float(target)
        num_blocks = estimate_context_blocks(
            average_prompt_tokens=average_tokens,
            block_size=block_size,
        )
        for name, ratios in ratio_policies.items():
            layer_map = derive_layer_budget_map(
                ratios=ratios,
                num_blocks=num_blocks,
                min_budget=min_budget,
                max_budget=max_budget,
                rounding=rounding,
            )
            adaptive = _load_adaptive_module()
            derived.append({
                "ratio_policy_name": name,
                "ratio_policy_spec": format_ratio_policy(ratios),
                "target_token_length": target,
                "average_actual_prompt_tokens": average_tokens,
                "estimated_context_blocks": num_blocks,
                "derived_layer_budget_map": (
                    adaptive.format_layer_budget_map(layer_map)
                ),
                "layer_budget_map": layer_map,
            })
    return derived


def _planned_rows(
    *,
    derived_maps: list[dict[str, Any]],
    model_policies: list[str],
    max_new_tokens_values: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for derived in derived_maps:
        for policy in model_policies:
            for max_new_tokens in max_new_tokens_values:
                rows.append({
                    "status": "planned",
                    "ratio_policy_name": derived["ratio_policy_name"],
                    "ratio_policy_spec": derived["ratio_policy_spec"],
                    "derived_layer_budget_map": (
                        derived["derived_layer_budget_map"]
                    ),
                    "target_token_length": derived["target_token_length"],
                    "average_actual_prompt_tokens": (
                        derived["average_actual_prompt_tokens"]
                    ),
                    "estimated_context_blocks": (
                        derived["estimated_context_blocks"]
                    ),
                    "policy": policy,
                    "max_new_tokens": max_new_tokens,
                    "failure_flags": [],
                    "warnings": ["prompt tokenization deferred until execution"],
                })
    return rows


def _run_rows(
    *,
    args: argparse.Namespace,
    derived_maps: list[dict[str, Any]],
    prompt_groups: dict[int, list[dict[str, Any]]],
    model_policies: list[str],
    max_new_tokens_values: list[int],
    tokenizer: Any,
    model: Any,
    device: Any,
) -> list[dict[str, Any]]:
    multilayer = _load_multilayer_module()
    phase11 = multilayer._load_phase11()
    generation_helpers = multilayer._load_phase11_generation()
    helpers = phase11._load_selected_attention_helpers()
    rows = []
    adaptive = _load_adaptive_module()
    for derived in derived_maps:
        prompt_records = prompt_groups[derived["target_token_length"]]
        for policy in model_policies:
            for max_new_tokens in max_new_tokens_values:
                evaluation_args = argparse.Namespace(
                    block_size=args.block_size,
                    selection_policy=policy,
                    sketch_dim=args.sketch_dim,
                    block_score_reduction=args.block_score_reduction,
                    max_length=args.max_length,
                    max_new_tokens=max_new_tokens,
                    teacher_forced_context=args.teacher_forced_context,
                    seed=args.seed,
                )
                try:
                    prompt_rows = [
                        multilayer.evaluate_prompt(
                            prompt=record["prompt"],
                            prompt_index=index,
                            tokenizer=tokenizer,
                            model=model,
                            args=evaluation_args,
                            layer_budget_map=derived["layer_budget_map"],
                            device=device,
                            phase11=phase11,
                            generation_helpers=generation_helpers,
                            helpers=helpers,
                        )
                        for index, record in enumerate(prompt_records)
                    ]
                    aggregate = multilayer.aggregate_rows(
                        prompt_rows,
                        derived["layer_budget_map"],
                    )
                    selected_ratio = aggregate[
                        "average_selected_block_ratio_across_patched_layers"
                    ]
                    row = {
                        "status": "succeeded",
                        "ratio_policy_name": derived["ratio_policy_name"],
                        "ratio_policy_spec": derived["ratio_policy_spec"],
                        "derived_layer_budget_map": (
                            derived["derived_layer_budget_map"]
                        ),
                        "target_token_length": (
                            derived["target_token_length"]
                        ),
                        "average_actual_prompt_tokens": (
                            sum(
                                item["prompt_token_length"]
                                for item in prompt_rows
                            ) / len(prompt_rows)
                        ),
                        "estimated_context_blocks": (
                            derived["estimated_context_blocks"]
                        ),
                        "policy": policy,
                        "max_new_tokens": max_new_tokens,
                        **{
                            field: aggregate[field]
                            for field in ("num_prompts", *METRIC_FIELDS[:-1])
                        },
                        "estimated_active_block_reduction_ratio": (
                            1.0 - selected_ratio
                        ),
                        "warnings": [],
                    }
                    row["failure_flags"] = failure_flags(row)
                    rows.append(row)
                except Exception as exc:
                    rows.append({
                        "status": "failed",
                        "ratio_policy_name": derived["ratio_policy_name"],
                        "ratio_policy_spec": derived["ratio_policy_spec"],
                        "derived_layer_budget_map": (
                            adaptive.format_layer_budget_map(
                                derived["layer_budget_map"]
                            )
                        ),
                        "target_token_length": (
                            derived["target_token_length"]
                        ),
                        "average_actual_prompt_tokens": (
                            derived["average_actual_prompt_tokens"]
                        ),
                        "estimated_context_blocks": (
                            derived["estimated_context_blocks"]
                        ),
                        "policy": policy,
                        "max_new_tokens": max_new_tokens,
                        "failure_flags": ["evaluation_failed"],
                        "warnings": [str(exc)],
                    })
                    if not args.continue_on_error:
                        return rows
    return rows


def validate_args(
    args: argparse.Namespace,
    *,
    max_new_tokens_values: list[int],
) -> None:
    for name in (
        "num_prompts_per_length",
        "min_budget",
        "block_size",
        "sketch_dim",
        "max_length",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_budget is not None and args.max_budget <= 0:
        raise ValueError("--max-budget must be positive")
    if args.max_budget is not None and args.max_budget < args.min_budget:
        raise ValueError("--max-budget must be at least --min-budget")
    if max(max_new_tokens_values) >= args.max_length:
        raise ValueError("max new tokens must be smaller than --max-length")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    adaptive = _load_adaptive_module()
    long_context = _load_long_context_module()
    target_lengths = long_context.parse_target_token_lengths(
        args.target_token_lengths
    )
    max_new_tokens_values = adaptive.parse_int_csv(
        args.max_new_tokens_values,
        label="--max-new-tokens-values",
    )
    validate_args(args, max_new_tokens_values=max_new_tokens_values)
    ratio_policies = parse_ratio_policies(args.ratio_policies)
    model_policies = adaptive.parse_policies(args.policies)
    tokenizer = None
    model = None
    device = None
    if args.dry_run:
        prompt_groups = None
    else:
        import torch

        multilayer = _load_multilayer_module()
        phase11 = multilayer._load_phase11()
        helpers = phase11._load_selected_attention_helpers()
        torch.manual_seed(args.seed)
        device = helpers.resolve_device(args.device)
        dtype = helpers.resolve_dtype(args.dtype)
        tokenizer, model = phase11.load_hf_model(args.model, device, dtype)
        prompt_groups = {
            target: long_context.generate_synthetic_prompts(
                tokenizer=tokenizer,
                target_token_length=target,
                num_prompts=args.num_prompts_per_length,
                max_prompt_tokens=(
                    args.max_length - max(max_new_tokens_values)
                ),
                seed=args.seed,
            )
            for target in target_lengths
        }
    derived_maps = _derive_maps(
        ratio_policies=ratio_policies,
        target_lengths=target_lengths,
        prompt_groups=prompt_groups,
        block_size=args.block_size,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        rounding=args.budget_rounding,
    )
    rows = (
        _planned_rows(
            derived_maps=derived_maps,
            model_policies=model_policies,
            max_new_tokens_values=max_new_tokens_values,
        )
        if args.dry_run
        else _run_rows(
            args=args,
            derived_maps=derived_maps,
            prompt_groups=prompt_groups or {},
            model_policies=model_policies,
            max_new_tokens_values=max_new_tokens_values,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
    )
    config = {
        "model": args.model,
        "target_token_lengths": target_lengths,
        "num_prompts_per_length": args.num_prompts_per_length,
        "ratio_policies": {
            name: format_ratio_policy(ratios)
            for name, ratios in ratio_policies.items()
        },
        "min_budget": args.min_budget,
        "max_budget": args.max_budget,
        "budget_rounding": args.budget_rounding,
        "policies": model_policies,
        "block_size": args.block_size,
        "max_new_tokens_values": max_new_tokens_values,
        "sketch_dim": args.sketch_dim,
        "block_score_reduction": args.block_score_reduction,
        "max_length": args.max_length,
        "teacher_forced_context": args.teacher_forced_context,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }
    summary = build_summary(rows, config=config, derived_maps=[
        {key: value for key, value in item.items() if key != "layer_budget_map"}
        for item in derived_maps
    ])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "ratio_scaled_generation_runs.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_json = output_dir / "ratio_scaled_generation_summary.json"
    summary_md = output_dir / "ratio_scaled_generation_summary.md"
    prompts_json = output_dir / "ratio_scaled_prompts.json"
    _write_json(summary_json, summary)
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    if prompt_groups:
        _write_json(
            prompts_json,
            {str(target): records for target, records in prompt_groups.items()},
        )
    return {
        "summary": summary,
        "rows_path": str(rows_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "prompts_json": str(prompts_json) if prompt_groups else None,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        result = run_sweep(_parse_args(argv))
        print(json.dumps({
            "status": result["summary"]["status"],
            "counts": result["summary"]["counts"],
            "phase11_7_ready": result["summary"]["readiness"][
                "phase11_7_ready"
            ],
            "phase12_ready": False,
            "rows_path": result["rows_path"],
            "summary_json": result["summary_json"],
            "summary_md": result["summary_md"],
            "prompts_json": result["prompts_json"],
        }, separators=(",", ":")))
        return 0 if result["summary"]["counts"]["failed"] == 0 else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
