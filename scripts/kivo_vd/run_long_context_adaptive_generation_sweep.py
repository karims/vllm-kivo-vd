#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sweep long-context adaptive multi-layer generation outside vLLM."""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LAYER_BUDGET_MAPS = "0:12,5:8,8:8,11:12"
DEFAULT_POLICIES = "query_key_block_score,oracle_topk"
DEFAULT_TARGET_LENGTHS = "768,896"
DEFAULT_MAX_NEW_TOKENS = "32"
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


def _load_multilayer_module() -> Any:
    return _load_script(
        "run_multilayer_selected_attention_generation_eval.py",
        "run_multilayer_selected_attention_generation_eval",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate adaptive selected attention on controlled long-context "
            "prompts outside vLLM."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--prompt-source",
        choices=["synthetic", "long_builtin", "file"],
        default="synthetic",
    )
    parser.add_argument("--prompts-file")
    parser.add_argument(
        "--target-token-lengths",
        default=DEFAULT_TARGET_LENGTHS,
    )
    parser.add_argument("--num-prompts-per-length", type=int, default=3)
    parser.add_argument(
        "--layer-budget-maps",
        default=DEFAULT_LAYER_BUDGET_MAPS,
    )
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
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
        default=(
            "outputs/kivo_vd/"
            "phase11_5_long_context_adaptive_generation_sweep"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def parse_target_token_lengths(value: str) -> list[int]:
    adaptive = _load_adaptive_module()
    return adaptive.parse_int_csv(
        value,
        label="--target-token-lengths",
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        ids = encoded["input_ids"]
    else:
        ids = encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return str(tokenizer.decode(token_ids))


def synthetic_prompt_templates() -> list[dict[str, str]]:
    return [
        {
            "name": "retrieval_key",
            "prefix": (
                "The secret retrieval key is BLUE ORCHID. Preserve this "
                "fact while reading the following archive. "
            ),
            "filler": (
                "The archive describes candidate search, exact reranking, "
                "bounded metadata, cache blocks, and reproducible tests. "
            ),
            "suffix": "What is the secret retrieval key?",
        },
        {
            "name": "system_debugging",
            "prefix": (
                "The first diagnostic action is CHECK CUDA AVAILABILITY. "
                "A long incident report follows. "
            ),
            "filler": (
                "Engineers inspect allocator state, request queues, model "
                "configuration, scheduler traces, and deterministic logs. "
            ),
            "suffix": "What is the first diagnostic action?",
        },
        {
            "name": "code_documentation",
            "prefix": (
                "The API contract requires returning sentinel value 731 "
                "when validation succeeds. "
            ),
            "filler": (
                "The documentation discusses typed inputs, stable outputs, "
                "error handling, compatibility, examples, and unit tests. "
            ),
            "suffix": "Which sentinel value does the API contract require?",
        },
        {
            "name": "long_explanation",
            "prefix": (
                "The central conclusion is that exact reranking follows "
                "candidate retrieval. "
            ),
            "filler": (
                "The explanation compares local context, distant evidence, "
                "approximate ranking, conservative fallbacks, and evaluation. "
            ),
            "suffix": "State the central conclusion.",
        },
        {
            "name": "structured_facts",
            "prefix": (
                "Fact zero names AMBER COMPASS as the recovery phrase. "
            ),
            "filler": (
                "Each later fact records a harmless project identifier, "
                "a review state, a timestamp category, and an owner role. "
            ),
            "suffix": "What recovery phrase was named in fact zero?",
        },
    ]


def _fit_template_to_target(
    *,
    tokenizer: Any,
    template: dict[str, str],
    target_token_length: int,
    max_prompt_tokens: int,
) -> tuple[str, int]:
    target = min(target_token_length, max_prompt_tokens)
    if target <= 0:
        raise ValueError("target prompt length must be positive")
    prefix_ids = _token_ids(tokenizer, template["prefix"])
    filler_ids = _token_ids(tokenizer, template["filler"])
    suffix_ids = _token_ids(tokenizer, template["suffix"])
    if not filler_ids:
        raise ValueError("prompt filler must tokenize to at least one token")
    fixed_count = len(prefix_ids) + len(suffix_ids)
    if fixed_count >= target:
        combined = (prefix_ids + suffix_ids)[:target]
    else:
        filler_budget = target - fixed_count
        repeats = (filler_budget + len(filler_ids) - 1) // len(filler_ids)
        combined = (
            prefix_ids
            + (filler_ids * repeats)[:filler_budget]
            + suffix_ids
        )
    prompt = _decode(tokenizer, combined)
    actual_ids = _token_ids(tokenizer, prompt)[:target]
    if len(actual_ids) != len(combined):
        prompt = _decode(tokenizer, actual_ids)
    return prompt, len(actual_ids)


def generate_synthetic_prompts(
    *,
    tokenizer: Any,
    target_token_length: int,
    num_prompts: int,
    max_prompt_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    templates = synthetic_prompt_templates()
    prompts = []
    for prompt_index in range(num_prompts):
        template = templates[(seed + prompt_index) % len(templates)]
        prompt, actual_length = _fit_template_to_target(
            tokenizer=tokenizer,
            template=template,
            target_token_length=target_token_length,
            max_prompt_tokens=max_prompt_tokens,
        )
        prompts.append({
            "prompt_index": prompt_index,
            "prompt_type": template["name"],
            "target_token_length": target_token_length,
            "actual_prompt_token_length": actual_length,
            "prompt": prompt,
        })
    return prompts


def _read_prompt_file(path: str | Path) -> list[str]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"prompts file is missing: {input_path}")
    prompts = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"prompts file is empty: {input_path}")
    return prompts


def generate_file_prompts(
    *,
    tokenizer: Any,
    source_prompts: list[str],
    target_token_length: int,
    num_prompts: int,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    prompts = []
    for prompt_index in range(num_prompts):
        source = source_prompts[prompt_index % len(source_prompts)]
        template = {
            "name": "file",
            "prefix": source + " ",
            "filler": source + " ",
            "suffix": "Summarize the most important earlier detail.",
        }
        prompt, actual_length = _fit_template_to_target(
            tokenizer=tokenizer,
            template=template,
            target_token_length=target_token_length,
            max_prompt_tokens=max_prompt_tokens,
        )
        prompts.append({
            "prompt_index": prompt_index,
            "prompt_type": "file",
            "target_token_length": target_token_length,
            "actual_prompt_token_length": actual_length,
            "prompt": prompt,
        })
    return prompts


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
    target = int(row["target_token_length"])
    lengths = row.get("actual_prompt_token_lengths", [])
    if not lengths or min(int(length) for length in lengths) < target * 0.95:
        flags.append("actual_prompt_length_too_short")
    return flags


def _average_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "succeeded":
            groups[tuple(row[field] for field in group_fields)].append(row)
    results = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        result = dict(zip(group_fields, key))
        result["count"] = len(values)
        for field in METRIC_FIELDS:
            result[field] = sum(float(row[field]) for row in values) / len(
                values
            )
        all_lengths = [
            int(length)
            for row in values
            for length in row["actual_prompt_token_lengths"]
        ]
        result["average_actual_prompt_token_length"] = (
            sum(all_lengths) / len(all_lengths) if all_lengths else 0.0
        )
        result["minimum_actual_prompt_token_length"] = (
            min(all_lengths) if all_lengths else None
        )
        result["maximum_actual_prompt_token_length"] = (
            max(all_lengths) if all_lengths else None
        )
        results.append(result)
    return results


def calculate_oracle_gaps(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "succeeded"]
    key_fields = (
        "target_token_length",
        "layer_budget_map",
        "max_new_tokens",
    )
    indexed = {
        (row["policy"], *(row[field] for field in key_fields)): row
        for row in successful
    }
    combinations = sorted({
        tuple(row[field] for field in key_fields) for row in successful
    })
    policies = sorted({
        row["policy"]
        for row in successful
        if row["policy"] != "oracle_topk"
    })
    gaps = []
    for combination in combinations:
        oracle = indexed.get(("oracle_topk", *combination))
        if oracle is None:
            continue
        for policy in policies:
            candidate = indexed.get((policy, *combination))
            if candidate is None:
                continue
            gaps.append({
                "policy": policy,
                **dict(zip(key_fields, combination)),
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


def best_deployable_config(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("status") == "succeeded"
        and row.get("policy") != "oracle_topk"
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            -float(row["exact_sequence_match_rate"]),
            -float(row["average_token_match_rate"]),
            float(row["average_normalized_edit_distance"]),
            float(row["average_per_step_kl_divergence"]),
            -float(row["estimated_active_block_reduction_ratio"]),
            str(row["policy"]),
        ),
    )


def build_summary(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    failed = [row for row in rows if row.get("status") == "failed"]
    planned = [row for row in rows if row.get("status") == "planned"]
    query_rows = [
        row
        for row in succeeded
        if row["policy"] == "query_key_block_score"
    ]
    ready = bool(query_rows) and all(
        not row["failure_flags"] for row in query_rows
    )
    if planned and not succeeded and not failed:
        next_step = "Execute the planned long-context sweep."
    elif ready:
        next_step = (
            "Expand prompt coverage or test a larger model before vLLM work."
        )
    else:
        next_step = (
            "Refine the adaptive map or selector before broader evaluation."
        )
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
        "per_policy": _average_rows(rows, ("policy",)),
        "per_target_token_length": _average_rows(
            rows,
            ("target_token_length",),
        ),
        "per_map": _average_rows(rows, ("layer_budget_map",)),
        "per_max_new_tokens": _average_rows(rows, ("max_new_tokens",)),
        "policy_length_map_tokens": _average_rows(
            rows,
            (
                "policy",
                "target_token_length",
                "layer_budget_map",
                "max_new_tokens",
            ),
        ),
        "selected_ratio_reduction": _average_rows(
            rows,
            ("policy", "target_token_length"),
        ),
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
        "oracle_gaps": calculate_oracle_gaps(rows),
        "best_deployable_config": best_deployable_config(rows),
        "readiness": {
            "phase11_6_ready": ready,
            "phase12_ready": False,
            "recommended_next_step": next_step,
        },
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "greedy_generation_only": True,
            "synthetic_long_prompts_unless_file_provided": True,
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
            "exact",
            "token match",
            "edit",
            "KL",
            "top-1",
            "selected ratio",
            "estimated reduction",
        ],
        [
            [
                *(row[field] for field in fields),
                row["count"],
                row["average_actual_prompt_token_length"],
                row["exact_sequence_match_rate"],
                row["average_token_match_rate"],
                row["average_normalized_edit_distance"],
                row["average_per_step_kl_divergence"],
                row["average_per_step_top1_match_rate"],
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
        "# Kivo-VD Phase 11.5 Long-Context Adaptive Generation Sweep",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        "This is a long-context greedy-generation quality probe outside vLLM.",
        "",
        "## Configuration",
        "",
    ]
    _table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in summary["config"].items()],
    )
    lines.extend(["", "## High-Level Summary", ""])
    _table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in summary["counts"].items()],
    )
    sections = [
        (
            "Prompt Lengths",
            "per_target_token_length",
            ("target_token_length",),
        ),
        ("Per-Policy Results", "per_policy", ("policy",)),
        (
            "Per-Target-Length Results",
            "per_target_token_length",
            ("target_token_length",),
        ),
        ("Per-Map Results", "per_map", ("layer_budget_map",)),
        (
            "Per-Max-New-Tokens Results",
            "per_max_new_tokens",
            ("max_new_tokens",),
        ),
        (
            "Selected Ratio And Estimated Reduction",
            "selected_ratio_reduction",
            ("policy", "target_token_length"),
        ),
        (
            "Policy / Length / Map / Token Results",
            "policy_length_map_tokens",
            (
                "policy",
                "target_token_length",
                "layer_budget_map",
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
            "target",
            "map",
            "tokens",
            "KL gap",
            "exact gap",
            "token gap",
            "edit gap",
        ],
        [
            [
                row["policy"],
                row["target_token_length"],
                row["layer_budget_map"],
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
                "target",
                "map",
                "tokens",
                "exact",
                "token match",
                "edit",
                "KL",
            ],
            [
                [
                    row["policy"],
                    row["target_token_length"],
                    row["layer_budget_map"],
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
    lines.extend(["## Best Deployable Configuration", ""])
    best = summary["best_deployable_config"]
    if best is None:
        lines.append("No successful non-oracle configuration is available.")
    else:
        _table(
            lines,
            ["field", "value"],
            [[key, value] for key, value in best.items()],
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The selected-block ratio is context dependent. Its complement is "
        "reported only as an estimated active-block reduction, not realized "
        "runtime memory savings.",
        "",
        f"- `phase11_6_ready`: `{summary['readiness']['phase11_6_ready']}`",
        "- `phase12_ready`: `false` by design.",
        f"- {summary['readiness']['recommended_next_step']}",
        "",
        "## Caveats",
        "",
        "- This experiment runs outside vLLM.",
        "- No vLLM integration or active routing is implemented.",
        "- Generation uses greedy decoding only.",
        "- Prompts are synthetic unless a file is provided.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency claim is made.",
        "- This is a generation-quality probe, not a preservation claim.",
        "- GPT-2's context limit applies.",
        "",
        "## Recommended Next Step",
        "",
        "Inspect long-context worst cases and oracle gaps. A clean result "
        "supports broader prompt coverage or a larger model, not vLLM "
        "integration.",
    ])
    return "\n".join(lines) + "\n"


def _planned_rows(
    *,
    policies: list[str],
    layer_maps: list[dict[int, int]],
    target_lengths: list[int],
    max_new_tokens_values: list[int],
    num_prompts: int,
    adaptive: Any,
) -> list[dict[str, Any]]:
    return [
        {
            "status": "planned",
            "policy": policy,
            "layer_budget_map": adaptive.format_layer_budget_map(layer_map),
            "max_new_tokens": max_new_tokens,
            "target_token_length": target,
            "actual_prompt_token_lengths": [],
            "num_prompts": num_prompts,
            "failure_flags": [],
            "warnings": ["prompt tokenization deferred until execution"],
        }
        for policy in policies
        for layer_map in layer_maps
        for target in target_lengths
        for max_new_tokens in max_new_tokens_values
    ]


def _build_prompt_groups(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    target_lengths: list[int],
    max_new_tokens_values: list[int],
) -> dict[int, list[dict[str, Any]]]:
    max_prompt_tokens = args.max_length - max(max_new_tokens_values)
    source_prompts = (
        _read_prompt_file(args.prompts_file)
        if args.prompt_source == "file"
        else None
    )
    groups = {}
    for target in target_lengths:
        if source_prompts is not None:
            prompts = generate_file_prompts(
                tokenizer=tokenizer,
                source_prompts=source_prompts,
                target_token_length=target,
                num_prompts=args.num_prompts_per_length,
                max_prompt_tokens=max_prompt_tokens,
            )
        else:
            prompts = generate_synthetic_prompts(
                tokenizer=tokenizer,
                target_token_length=target,
                num_prompts=args.num_prompts_per_length,
                max_prompt_tokens=max_prompt_tokens,
                seed=args.seed + (100 if args.prompt_source == "long_builtin"
                                  else 0),
            )
        groups[target] = prompts
    return groups


def _run_rows(
    *,
    args: argparse.Namespace,
    policies: list[str],
    layer_maps: list[dict[int, int]],
    target_lengths: list[int],
    max_new_tokens_values: list[int],
    adaptive: Any,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    multilayer = _load_multilayer_module()
    phase11 = multilayer._load_phase11()
    generation_helpers = multilayer._load_phase11_generation()
    helpers = phase11._load_selected_attention_helpers()
    import torch

    torch.manual_seed(args.seed)
    device = helpers.resolve_device(args.device)
    dtype = helpers.resolve_dtype(args.dtype)
    tokenizer, model = phase11.load_hf_model(args.model, device, dtype)
    prompt_groups = _build_prompt_groups(
        args=args,
        tokenizer=tokenizer,
        target_lengths=target_lengths,
        max_new_tokens_values=max_new_tokens_values,
    )
    rows = []
    for policy in policies:
        for layer_map in layer_maps:
            for target in target_lengths:
                prompt_records = prompt_groups[target]
                for max_new_tokens in max_new_tokens_values:
                    map_key = adaptive.format_layer_budget_map(layer_map)
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
                                layer_budget_map=layer_map,
                                device=device,
                                phase11=phase11,
                                generation_helpers=generation_helpers,
                                helpers=helpers,
                            )
                            for index, record in enumerate(prompt_records)
                        ]
                        aggregate = multilayer.aggregate_rows(
                            prompt_rows,
                            layer_map,
                        )
                        selected_ratio = aggregate[
                            "average_selected_block_ratio_across_patched_layers"
                        ]
                        row = {
                            "status": "succeeded",
                            "policy": policy,
                            "layer_budget_map": map_key,
                            "max_new_tokens": max_new_tokens,
                            "target_token_length": target,
                            "actual_prompt_token_lengths": [
                                item["prompt_token_length"]
                                for item in prompt_rows
                            ],
                            **{
                                field: aggregate[field]
                                for field in (
                                    "num_prompts",
                                    *METRIC_FIELDS[:-1],
                                )
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
                            "policy": policy,
                            "layer_budget_map": map_key,
                            "max_new_tokens": max_new_tokens,
                            "target_token_length": target,
                            "actual_prompt_token_lengths": [],
                            "num_prompts": len(prompt_records),
                            "failure_flags": ["evaluation_failed"],
                            "warnings": [str(exc)],
                        })
                        if not args.continue_on_error:
                            return rows, prompt_groups
    return rows, prompt_groups


def validate_args(
    args: argparse.Namespace,
    *,
    target_lengths: list[int],
    max_new_tokens_values: list[int],
) -> None:
    for name in (
        "num_prompts_per_length",
        "block_size",
        "sketch_dim",
        "max_length",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.prompt_source == "file" and not args.prompts_file:
        raise ValueError("--prompts-file is required for --prompt-source file")
    prompt_limit = args.max_length - max(max_new_tokens_values)
    if prompt_limit <= 0:
        raise ValueError("max new tokens must be smaller than --max-length")
    if any(target > prompt_limit for target in target_lengths):
        raise ValueError(
            "target token lengths must not exceed max_length minus the "
            "largest max_new_tokens value"
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    adaptive = _load_adaptive_module()
    target_lengths = parse_target_token_lengths(args.target_token_lengths)
    max_new_tokens_values = adaptive.parse_int_csv(
        args.max_new_tokens_values,
        label="--max-new-tokens-values",
    )
    validate_args(
        args,
        target_lengths=target_lengths,
        max_new_tokens_values=max_new_tokens_values,
    )
    policies = adaptive.parse_policies(args.policies)
    layer_maps = adaptive.parse_layer_budget_maps(args.layer_budget_maps)
    config = {
        "model": args.model,
        "prompt_source": args.prompt_source,
        "prompts_file": args.prompts_file,
        "target_token_lengths": target_lengths,
        "num_prompts_per_length": args.num_prompts_per_length,
        "layer_budget_maps": [
            adaptive.format_layer_budget_map(layer_map)
            for layer_map in layer_maps
        ],
        "policies": policies,
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
    if args.dry_run:
        rows = _planned_rows(
            policies=policies,
            layer_maps=layer_maps,
            target_lengths=target_lengths,
            max_new_tokens_values=max_new_tokens_values,
            num_prompts=args.num_prompts_per_length,
            adaptive=adaptive,
        )
        prompt_groups = {}
    else:
        rows, prompt_groups = _run_rows(
            args=args,
            policies=policies,
            layer_maps=layer_maps,
            target_lengths=target_lengths,
            max_new_tokens_values=max_new_tokens_values,
            adaptive=adaptive,
        )
    summary = build_summary(rows, config=config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "long_context_generation_runs.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_json = output_dir / "long_context_generation_summary.json"
    summary_md = output_dir / "long_context_generation_summary.md"
    prompts_json = output_dir / "long_context_prompts.json"
    _write_json(summary_json, summary)
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    if prompt_groups:
        _write_json(
            prompts_json,
            {
                str(target): records
                for target, records in prompt_groups.items()
            },
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
            "phase11_6_ready": result["summary"]["readiness"][
                "phase11_6_ready"
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
