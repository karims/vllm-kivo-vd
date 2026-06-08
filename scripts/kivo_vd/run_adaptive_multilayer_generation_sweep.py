#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sweep adaptive multi-layer generation policies outside vLLM."""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LAYER_BUDGET_MAPS = "0:12,5:8,8:8,11:12"
DEFAULT_POLICIES = "query_key_block_score,oracle_topk"
DEFAULT_MAX_NEW_TOKENS = "32,64"
METRIC_FIELDS = (
    "exact_sequence_match_rate",
    "average_token_match_rate",
    "average_prefix_match_length",
    "average_normalized_edit_distance",
    "average_per_step_kl_divergence",
    "average_per_step_top1_match_rate",
    "average_selected_block_ratio_across_patched_layers",
)
NON_ORACLE_POLICIES = {
    "recent",
    "query_key_block_score",
    "count_sketch",
    "random_projection",
    "bidiagonal_sign_subsample",
}
ALLOWED_POLICIES = NON_ORACLE_POLICIES | {"oracle_topk"}


def _load_multilayer_module() -> Any:
    module_path = (
        Path(__file__).resolve().parent
        / "run_multilayer_selected_attention_generation_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_multilayer_selected_attention_generation_eval",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load helper script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep adaptive layer-budget maps, policies, generation lengths, "
            "and prompt sets outside vLLM."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompts-file")
    parser.add_argument(
        "--layer-budget-maps",
        default=DEFAULT_LAYER_BUDGET_MAPS,
        help="Semicolon-separated layer:budget maps.",
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
    parser.add_argument("--max-length", type=int, default=768)
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
        "--prompt-set",
        choices=["default", "extended"],
        default="default",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/kivo_vd/"
            "phase11_4_adaptive_multilayer_generation_sweep"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def parse_int_csv(value: str, *, label: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must contain integers") from exc
    if not result:
        raise ValueError(f"{label} must not be empty")
    if any(item <= 0 for item in result):
        raise ValueError(f"{label} values must be positive")
    return result


def parse_policies(value: str) -> list[str]:
    policies = [part.strip() for part in value.split(",") if part.strip()]
    if not policies:
        raise ValueError("--policies must not be empty")
    invalid = sorted(set(policies) - ALLOWED_POLICIES)
    if invalid:
        raise ValueError(f"unsupported policies: {', '.join(invalid)}")
    return list(dict.fromkeys(policies))


def parse_layer_budget_map(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError("layer budgets must use layer:budget syntax")
        layer_text, budget_text = entry.split(":", 1)
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


def format_layer_budget_map(value: dict[int, int]) -> str:
    return ",".join(f"{layer}:{budget}" for layer, budget in sorted(value.items()))


def parse_layer_budget_maps(value: str) -> list[dict[int, int]]:
    maps = [
        parse_layer_budget_map(part)
        for part in value.split(";")
        if part.strip()
    ]
    if not maps:
        raise ValueError("--layer-budget-maps must not be empty")
    unique: dict[str, dict[int, int]] = {}
    for layer_map in maps:
        unique[format_layer_budget_map(layer_map)] = layer_map
    return list(unique.values())


def extended_prompts(default_prompts: list[str]) -> list[str]:
    extra = [
        (
            "Continue this ordered list with the next item: one, two, three, "
            "four, five. The next item is"
        ),
        (
            "A service fails only after a cache entry expires. Explain the "
            "most useful first debugging hypothesis and the evidence needed."
        ),
        (
            "Design a small idempotent API for submitting jobs, checking job "
            "status, and retrying failed work without duplicate execution."
        ),
        (
            "Summarize the documentation rule: validate inputs, preserve "
            "backward compatibility, emit bounded diagnostics, and test the "
            "failure path. The summary is"
        ),
        (
            "Explain why the dot product of orthogonal vectors is zero and "
            "how this relates to projecting one vector onto another."
        ),
        (
            "A deployment has healthy CPU and memory but rising request "
            "latency. Give a concise sequence of system checks."
        ),
        (
            "Write a short Python-oriented explanation of why deterministic "
            "seeds are useful in numerical experiments."
        ),
    ]
    return default_prompts + extra


def read_prompts(
    *,
    prompts_file: str | None,
    prompt_set: str,
    default_prompts: list[str],
) -> list[str]:
    if prompts_file:
        path = Path(prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"prompts file is missing: {path}")
        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError(f"prompts file is empty: {path}")
        return prompts
    if prompt_set == "extended":
        return extended_prompts(default_prompts)
    return list(default_prompts)


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
    return flags


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
        summaries.append(summary)
    return summaries


def calculate_oracle_gaps(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "succeeded"]
    by_key = {
        (
            row["policy"],
            row["layer_budget_map"],
            row["max_new_tokens"],
        ): row
        for row in successful
    }
    gaps = []
    comparison_policies = sorted(
        {
            row["policy"]
            for row in successful
            if row["policy"] != "oracle_topk"
        }
    )
    maps_and_lengths = sorted({
        (row["layer_budget_map"], row["max_new_tokens"])
        for row in successful
    })
    for layer_map, max_new_tokens in maps_and_lengths:
        oracle = by_key.get(("oracle_topk", layer_map, max_new_tokens))
        if oracle is None:
            continue
        for policy in comparison_policies:
            candidate = by_key.get((policy, layer_map, max_new_tokens))
            if candidate is None:
                continue
            gaps.append({
                "policy": policy,
                "layer_budget_map": layer_map,
                "max_new_tokens": max_new_tokens,
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
            str(row["policy"]),
            str(row["layer_budget_map"]),
            int(row["max_new_tokens"]),
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
    qk_rows = [
        row
        for row in succeeded
        if row["policy"] == "query_key_block_score"
    ]
    phase11_5_ready = bool(qk_rows) and all(
        not row["failure_flags"] for row in qk_rows
    )
    if planned and not succeeded and not failed:
        recommended_next_step = "Execute the planned offline sweep."
    elif phase11_5_ready:
        recommended_next_step = (
            "Run more prompts and a larger model before any vLLM work."
        )
    else:
        recommended_next_step = (
            "Refine adaptive maps or selector quality in Phase 11.5."
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
        "per_map": _average_rows(rows, ("layer_budget_map",)),
        "per_max_new_tokens": _average_rows(rows, ("max_new_tokens",)),
        "policy_map_max_new_tokens": _average_rows(
            rows,
            ("policy", "layer_budget_map", "max_new_tokens"),
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
            "phase12_ready": False,
            "phase11_5_ready": phase11_5_ready,
            "recommended_next_step": recommended_next_step,
        },
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "greedy_generation_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "latency_claim": False,
            "generation_quality_probe_only": True,
            "gpt2_only_unless_configured_otherwise": True,
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
    group_fields: tuple[str, ...],
) -> None:
    headers = [
        *group_fields,
        "count",
        "exact",
        "token match",
        "edit distance",
        "avg KL",
        "step top-1",
        "selected ratio",
    ]
    table_rows = [
        [
            *(row[field] for field in group_fields),
            row["count"],
            row["exact_sequence_match_rate"],
            row["average_token_match_rate"],
            row["average_normalized_edit_distance"],
            row["average_per_step_kl_divergence"],
            row["average_per_step_top1_match_rate"],
            row["average_selected_block_ratio_across_patched_layers"],
        ]
        for row in rows
    ]
    _table(lines, headers, table_rows)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 11.4 Adaptive Multi-Layer Generation Sweep",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        "This is a greedy-generation quality probe outside vLLM.",
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
        ("Per-Policy Results", "per_policy", ("policy",)),
        ("Per-Map Results", "per_map", ("layer_budget_map",)),
        (
            "Per-Max-New-Tokens Results",
            "per_max_new_tokens",
            ("max_new_tokens",),
        ),
        (
            "Policy / Map / Max-New-Tokens Results",
            "policy_map_max_new_tokens",
            ("policy", "layer_budget_map", "max_new_tokens"),
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
    for key, rows in summary["worst_cases"].items():
        lines.extend([f"### {key.replace('_', ' ').title()}", ""])
        _table(
            lines,
            ["policy", "map", "tokens", "exact", "token match", "edit", "KL"],
            [
                [
                    row["policy"],
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
        f"- `phase11_5_ready`: `{summary['readiness']['phase11_5_ready']}`",
        "- `phase12_ready`: `false` by design.",
        f"- {summary['readiness']['recommended_next_step']}",
        "",
        "## Caveats",
        "",
        "- This experiment runs outside vLLM.",
        "- No vLLM integration or active routing is implemented.",
        "- Generation uses greedy decoding only.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency claim is made.",
        "- This is a generation-quality probe, not a preservation claim.",
        "- Results apply to GPT-2 unless another model is explicitly used.",
        "",
        "## Recommended Next Step",
        "",
        "Use the worst-case and oracle-gap rows to refine maps or selector "
        "quality. Even a clean sweep should be expanded to more prompts and a "
        "larger model before any vLLM integration is considered.",
    ])
    return "\n".join(lines) + "\n"


def _planned_rows(
    *,
    policies: list[str],
    layer_maps: list[dict[int, int]],
    max_new_tokens_values: list[int],
    prompt_set: str,
    num_prompts: int,
) -> list[dict[str, Any]]:
    return [
        {
            "status": "planned",
            "policy": policy,
            "layer_budget_map": format_layer_budget_map(layer_map),
            "max_new_tokens": max_new_tokens,
            "prompt_set": prompt_set,
            "num_prompts": num_prompts,
            "failure_flags": [],
            "warnings": [],
            "warning": None,
        }
        for policy in policies
        for layer_map in layer_maps
        for max_new_tokens in max_new_tokens_values
    ]


def _run_rows(
    *,
    args: argparse.Namespace,
    policies: list[str],
    layer_maps: list[dict[int, int]],
    max_new_tokens_values: list[int],
    prompts: list[str],
) -> list[dict[str, Any]]:
    multilayer = _load_multilayer_module()
    phase11 = multilayer._load_phase11()
    generation_helpers = multilayer._load_phase11_generation()
    helpers = phase11._load_selected_attention_helpers()
    import torch

    torch.manual_seed(args.seed)
    device = helpers.resolve_device(args.device)
    dtype = helpers.resolve_dtype(args.dtype)
    tokenizer, model = phase11.load_hf_model(args.model, device, dtype)
    rows = []
    for policy in policies:
        for layer_map in layer_maps:
            for max_new_tokens in max_new_tokens_values:
                map_key = format_layer_budget_map(layer_map)
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
                            prompt=prompt,
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
                        for index, prompt in enumerate(prompts)
                    ]
                    aggregate = multilayer.aggregate_rows(
                        prompt_rows,
                        layer_map,
                    )
                    row = {
                        "status": "succeeded",
                        "policy": policy,
                        "layer_budget_map": map_key,
                        "max_new_tokens": max_new_tokens,
                        "prompt_set": args.prompt_set,
                        **{
                            field: aggregate[field]
                            for field in ("num_prompts", *METRIC_FIELDS)
                        },
                        "warning": None,
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
                        "prompt_set": args.prompt_set,
                        "num_prompts": len(prompts),
                        "failure_flags": ["evaluation_failed"],
                        "warnings": [str(exc)],
                        "warning": str(exc),
                    })
                    if not args.continue_on_error:
                        return rows
    return rows


def validate_args(args: argparse.Namespace) -> None:
    for name in ("block_size", "sketch_dim", "max_length"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    policies = parse_policies(args.policies)
    layer_maps = parse_layer_budget_maps(args.layer_budget_maps)
    max_new_tokens_values = parse_int_csv(
        args.max_new_tokens_values,
        label="--max-new-tokens-values",
    )
    if args.prompts_file:
        default_prompts: list[str] = []
    else:
        multilayer = _load_multilayer_module()
        default_prompts = multilayer._load_phase11().built_in_prompts()
    prompts = read_prompts(
        prompts_file=args.prompts_file,
        prompt_set=args.prompt_set,
        default_prompts=default_prompts,
    )
    config = {
        "model": args.model,
        "prompt_set": args.prompt_set,
        "prompts_file": args.prompts_file,
        "num_prompts": len(prompts),
        "layer_budget_maps": [
            format_layer_budget_map(layer_map) for layer_map in layer_maps
        ],
        "policies": policies,
        "max_new_tokens_values": max_new_tokens_values,
        "block_size": args.block_size,
        "sketch_dim": args.sketch_dim,
        "block_score_reduction": args.block_score_reduction,
        "max_length": args.max_length,
        "teacher_forced_context": args.teacher_forced_context,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }
    rows = (
        _planned_rows(
            policies=policies,
            layer_maps=layer_maps,
            max_new_tokens_values=max_new_tokens_values,
            prompt_set=args.prompt_set,
            num_prompts=len(prompts),
        )
        if args.dry_run
        else _run_rows(
            args=args,
            policies=policies,
            layer_maps=layer_maps,
            max_new_tokens_values=max_new_tokens_values,
            prompts=prompts,
        )
    )
    summary = build_summary(rows, config=config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "adaptive_multilayer_generation_runs.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_json = (
        output_dir / "adaptive_multilayer_generation_summary.json"
    )
    summary_md = output_dir / "adaptive_multilayer_generation_summary.md"
    _write_json(summary_json, summary)
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return {
        "summary": summary,
        "rows_path": str(rows_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        result = run_sweep(args)
        print(json.dumps({
            "status": result["summary"]["status"],
            "counts": result["summary"]["counts"],
            "phase11_5_ready": result["summary"]["readiness"][
                "phase11_5_ready"
            ],
            "phase12_ready": False,
            "rows_path": result["rows_path"],
            "summary_json": result["summary_json"],
            "summary_md": result["summary_md"],
        }, separators=(",", ":")))
        return 0 if result["summary"]["counts"]["failed"] == 0 else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
