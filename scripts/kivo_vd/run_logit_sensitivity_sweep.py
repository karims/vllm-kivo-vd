#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sweep single-layer selected-attention logit sensitivity outside vLLM."""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE_POLICIES = {"recent", "oracle_topk", "query_key_block_score"}
SKETCH_POLICIES = {
    "count_sketch",
    "random_projection",
    "bidiagonal_sign_subsample",
}
ALLOWED_POLICIES = BASE_POLICIES | SKETCH_POLICIES
FAILURE_THRESHOLDS = {
    "top1_match_rate_below": 0.95,
    "average_kl_above": 0.01,
    "logits_relative_l2_above": 0.05,
    "average_top5_overlap_below": 4.0,
}


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_csv(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("comma-separated argument must not be empty")
    return result


def _parse_int_csv(value: str) -> list[int]:
    result = [int(part) for part in _parse_csv(value)]
    if any(item < 0 for item in result):
        raise ValueError("integer list values must be non-negative")
    return result


def parse_policies(value: str) -> list[str]:
    policies = _parse_csv(value)
    invalid = [policy for policy in policies if policy not in ALLOWED_POLICIES]
    if invalid:
        raise ValueError(
            f"unsupported policies {invalid}; choose from "
            f"{sorted(ALLOWED_POLICIES)}"
        )
    return policies


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep single-layer selected-attention logit sensitivity on "
            "GPT-2 outside vLLM."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompts-file")
    parser.add_argument("--layers", default="0,5,8,11")
    parser.add_argument("--budgets", default="8,16,32")
    parser.add_argument("--block-sizes", default="16")
    parser.add_argument(
        "--policies",
        default="query_key_block_score,oracle_topk",
    )
    parser.add_argument("--sketch-dims", default="32")
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
        "--output-dir",
        default="outputs/kivo_vd/phase11_1_logit_sensitivity_sweep",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def build_combinations(
    *,
    layers: list[int],
    budgets: list[int],
    block_sizes: list[int],
    policies: list[str],
    sketch_dims: list[int],
) -> list[dict[str, Any]]:
    if any(layer < 0 for layer in layers):
        raise ValueError("layers must be non-negative")
    if any(value <= 0 for value in budgets):
        raise ValueError("budgets must be positive")
    if any(value <= 0 for value in block_sizes):
        raise ValueError("block sizes must be positive")
    if any(value <= 0 for value in sketch_dims):
        raise ValueError("sketch dimensions must be positive")
    combinations = []
    for layer in layers:
        for budget in budgets:
            for block_size in block_sizes:
                for policy in policies:
                    dims: list[int | None]
                    dims = sketch_dims if policy in SKETCH_POLICIES else [None]
                    for sketch_dim in dims:
                        combinations.append({
                            "policy": policy,
                            "layer_index": layer,
                            "candidate_budget_blocks": budget,
                            "block_size": block_size,
                            "sketch_dim": sketch_dim,
                        })
    return combinations


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
        raise RuntimeError(f"unable to load Phase 11.0: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failure_flags(row: dict[str, Any]) -> list[str]:
    if row.get("status") != "succeeded":
        return ["run_failed"]
    flags = []
    if row["top1_match_rate"] < 0.95:
        flags.append("top1_match_rate_below_0.95")
    if row["average_kl_divergence"] > 0.01:
        flags.append("average_kl_above_0.01")
    if row["average_logits_relative_l2_error"] > 0.05:
        flags.append("logits_relative_l2_above_0.05")
    if row["average_top5_overlap"] < 4.0:
        flags.append("average_top5_overlap_below_4")
    return flags


METRIC_FIELDS = (
    "average_logits_cosine_similarity",
    "average_logits_relative_l2_error",
    "average_kl_divergence",
    "top1_match_rate",
    "average_top5_overlap",
    "average_top10_overlap",
    "average_attention_output_cosine",
    "average_attention_output_relative_l2",
)


def _average(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def group_averages(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "succeeded":
            grouped[tuple(row.get(field) for field in group_fields)].append(
                row
            )
    result = []
    for values, group_rows in grouped.items():
        summary = {
            field: value for field, value in zip(group_fields, values)
        }
        summary["count"] = len(group_rows)
        summary.update({
            field: _average(group_rows, field) for field in METRIC_FIELDS
        })
        summary["min_top1_match_rate"] = min(
            row["top1_match_rate"] for row in group_rows
        )
        summary["max_average_kl_divergence"] = max(
            row["average_kl_divergence"] for row in group_rows
        )
        summary["max_average_logits_relative_l2_error"] = max(
            row["average_logits_relative_l2_error"] for row in group_rows
        )
        summary["min_average_top5_overlap"] = min(
            row["average_top5_overlap"] for row in group_rows
        )
        result.append(summary)
    return sorted(
        result,
        key=lambda row: tuple(str(row.get(field)) for field in group_fields),
    )


def calculate_oracle_gaps(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "succeeded"]
    oracle_rows = {
        (
            row["layer_index"],
            row["candidate_budget_blocks"],
            row["block_size"],
        ): row
        for row in successful
        if row["policy"] == "oracle_topk"
    }
    gaps = []
    for row in successful:
        if row["policy"] == "oracle_topk":
            continue
        oracle = oracle_rows.get((
            row["layer_index"],
            row["candidate_budget_blocks"],
            row["block_size"],
        ))
        if oracle is None:
            continue
        gaps.append({
            "policy": row["policy"],
            "sketch_dim": row.get("sketch_dim"),
            "layer_index": row["layer_index"],
            "candidate_budget_blocks": row["candidate_budget_blocks"],
            "block_size": row["block_size"],
            "kl_gap": (
                row["average_kl_divergence"]
                - oracle["average_kl_divergence"]
            ),
            "top1_gap": (
                oracle["top1_match_rate"] - row["top1_match_rate"]
            ),
            "logits_l2_gap": (
                row["average_logits_relative_l2_error"]
                - oracle["average_logits_relative_l2_error"]
            ),
            "attention_l2_gap": (
                row["average_attention_output_relative_l2"]
                - oracle["average_attention_output_relative_l2"]
            ),
        })
    return sorted(
        gaps,
        key=lambda row: (
            row["layer_index"],
            row["candidate_budget_blocks"],
            row["block_size"],
            row["policy"],
            str(row.get("sketch_dim")),
        ),
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "succeeded"]
    failed = [row for row in rows if row.get("status") == "failed"]
    per_policy_sketch = group_averages(
        successful,
        ("policy", "sketch_dim"),
    )
    deployable = [
        row for row in per_policy_sketch if row["policy"] != "oracle_topk"
    ]
    best_deployable = (
        sorted(
            deployable,
            key=lambda row: (
                -row["top1_match_rate"],
                row["average_kl_divergence"],
                row["average_logits_relative_l2_error"],
            ),
        )[0]
        if deployable
        else None
    )
    practical_qk_rows = [
        row
        for row in successful
        if row["policy"] == "query_key_block_score"
        and row["candidate_budget_blocks"] >= 8
    ]
    phase11_2_ready = bool(practical_qk_rows) and all(
        row["top1_match_rate"] >= 0.95
        and row["average_kl_divergence"] <= 0.01
        for row in practical_qk_rows
    )
    summary: dict[str, Any] = {
        "num_runs": len(rows),
        "num_succeeded": len(successful),
        "num_failed": len(failed),
        "failure_thresholds": FAILURE_THRESHOLDS,
        "per_policy": group_averages(successful, ("policy",)),
        "per_policy_sketch_dim": per_policy_sketch,
        "per_layer": group_averages(successful, ("layer_index",)),
        "per_budget": group_averages(
            successful,
            ("candidate_budget_blocks",),
        ),
        "per_policy_layer_budget": group_averages(
            successful,
            (
                "policy",
                "sketch_dim",
                "layer_index",
                "candidate_budget_blocks",
            ),
        ),
        "oracle_gaps": calculate_oracle_gaps(successful),
        "best_deployable_policy": best_deployable,
        "phase11_2_ready": phase11_2_ready,
        "recommended_next_step": (
            "Proceed to Phase 11.2 generation-level evaluation outside vLLM."
            if phase11_2_ready
            else "Improve selector evidence before generation-level tests."
        ),
    }
    if not successful:
        summary.update({
            "worst_by_top1_match_rate": None,
            "worst_by_average_kl": None,
            "worst_by_logits_relative_l2": None,
            "worst_by_top5_overlap": None,
        })
        return summary
    summary.update({
        "worst_by_top1_match_rate": min(
            successful, key=lambda row: row["top1_match_rate"]
        ),
        "worst_by_average_kl": max(
            successful, key=lambda row: row["average_kl_divergence"]
        ),
        "worst_by_logits_relative_l2": max(
            successful,
            key=lambda row: row["average_logits_relative_l2_error"],
        ),
        "worst_by_top5_overlap": min(
            successful, key=lambda row: row["average_top5_overlap"]
        ),
    })
    return summary


def _row_from_report(
    combination: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    row = {
        **combination,
        "status": "succeeded",
        "warning": None,
        **report["aggregate"],
    }
    row["failure_flags"] = failure_flags(row)
    return row


def _evaluate_combination(
    *,
    phase11: Any,
    combination: dict[str, Any],
    prompts: list[str],
    tokenizer: Any,
    model: Any,
    device: Any,
    helpers: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    namespace = argparse.Namespace(
        layer_idx=combination["layer_index"],
        block_size=combination["block_size"],
        candidate_budget_blocks=combination["candidate_budget_blocks"],
        selection_policy=combination["policy"],
        sketch_dim=combination.get("sketch_dim") or config["sketch_dims"][0],
        block_score_reduction=config["block_score_reduction"],
        max_length=config["max_length"],
        seed=config["seed"],
    )
    rows = [
        phase11.evaluate_prompt(
            prompt=prompt,
            prompt_index=index,
            tokenizer=tokenizer,
            model=model,
            args=namespace,
            device=device,
            helpers=helpers,
        )
        for index, prompt in enumerate(prompts)
    ]
    return phase11.build_report(config={**combination}, rows=rows)


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


def _metric_table(
    lines: list[str],
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> None:
    headers = [
        *group_fields,
        "count",
        "top-1",
        "avg KL",
        "logits rel L2",
        "top-5",
        "attention rel L2",
    ]
    values = [
        [
            *(row.get(field) for field in group_fields),
            row["count"],
            row["top1_match_rate"],
            row["average_kl_divergence"],
            row["average_logits_relative_l2_error"],
            row["average_top5_overlap"],
            row["average_attention_output_relative_l2"],
        ]
        for row in rows
    ]
    _append_table(lines, headers, values)


def render_markdown(
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Kivo-VD Phase 11.1 Logit-Sensitivity Sweep",
        "",
        "**Status:** Single-layer, last-token logits sensitivity on real "
        "GPT-2 Q/K/V outside vLLM.",
        "",
        "## Configuration",
        "",
    ]
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in config.items()],
    )
    lines.extend(["", "## High-Level Summary", ""])
    _append_table(
        lines,
        ["metric", "value"],
        [
            ["num_runs", summary["num_runs"]],
            ["num_succeeded", summary["num_succeeded"]],
            ["num_failed", summary["num_failed"]],
            ["phase11_2_ready", summary["phase11_2_ready"]],
        ],
    )
    for title, key, fields in (
        ("Per-Policy", "per_policy", ("policy",)),
        ("Per-Layer", "per_layer", ("layer_index",)),
        ("Per-Budget", "per_budget", ("candidate_budget_blocks",)),
        (
            "Policy, Layer, And Budget",
            "per_policy_layer_budget",
            (
                "policy",
                "sketch_dim",
                "layer_index",
                "candidate_budget_blocks",
            ),
        ),
    ):
        lines.extend(["", f"## {title}", ""])
        _metric_table(lines, summary[key], fields)

    lines.extend(["", "## Worst Cases", ""])
    worst = [
        ("lowest top-1", summary["worst_by_top1_match_rate"]),
        ("highest KL", summary["worst_by_average_kl"]),
        ("highest logits L2", summary["worst_by_logits_relative_l2"]),
        ("lowest top-5 overlap", summary["worst_by_top5_overlap"]),
    ]
    _append_table(
        lines,
        ["criterion", "policy", "layer", "budget", "sketch dim"],
        [
            [
                label,
                row["policy"] if row else None,
                row["layer_index"] if row else None,
                row["candidate_budget_blocks"] if row else None,
                row.get("sketch_dim") if row else None,
            ]
            for label, row in worst
        ],
    )
    lines.extend(["", "## Oracle Gaps", ""])
    _append_table(
        lines,
        [
            "policy",
            "sketch dim",
            "layer",
            "budget",
            "KL gap",
            "top-1 gap",
            "logits L2 gap",
            "attention L2 gap",
        ],
        [
            [
                row["policy"],
                row["sketch_dim"],
                row["layer_index"],
                row["candidate_budget_blocks"],
                row["kl_gap"],
                row["top1_gap"],
                row["logits_l2_gap"],
                row["attention_l2_gap"],
            ]
            for row in summary["oracle_gaps"]
        ],
    )
    lines.extend(["", "## Best Deployable Policy", ""])
    best = summary["best_deployable_policy"] or {}
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in best.items()],
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The failure thresholds are research heuristics, not model-quality "
        "claims. A strong oracle with a weak deployable policy identifies "
        "selection as the bottleneck. An unstable oracle indicates that the "
        "selected-attention budget itself may be too aggressive.",
        "",
        "## Caveats",
        "",
        "- Evaluation runs outside vLLM.",
        "- No vLLM integration is implemented or authorized.",
        "- Each run patches one layer and the last-token attention output.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency improvement is claimed.",
        "- Full generation quality is not measured or preserved by claim.",
        "",
        "## Recommended Next Step",
        "",
        summary["recommended_next_step"],
    ])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        layers = _parse_int_csv(args.layers)
        budgets = _parse_int_csv(args.budgets)
        block_sizes = _parse_int_csv(args.block_sizes)
        policies = parse_policies(args.policies)
        sketch_dims = _parse_int_csv(args.sketch_dims)
        combinations = build_combinations(
            layers=layers,
            budgets=budgets,
            block_sizes=block_sizes,
            policies=policies,
            sketch_dims=sketch_dims,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        runs_path = output_dir / "logit_sensitivity_runs.jsonl"
        summary_path = output_dir / "logit_sensitivity_summary.json"
        markdown_path = output_dir / "logit_sensitivity_summary.md"
        phase11 = _load_phase11()
        prompts = phase11.read_prompts(None, args.prompts_file)
        config = {
            "model": args.model,
            "prompts_file": args.prompts_file,
            "num_prompts": len(prompts),
            "layers": layers,
            "budgets": budgets,
            "block_sizes": block_sizes,
            "policies": policies,
            "sketch_dims": sketch_dims,
            "block_score_reduction": args.block_score_reduction,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "dry_run": bool(args.dry_run),
            "continue_on_error": bool(args.continue_on_error),
        }
        started_at = _iso_now()
        if args.dry_run:
            rows = [
                {
                    **combination,
                    "status": "planned",
                    "warning": None,
                    "failure_flags": [],
                }
                for combination in combinations
            ]
            summary = summarize_rows(rows)
            payload = {
                "config": config,
                "started_at": started_at,
                "ended_at": _iso_now(),
                "success": True,
                "dry_run": True,
                "summary": summary,
            }
            _write_jsonl(runs_path, rows)
            _write_json(summary_path, payload)
            markdown_path.write_text(
                render_markdown(config=config, summary=summary),
                encoding="utf-8",
            )
            print(json.dumps(payload, separators=(",", ":")))
            return 0

        helpers = phase11._load_selected_attention_helpers()
        device = helpers.resolve_device(args.device)
        dtype = helpers.resolve_dtype(args.dtype)
        tokenizer, model = phase11.load_hf_model(
            args.model,
            device,
            dtype,
        )
        rows = []
        for combination in combinations:
            try:
                report = _evaluate_combination(
                    phase11=phase11,
                    combination=combination,
                    prompts=prompts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    helpers=helpers,
                    config=config,
                )
                rows.append(_row_from_report(combination, report))
            except Exception as exc:
                rows.append({
                    **combination,
                    "status": "failed",
                    "warning": str(exc),
                    "failure_flags": ["run_failed"],
                })
                if not args.continue_on_error:
                    break
        summary = summarize_rows(rows)
        success = summary["num_failed"] == 0
        caveats = {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "single_layer_patch_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "latency_improvement": False,
            "generation_quality_not_fully_measured": True,
        }
        payload = {
            "config": config,
            "started_at": started_at,
            "ended_at": _iso_now(),
            "success": success,
            "dry_run": False,
            "summary": summary,
            "caveats": caveats,
            "outputs": {
                "runs_jsonl": str(runs_path),
                "summary_json": str(summary_path),
                "summary_markdown": str(markdown_path),
            },
        }
        _write_jsonl(runs_path, rows)
        _write_json(summary_path, payload)
        markdown_path.write_text(
            render_markdown(config=config, summary=summary),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "success": success,
                    "num_runs": summary["num_runs"],
                    "num_succeeded": summary["num_succeeded"],
                    "num_failed": summary["num_failed"],
                    "phase11_2_ready": summary["phase11_2_ready"],
                    "best_deployable_policy": (
                        summary["best_deployable_policy"] or {}
                    ).get("policy"),
                    "outputs": payload["outputs"],
                    **caveats,
                },
                separators=(",", ":"),
            )
        )
        return 0 if success else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
