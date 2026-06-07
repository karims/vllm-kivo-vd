#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sweep selected-attention policies over real GPT-2 Q/K/V tensors."""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ALLOWED_POLICIES = {"recent", "first", "random", "oracle_topk"}
FAILURE_THRESHOLDS = {
    "average_cosine_similarity_below": 0.95,
    "min_cosine_similarity_below": 0.90,
    "average_relative_l2_error_above": 0.25,
    "max_relative_l2_error_above": 0.50,
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
            "Sweep selected-attention policies over real GPT-2 Q/K/V "
            "tensors outside vLLM."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompts-file")
    parser.add_argument("--layers", default="0,5,11")
    parser.add_argument("--budgets", default="4,8,16")
    parser.add_argument("--block-sizes", default="16")
    parser.add_argument(
        "--policies",
        default="recent,random,oracle_topk",
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
        default="outputs/kivo_vd/phase10_2_real_qkv_policy_sweep",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def built_in_prompts() -> list[str]:
    fillers = {
        "retrieval": (
            "A retrieval system records candidate memory blocks and compares "
            "their relevance before exact reranking. "
        ),
        "systems": (
            "A systems engineer studies scheduler traces, allocator state, "
            "cache residency, and reproducible debugging procedures. "
        ),
        "code": (
            "A Python function validates its inputs, transforms structured "
            "records, handles errors, and returns deterministic output. "
        ),
        "failure": (
            "Later paragraphs contain ordinary distractors about weather, "
            "gardens, books, transit, and office supplies. "
        ),
        "context": (
            "Long-context attention balances local continuity with retrieval "
            "of information introduced much earlier in the sequence. "
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


def read_prompts(path: str | None) -> list[str]:
    if path is None:
        return built_in_prompts()
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompts file is missing: {prompt_path}")
    prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("prompts file contains no non-empty prompts")
    return prompts


def resolve_output_dir(output_dir: str | Path, run_name: str | None) -> Path:
    base = Path(output_dir)
    return base / run_name if run_name else base


def build_combinations(
    layers: list[int],
    budgets: list[int],
    block_sizes: list[int],
    policies: list[str],
) -> list[dict[str, Any]]:
    if any(value <= 0 for value in budgets):
        raise ValueError("budgets must be positive")
    if any(value <= 0 for value in block_sizes):
        raise ValueError("block sizes must be positive")
    return [
        {
            "layer_index": layer,
            "candidate_budget_blocks": budget,
            "block_size": block_size,
            "policy": policy,
        }
        for layer in layers
        for budget in budgets
        for block_size in block_sizes
        for policy in policies
    ]


def _load_evaluator() -> Any:
    module_path = (
        Path(__file__).resolve().parent
        / "run_real_qkv_selected_attention_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_real_qkv_selected_attention_eval",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load evaluator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failure_flags(row: dict[str, Any]) -> list[str]:
    if row.get("status") != "succeeded":
        return ["run_failed"]
    flags = []
    if row["average_cosine_similarity"] < 0.95:
        flags.append("average_cosine_below_0.95")
    if row["min_cosine_similarity"] < 0.90:
        flags.append("min_cosine_below_0.90")
    if row["average_relative_l2_error"] > 0.25:
        flags.append("average_relative_l2_above_0.25")
    if row["max_relative_l2_error"] > 0.50:
        flags.append("max_relative_l2_above_0.50")
    return flags


def _average(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def group_averages(
    rows: list[dict[str, Any]],
    group_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "succeeded":
            grouped[row[group_field]].append(row)
    result = []
    for group_value, group_rows in grouped.items():
        result.append({
            group_field: group_value,
            "count": len(group_rows),
            "average_cosine_similarity": _average(
                group_rows, "average_cosine_similarity"
            ),
            "min_cosine_similarity": min(
                row["min_cosine_similarity"] for row in group_rows
            ),
            "average_relative_l2_error": _average(
                group_rows, "average_relative_l2_error"
            ),
            "max_relative_l2_error": max(
                row["max_relative_l2_error"] for row in group_rows
            ),
            "average_attention_mass_captured": _average(
                group_rows, "average_attention_mass_captured"
            ),
        })
    return sorted(result, key=lambda row: str(row[group_field]))


def calculate_oracle_gaps(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "succeeded"]
    grouped = {
        (
            row["layer_index"],
            row["candidate_budget_blocks"],
            row["block_size"],
            row["policy"],
        ): row
        for row in successful
    }
    gaps = []
    for row in successful:
        if row["policy"] == "oracle_topk":
            continue
        oracle = grouped.get((
            row["layer_index"],
            row["candidate_budget_blocks"],
            row["block_size"],
            "oracle_topk",
        ))
        if oracle is None:
            continue
        gaps.append({
            "policy": row["policy"],
            "layer_index": row["layer_index"],
            "candidate_budget_blocks": row["candidate_budget_blocks"],
            "block_size": row["block_size"],
            "cosine_gap": (
                oracle["average_cosine_similarity"]
                - row["average_cosine_similarity"]
            ),
            "relative_l2_gap": (
                row["average_relative_l2_error"]
                - oracle["average_relative_l2_error"]
            ),
            "attention_mass_gap": (
                oracle["average_attention_mass_captured"]
                - row["average_attention_mass_captured"]
            ),
        })
    return sorted(
        gaps,
        key=lambda row: (
            row["layer_index"],
            row["candidate_budget_blocks"],
            row["block_size"],
            row["policy"],
        ),
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "succeeded"]
    failed = [row for row in rows if row.get("status") == "failed"]
    summary: dict[str, Any] = {
        "num_runs": len(rows),
        "num_succeeded": len(successful),
        "num_failed": len(failed),
        "failure_thresholds": FAILURE_THRESHOLDS,
        "per_policy": group_averages(successful, "policy"),
        "per_layer": group_averages(successful, "layer_index"),
        "per_budget": group_averages(
            successful, "candidate_budget_blocks"
        ),
        "oracle_gaps": calculate_oracle_gaps(successful),
    }
    if not successful:
        summary.update({
            "best_by_average_cosine": None,
            "worst_by_average_cosine": None,
            "worst_by_min_cosine": None,
            "best_by_average_relative_l2": None,
            "worst_by_max_relative_l2": None,
            "best_by_attention_mass": None,
        })
        return summary
    summary.update({
        "best_by_average_cosine": max(
            successful, key=lambda row: row["average_cosine_similarity"]
        ),
        "worst_by_average_cosine": min(
            successful, key=lambda row: row["average_cosine_similarity"]
        ),
        "worst_by_min_cosine": min(
            successful, key=lambda row: row["min_cosine_similarity"]
        ),
        "best_by_average_relative_l2": min(
            successful, key=lambda row: row["average_relative_l2_error"]
        ),
        "worst_by_max_relative_l2": max(
            successful, key=lambda row: row["max_relative_l2_error"]
        ),
        "best_by_attention_mass": max(
            successful,
            key=lambda row: row["average_attention_mass_captured"],
        ),
    })
    return summary


def _row_from_report(
    combination: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    aggregate = report["aggregate"]
    prompt_rows = report["per_prompt"]
    row = {
        **combination,
        "status": "succeeded",
        "warning": None,
        "num_prompts": aggregate["num_prompts"],
        "average_cosine_similarity": aggregate[
            "average_cosine_similarity"
        ],
        "min_cosine_similarity": aggregate["min_cosine_similarity"],
        "average_relative_l2_error": aggregate[
            "average_relative_l2_error"
        ],
        "max_relative_l2_error": aggregate["max_relative_l2_error"],
        "average_attention_mass_captured": aggregate[
            "average_attention_mass_captured"
        ],
        "average_selected_block_ratio": _average(
            prompt_rows, "selected_block_ratio"
        ),
        "average_selected_token_ratio": _average(
            prompt_rows, "selected_token_ratio"
        ),
    }
    row["failure_flags"] = failure_flags(row)
    return row


def _extract_qkv_cache(
    *,
    evaluator: Any,
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    layers: list[int],
    max_length: int,
    device: Any,
) -> dict[tuple[int, int], dict[str, Any]]:
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        with evaluator.torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        for layer_index in layers:
            if layer_index < 0 or layer_index >= len(model.transformer.h):
                raise ValueError(
                    f"layer {layer_index} is outside "
                    f"[0, {len(model.transformer.h)})"
                )
            block = model.transformer.h[layer_index]
            attention_input = block.ln_1(
                outputs.hidden_states[layer_index]
            )
            fused = block.attn.c_attn(attention_input)
            query, keys, values = evaluator.split_gpt2_fused_qkv(
                fused,
                int(model.config.n_head),
            )
            cache[(prompt_index, layer_index)] = {
                "query": query,
                "keys": keys,
                "values": values,
                "token_length": int(input_ids.shape[1]),
            }
    return cache


def _evaluate_combination(
    *,
    evaluator: Any,
    combination: dict[str, Any],
    prompts: list[str],
    cache: dict[tuple[int, int], dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    prompt_rows = []
    layer = combination["layer_index"]
    block_size = combination["block_size"]
    budget = combination["candidate_budget_blocks"]
    policy = combination["policy"]
    for prompt_index, _prompt in enumerate(prompts):
        tensors = cache[(prompt_index, layer)]
        full_output, probabilities = evaluator.last_query_attention(
            tensors["query"],
            tensors["keys"],
            tensors["values"],
        )
        masses = evaluator.block_attention_mass(probabilities, block_size)
        selected_ids = evaluator.select_block_ids(
            policy=policy,
            num_blocks=int(masses.shape[0]),
            candidate_budget_blocks=budget,
            seed=seed + prompt_index,
            masses=masses,
        )
        selected_keys = evaluator.gather_selected_blocks(
            tensors["keys"], selected_ids, block_size
        )
        selected_values = evaluator.gather_selected_blocks(
            tensors["values"], selected_ids, block_size
        )
        selected_output, _ = evaluator.last_query_attention(
            tensors["query"][:, :, -1:, :],
            selected_keys,
            selected_values,
        )
        metrics = evaluator.calculate_metrics(full_output, selected_output)
        token_length = tensors["token_length"]
        prompt_rows.append({
            "cosine_similarity": metrics["cosine_similarity"],
            "relative_l2_error": metrics["relative_l2_error"],
            "attention_mass_captured": evaluator.captured_attention_mass(
                masses, selected_ids
            ),
            "selected_block_ratio": len(selected_ids) / int(masses.shape[0]),
            "selected_token_ratio": (
                int(selected_keys.shape[2]) / token_length
            ),
        })
    return evaluator.build_report(
        config={
            **combination,
            "selection_policy": policy,
            "seed": seed,
        },
        rows=prompt_rows,
    )


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
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


def render_markdown(
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Kivo-VD Phase 10.2 Real-QKV Policy Sweep",
        "",
        "**Status:** Standalone real-model Q/K/V attention-output comparison "
        "outside vLLM.",
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
        ],
    )
    for title, key, group_field in (
        ("Per-Policy", "per_policy", "policy"),
        ("Per-Layer", "per_layer", "layer_index"),
        ("Per-Budget", "per_budget", "candidate_budget_blocks"),
    ):
        lines.extend(["", f"## {title}", ""])
        _append_table(
            lines,
            [
                group_field,
                "count",
                "avg cosine",
                "min cosine",
                "avg rel L2",
                "max rel L2",
                "avg mass",
            ],
            [
                [
                    row[group_field],
                    row["count"],
                    row["average_cosine_similarity"],
                    row["min_cosine_similarity"],
                    row["average_relative_l2_error"],
                    row["max_relative_l2_error"],
                    row["average_attention_mass_captured"],
                ]
                for row in summary[key]
            ],
        )
    lines.extend(["", "## Worst Cases", ""])
    worst_rows = [
        ("worst average cosine", summary["worst_by_average_cosine"]),
        ("worst minimum cosine", summary["worst_by_min_cosine"]),
        ("worst maximum relative L2", summary["worst_by_max_relative_l2"]),
    ]
    _append_table(
        lines,
        ["criterion", "policy", "layer", "budget", "block", "flags"],
        [
            [
                label,
                row["policy"] if row else None,
                row["layer_index"] if row else None,
                row["candidate_budget_blocks"] if row else None,
                row["block_size"] if row else None,
                row["failure_flags"] if row else None,
            ]
            for label, row in worst_rows
        ],
    )
    lines.extend(["", "## Oracle Gaps", ""])
    _append_table(
        lines,
        [
            "policy",
            "layer",
            "budget",
            "block",
            "cosine gap",
            "relative L2 gap",
            "mass gap",
        ],
        [
            [
                row["policy"],
                row["layer_index"],
                row["candidate_budget_blocks"],
                row["block_size"],
                row["cosine_gap"],
                row["relative_l2_gap"],
                row["attention_mass_gap"],
            ]
            for row in summary["oracle_gaps"]
        ],
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Oracle top-k is an undeployable upper bound. Consistently strong "
        "oracle results with weak recent or random rows identify candidate "
        "selection as the bottleneck. Oracle failures at low budgets indicate "
        "that selected attention itself may be risky at those budgets.",
        "",
        "Failure flags are research heuristics, not model-quality thresholds.",
        "",
        "## Caveats",
        "",
        "- Q/K/V projections come from a real GPT-2-style model.",
        "- Evaluation runs outside vLLM.",
        "- No logits or generation quality is measured.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency improvement is claimed.",
        "",
        "## Recommended Next Step",
        "",
        "If oracle remains strong, Phase 10.3 should evaluate sketch-based "
        "selectors on the same real Q/K/V evidence before any vLLM attention "
        "integration.",
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
        prompts = read_prompts(args.prompts_file)
        combinations = build_combinations(
            layers, budgets, block_sizes, policies
        )
        output_dir = resolve_output_dir(args.output_dir, args.run_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        runs_path = output_dir / "policy_sweep_runs.jsonl"
        summary_path = output_dir / "policy_sweep_summary.json"
        markdown_path = output_dir / "policy_sweep_summary.md"
        config = {
            "model": args.model,
            "prompts_file": args.prompts_file,
            "num_prompts": len(prompts),
            "layers": layers,
            "budgets": budgets,
            "block_sizes": block_sizes,
            "policies": policies,
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
            print(json.dumps(payload, separators=(",", ":")))
            return 0

        evaluator = _load_evaluator()
        device = evaluator.resolve_device(args.device)
        dtype = evaluator.resolve_dtype(args.dtype)
        tokenizer, model = evaluator.load_hf_model(
            args.model, device, dtype
        )
        cache = _extract_qkv_cache(
            evaluator=evaluator,
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            layers=layers,
            max_length=args.max_length,
            device=device,
        )
        rows = []
        for combination in combinations:
            try:
                report = _evaluate_combination(
                    evaluator=evaluator,
                    combination=combination,
                    prompts=prompts,
                    cache=cache,
                    seed=args.seed,
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
        payload = {
            "config": config,
            "started_at": started_at,
            "ended_at": _iso_now(),
            "success": success,
            "dry_run": False,
            "summary": summary,
            "outputs": {
                "runs_jsonl": str(runs_path),
                "summary_json": str(summary_path),
                "summary_markdown": str(markdown_path),
            },
            "caveats": {
                "real_model_qkv": True,
                "outside_vllm": True,
                "no_logits_or_generation_quality": True,
                "active_routing": False,
                "measured_runtime_reduction": False,
            },
        }
        _write_jsonl(runs_path, rows)
        _write_json(summary_path, payload)
        markdown_path.write_text(
            render_markdown(config=config, summary=summary),
            encoding="utf-8",
        )
        compact = {
            "num_runs": summary["num_runs"],
            "num_succeeded": summary["num_succeeded"],
            "num_failed": summary["num_failed"],
            "best_policy_by_average_cosine": (
                summary["best_by_average_cosine"]["policy"]
                if summary["best_by_average_cosine"]
                else None
            ),
            "worst_policy_layer_budget": (
                {
                    key: summary["worst_by_max_relative_l2"][key]
                    for key in (
                        "policy",
                        "layer_index",
                        "candidate_budget_blocks",
                        "block_size",
                    )
                }
                if summary["worst_by_max_relative_l2"]
                else None
            ),
            "outputs": payload["outputs"],
            "outside_vllm": True,
            "no_logits_or_generation_quality": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
        }
        print(json.dumps(compact, separators=(",", ":")))
        return 0 if success else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
