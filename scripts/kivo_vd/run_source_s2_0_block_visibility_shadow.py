#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S2.0 source-level block visibility shadow probe."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1

DEFAULT_PROMPTS = [
    "Kivo S2 block visibility shadow probe.",
    (
        "A careful runtime experiment observes allocation metadata without "
        "changing the model path. The prompt is long enough to exercise more "
        "than a single short phrase while keeping the expected answer simple."
    ),
    (
        "Repeated context segment for block visibility. " * 16
        + "Summarize the repeated context in one sentence."
    ),
    (
        "Longer repeated context segment for KV block observation. " * 28
        + "What kind of context was repeated?"
    ),
]

_SOURCE_ENV_KEYS = [
    "KIVO_SOURCE_ENABLE",
    "KIVO_SOURCE_OBS_PATH",
    "KIVO_SOURCE_ACTIVE",
    "KIVO_SOURCE_POLICY",
    "KIVO_SOURCE_MAX_MUTATIONS",
    "KIVO_SOURCE_FAIL_CLOSED",
    "KIVO_SOURCE_BUDGET_RATIO",
    "KIVO_SOURCE_KEEP_RECENT_BLOCKS",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase S2.0 block visibility and shadow selection."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to evaluate. May be repeated.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--keep-recent-blocks", type=int, default=1)
    parser.add_argument(
        "--observation-dir",
        default="outputs/kivo_vd/runs/source_s2_0_observations",
    )
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s2_0_block_visibility_shadow.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s2_0_block_visibility_shadow.md"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _resolve_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    return [str(prompt) for prompt in prompts if str(prompt).strip()]


def _clear_source_env() -> None:
    for key in _SOURCE_ENV_KEYS:
        os.environ.pop(key, None)


def _set_shadow_env(
    observation_path: str | Path,
    *,
    budget_ratio: float,
    keep_recent_blocks: int,
) -> None:
    _clear_source_env()
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_OBS_PATH"] = str(observation_path)
    os.environ["KIVO_SOURCE_POLICY"] = "sketch_shadow_blocks"
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"
    os.environ["KIVO_SOURCE_BUDGET_RATIO"] = str(budget_ratio)
    os.environ["KIVO_SOURCE_KEEP_RECENT_BLOCKS"] = str(
        keep_recent_blocks
    )


def _build_generation_args(
    args: argparse.Namespace,
    prompt: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )


def _run_generation(args: argparse.Namespace) -> dict[str, Any]:
    return source_s1._run_generation(args)


def _run_generation_safe(
    args: argparse.Namespace,
    generation_fn: Any,
) -> dict[str, Any]:
    try:
        return generation_fn(args)
    except Exception as exc:
        return {
            "status": "failed",
            "output_text": None,
            "error": (
                f"{type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()[-4000:]}"
            ),
        }


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return source_s1.load_records(path)


def _max_int(records: list[dict[str, Any]], field: str) -> int:
    values = [
        int(record[field])
        for record in records
        if isinstance(record.get(field), int)
    ]
    return max(values, default=0)


def _max_float(records: list[dict[str, Any]], field: str) -> float:
    values = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float))
    ]
    return max(values, default=0.0)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records_written": len(records),
        "max_visible_block_count": _max_int(
            records, "visible_block_count"
        ),
        "max_selected_block_count": _max_int(
            records, "selected_block_count"
        ),
        "max_theoretical_visible_block_reduction": _max_int(
            records, "theoretical_visible_block_reduction"
        ),
        "max_theoretical_visible_block_reduction_ratio": _max_float(
            records, "theoretical_visible_block_reduction_ratio"
        ),
        "mutation_applied_count": sum(
            record.get("mutation_applied") is True for record in records
        ),
        "active_routing_count": sum(
            record.get("active_routing") is True for record in records
        ),
        "runtime_behavior_changed_count": sum(
            record.get("runtime_behavior_changed") is True
            for record in records
        ),
    }


def build_report(
    args: argparse.Namespace,
    *,
    generation_fn: Any = _run_generation,
    record_loader: Any = _load_records,
) -> dict[str, Any]:
    prompts = _resolve_prompts(args)
    observation_dir = Path(args.observation_dir)
    observation_dir.mkdir(parents=True, exist_ok=True)
    prompt_results: list[dict[str, Any]] = []
    stop_early = False

    for prompt_index, prompt in enumerate(prompts):
        generation_args = _build_generation_args(args, prompt)
        _clear_source_env()
        baseline = _run_generation_safe(generation_args, generation_fn)

        observation_path = observation_dir / (
            f"prompt_{prompt_index:02d}_shadow.jsonl"
        )
        if observation_path.exists():
            observation_path.unlink()
        _set_shadow_env(
            observation_path,
            budget_ratio=args.budget_ratio,
            keep_recent_blocks=args.keep_recent_blocks,
        )
        shadow = _run_generation_safe(generation_args, generation_fn)
        records = record_loader(observation_path)
        summary = summarize_records(records)
        output_changed = bool(
            baseline["status"] == "succeeded"
            and shadow["status"] == "succeeded"
            and baseline["output_text"] != shadow["output_text"]
        )
        prompt_results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "baseline_status": baseline["status"],
                "shadow_status": shadow["status"],
                "baseline_output": baseline["output_text"],
                "shadow_output": shadow["output_text"],
                "baseline_error": baseline["error"],
                "shadow_error": shadow["error"],
                "output_changed": output_changed,
                "observation_jsonl": str(observation_path),
                **summary,
            }
        )
        if (
            not args.continue_on_error
            and (
                baseline["status"] != "succeeded"
                or shadow["status"] != "succeeded"
            )
        ):
            stop_early = True
            break

    _clear_source_env()
    total_prompts = len(prompt_results)
    baseline_success_count = sum(
        result["baseline_status"] == "succeeded"
        for result in prompt_results
    )
    shadow_success_count = sum(
        result["shadow_status"] == "succeeded"
        for result in prompt_results
    )
    total_records = sum(
        result["records_written"] for result in prompt_results
    )
    mutation_applied_count = sum(
        result["mutation_applied_count"] for result in prompt_results
    )
    active_routing_count = sum(
        result["active_routing_count"] for result in prompt_results
    )
    runtime_behavior_changed_count = sum(
        result["runtime_behavior_changed_count"]
        for result in prompt_results
    )
    max_visible_block_count = max(
        (
            result["max_visible_block_count"]
            for result in prompt_results
        ),
        default=0,
    )
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "budget_ratio": args.budget_ratio,
        "keep_recent_blocks": args.keep_recent_blocks,
        "prompts": prompts,
        "total_prompts": total_prompts,
        "baseline_success_count": baseline_success_count,
        "shadow_success_count": shadow_success_count,
        "output_changed_count": sum(
            result["output_changed"] for result in prompt_results
        ),
        "total_records": total_records,
        "max_visible_block_count": max_visible_block_count,
        "max_selected_block_count": max(
            (
                result["max_selected_block_count"]
                for result in prompt_results
            ),
            default=0,
        ),
        "max_theoretical_visible_block_reduction": max(
            (
                result["max_theoretical_visible_block_reduction"]
                for result in prompt_results
            ),
            default=0,
        ),
        "max_theoretical_visible_block_reduction_ratio": max(
            (
                result[
                    "max_theoretical_visible_block_reduction_ratio"
                ]
                for result in prompt_results
            ),
            default=0.0,
        ),
        "mutation_applied_count": mutation_applied_count,
        "active_routing_count": active_routing_count,
        "runtime_behavior_changed_count": runtime_behavior_changed_count,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s2_shadow_passed": bool(
            total_prompts > 0
            and baseline_success_count == total_prompts
            and shadow_success_count == total_prompts
            and total_records > 0
            and mutation_applied_count == 0
            and active_routing_count == 0
            and runtime_behavior_changed_count == 0
            and max_visible_block_count >= 2
        ),
        "stop_early": stop_early,
        "prompt_results": prompt_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S2.0 Block Visibility Shadow",
        "",
        "## Summary",
        "",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Baseline successes: `{report['baseline_success_count']}`",
        f"- Shadow successes: `{report['shadow_success_count']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        f"- Total records: `{report['total_records']}`",
        (
            "- Maximum visible block count: "
            f"`{report['max_visible_block_count']}`"
        ),
        (
            "- Maximum selected block count: "
            f"`{report['max_selected_block_count']}`"
        ),
        (
            "- Maximum theoretical visible-block reduction: "
            f"`{report['max_theoretical_visible_block_reduction']}`"
        ),
        (
            "- Maximum theoretical visible-block reduction ratio: "
            f"`{report['max_theoretical_visible_block_reduction_ratio']}`"
        ),
        f"- Mutation applied count: `{report['mutation_applied_count']}`",
        f"- Active routing count: `{report['active_routing_count']}`",
        (
            "- Runtime behavior changed count: "
            f"`{report['runtime_behavior_changed_count']}`"
        ),
        "- Measured runtime reduction: `false`",
        "- Selected-attention claim allowed: `false`",
        "- Performance claim allowed: `false`",
        f"- S2 shadow passed: `{report['s2_shadow_passed']}`",
        "",
        "The reduction fields describe visible blocks excluded by a shadow "
        "policy. They are not measured KV memory savings.",
        "",
        "## Prompt Results",
        "",
    ]
    for result in report["prompt_results"]:
        lines.extend(
            [
                f"### Prompt {result['prompt_index']}",
                "",
                f"- Baseline status: `{result['baseline_status']}`",
                f"- Shadow status: `{result['shadow_status']}`",
                f"- Output changed: `{result['output_changed']}`",
                f"- Records written: `{result['records_written']}`",
                (
                    "- Maximum visible blocks: "
                    f"`{result['max_visible_block_count']}`"
                ),
                (
                    "- Maximum selected blocks: "
                    f"`{result['max_selected_block_count']}`"
                ),
                (
                    "- Maximum theoretical reduction ratio: "
                    f"`{result['max_theoretical_visible_block_reduction_ratio']}`"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "total_prompts": report["total_prompts"],
                "baseline_success_count": report[
                    "baseline_success_count"
                ],
                "shadow_success_count": report["shadow_success_count"],
                "max_visible_block_count": report[
                    "max_visible_block_count"
                ],
                "s2_shadow_passed": report["s2_shadow_passed"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["s2_shadow_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
