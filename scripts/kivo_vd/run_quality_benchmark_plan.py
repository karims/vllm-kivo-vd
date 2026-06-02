#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Create offline Kivo-VD quality benchmark plans and synthetic prompts."""

import argparse
import json
from pathlib import Path
from typing import Any


def generate_needle_prompt(
    *,
    num_filler_repeats: int = 32,
    needle: str = "BLUE ORCHID",
    query: str = "What is the secret phrase?",
) -> tuple[str, str]:
    if num_filler_repeats < 0:
        raise ValueError("num_filler_repeats must be non-negative.")

    filler = (
        "The archive contains routine notes about weather, schedules, "
        "inventory, and ordinary project updates."
    )
    parts = [
        f"Important record: the secret phrase is {needle}.",
        *[filler for _ in range(num_filler_repeats)],
        query,
    ]
    return "\n\n".join(parts), needle


def _dry_run_equality_plan(model: str) -> dict[str, Any]:
    command = [
        ".venv/bin/python",
        "scripts/kivo_vd/run_vllm_kivo_dry_run.py",
        "--model",
        model,
        "--max-tokens",
        "16",
        "--enable-kivo-vd",
    ]
    return {
        "benchmark": "dry_run_equality",
        "model": model,
        "status": "plan_only",
        "goal": "Verify Kivo dry-run output matches baseline greedy output.",
        "command": command,
        "success_criteria": [
            "baseline inference completes",
            "Kivo-enabled dry-run inference completes",
            "outputs_match is true",
            "dry-run events export successfully when observer is accessible",
        ],
    }


def _needle_synthetic_plan(
    *,
    model: str,
    num_filler_repeats: int,
    needle: str,
    query: str,
) -> dict[str, Any]:
    prompt, expected_answer = generate_needle_prompt(
        num_filler_repeats=num_filler_repeats,
        needle=needle,
        query=query,
    )
    return {
        "benchmark": "needle_synthetic",
        "model": model,
        "status": "prompt_generation_only",
        "prompt": prompt,
        "expected_answer": expected_answer,
        "num_filler_repeats": num_filler_repeats,
        "success_criteria": [
            "future baseline model answers with the needle phrase",
            "future Kivo candidate-attention variant preserves the answer",
            "quality is evaluated before memory-reduction claims",
        ],
    }


def _perplexity_stub_plan(model: str) -> dict[str, Any]:
    return {
        "benchmark": "perplexity_stub",
        "model": model,
        "status": "plan_only",
        "dataset": "WikiText-style text, to be wired in a future phase",
        "metric": "perplexity_delta_vs_baseline",
        "success_criteria": [
            "baseline perplexity is measured",
            "future Kivo variant perplexity delta is bounded",
            "same tokenizer/model/config are used for both paths",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Kivo-VD quality benchmark scaffold output."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--benchmark",
        choices=["dry_run_equality", "needle_synthetic", "perplexity_stub"],
        default="needle_synthetic",
    )
    parser.add_argument(
        "--output", default="outputs/kivo_vd/quality_benchmark_plan.json"
    )
    parser.add_argument("--num-filler-repeats", type=int, default=32)
    parser.add_argument("--needle", default="BLUE ORCHID")
    parser.add_argument("--query", default="What is the secret phrase?")
    return parser.parse_args()


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.benchmark == "dry_run_equality":
        return _dry_run_equality_plan(args.model)
    if args.benchmark == "perplexity_stub":
        return _perplexity_stub_plan(args.model)
    return _needle_synthetic_plan(
        model=args.model,
        num_filler_repeats=args.num_filler_repeats,
        needle=args.needle,
        query=args.query,
    )


def main() -> int:
    args = _parse_args()
    plan = build_plan(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(plan, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
