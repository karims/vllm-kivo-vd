#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny vLLM generation and report Kivo demotion transport counters."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny vLLM generation with Kivo demotion counters."
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "Kivo transport counter probe prompt one.",
            "Kivo transport counter probe prompt two.",
        ],
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def build_summary(
    *,
    generation_success: bool,
    prompt_count: int,
    counters: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    transport_observed = (
        int(counters.get("scheduler_envelopes_received", 0) or 0) > 0
        or int(counters.get("core_commands_attempted", 0) or 0) > 0
        or int(counters.get("manager_mark_demoted_attempted", 0) or 0) > 0
    )
    return {
        "phase": "S5.18",
        "generation_success": generation_success,
        "prompt_count": prompt_count,
        "counters": counters,
        "transport_observed": transport_observed,
        "transport_claim_allowed": transport_observed,
        "memory_claim_allowed": False,
        "free_to_pool_claim_allowed": False,
        "error": error,
    }


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "enforce_eager": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.device != "auto":
        llm_kwargs["device"] = args.device
    return llm_kwargs


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.v1.core.kivo_demotion_counters import (
        get_kivo_demotion_counters_snapshot,
        reset_kivo_demotion_counters,
    )

    reset_kivo_demotion_counters()
    llm = None
    try:
        llm = LLM(**_build_llm_kwargs(args))
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        llm.generate(args.prompts, sampling_params, use_tqdm=False)
        return build_summary(
            generation_success=True,
            prompt_count=len(args.prompts),
            counters=get_kivo_demotion_counters_snapshot(),
        )
    except Exception as exc:
        return build_summary(
            generation_success=False,
            prompt_count=len(args.prompts),
            counters=get_kivo_demotion_counters_snapshot(),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_generation(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["generation_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
