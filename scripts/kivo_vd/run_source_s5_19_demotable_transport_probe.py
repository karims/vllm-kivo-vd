#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a long-prompt probe to observe Kivo demotion transport counters."""

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

DEFAULT_SENTENCE = (
    "This is a cached context sentence about retrieval, memory, and attention."
)


def build_long_context_prompt(*, repeats: int = 200) -> str:
    long_context = " ".join([DEFAULT_SENTENCE] * repeats)
    return (
        long_context
        + "\nQuestion: summarize the repeated context in one sentence."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a long-prompt vLLM probe for demotable Kivo transport."
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-repeats", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def build_prompts(*, repeats: int, num_prompts: int) -> list[str]:
    prompt = build_long_context_prompt(repeats=repeats)
    return [prompt for _ in range(max(1, num_prompts))]


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


def build_summary(
    *,
    generation_success: bool,
    prompt_count: int,
    prompt_char_lengths: list[int],
    prompt_token_lengths: list[int | None],
    counters: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    worker_envelope_observed = int(counters.get("worker_envelopes_built", 0) or 0) > 0
    scheduler_envelope_observed = (
        int(counters.get("scheduler_envelopes_received", 0) or 0) > 0
    )
    core_command_observed = int(counters.get("core_commands_attempted", 0) or 0) > 0
    manager_mark_demoted_observed = (
        int(counters.get("manager_mark_demoted_attempted", 0) or 0) > 0
    )
    transport_observed = (
        worker_envelope_observed
        or scheduler_envelope_observed
        or core_command_observed
        or manager_mark_demoted_observed
    )
    return {
        "phase": "S5.19",
        "generation_success": generation_success,
        "prompt_count": prompt_count,
        "prompt_char_lengths": prompt_char_lengths,
        "prompt_token_lengths": prompt_token_lengths,
        "counters": counters,
        "transport_observed": transport_observed,
        "worker_envelope_observed": worker_envelope_observed,
        "scheduler_envelope_observed": scheduler_envelope_observed,
        "core_command_observed": core_command_observed,
        "manager_mark_demoted_observed": manager_mark_demoted_observed,
        "req_to_blocks_removed": int(counters.get("req_to_blocks_removed", 0) or 0),
        "free_to_pool_calls": int(counters.get("free_to_pool_calls", 0) or 0),
        "memory_claim_allowed": False,
        "free_to_pool_claim_allowed": False,
        "error": error,
    }


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.v1.core.kivo_demotion_counters import (
        get_kivo_demotion_counters_snapshot,
        reset_kivo_demotion_counters,
    )

    prompts = build_prompts(repeats=args.prompt_repeats, num_prompts=args.num_prompts)
    prompt_char_lengths = [len(prompt) for prompt in prompts]
    llm = None
    reset_kivo_demotion_counters()
    try:
        llm = LLM(**_build_llm_kwargs(args))
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        prompt_token_lengths = [
            len(getattr(output, "prompt_token_ids", None) or [])
            if getattr(output, "prompt_token_ids", None) is not None
            else None
            for output in outputs
        ]
        return build_summary(
            generation_success=True,
            prompt_count=len(prompts),
            prompt_char_lengths=prompt_char_lengths,
            prompt_token_lengths=prompt_token_lengths,
            counters=get_kivo_demotion_counters_snapshot(),
        )
    except Exception as exc:
        return build_summary(
            generation_success=False,
            prompt_count=len(prompts),
            prompt_char_lengths=prompt_char_lengths,
            prompt_token_lengths=[None] * len(prompts),
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
