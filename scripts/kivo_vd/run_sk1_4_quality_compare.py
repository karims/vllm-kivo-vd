#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny GPU-pod quality comparison for baseline vs Kivo modes."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kivo_vd.run_sk1_3c_live_sketch_smoke import (  # noqa: E402
    patched_environ,
    resolve_model_reference,
    summarize_sketch_counters,
)
from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (  # noqa: E402
    _build_llm_kwargs,
    _load_exported_counters,
)

BASELINE_MODE = "baseline"
RECENT_ONLY_MODE = "recent_only"
RANDOM_PROJECTION_MODE = "random_projection"
MODE_ORDER = [BASELINE_MODE, RECENT_ONLY_MODE, RANDOM_PROJECTION_MODE]

KIVO_ENV_KEYS = [
    "KIVO_KV_SKETCH_ENABLE",
    "KIVO_KV_SKETCH_BACKEND",
    "KIVO_KV_SKETCH_DIM",
    "KIVO_KV_SKETCH_SEED",
    "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE",
    "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION",
    "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY",
    "KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS",
    "KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS",
    "KIVO_KV_DEMOTION_TRANSPORT_ENABLE",
    "KIVO_KV_DEMOTION_TRANSPORT_ACTION",
    "KIVO_KV_CORE_DEMOTION_ENABLE",
    "KIVO_KV_CORE_DEMOTION_ACTION",
    "KIVO_KV_OWNERSHIP_REMOVE_ENABLE",
    "KIVO_KV_OWNERSHIP_REMOVE_ACTION",
    "KIVO_KV_FREE_TO_POOL_ENABLE",
    "KIVO_KV_FREE_TO_POOL_ACTION",
    "KIVO_KV_DEMOTION_COUNTERS_ENABLE",
    "KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Sk-1.4 tiny quality comparison on a GPU pod."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="/tmp/sk1_4_quality_compare.json")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--keep-recent-blocks", type=int, default=2)
    parser.add_argument("--max-full-blocks", type=int, default=2)
    parser.add_argument("--sketch-topk", type=int, default=2)
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--sketch-seed", type=int, default=123)
    parser.add_argument("--prompt-repeats", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def build_quality_prompts(*, repeats: int) -> list[dict[str, str]]:
    factual_context = " ".join(
        [
            (
                "Project Orion launched in 2018, moved to Toronto in 2020, "
                "and its lead researcher is named Maya Chen."
            )
        ]
        * repeats
    )
    code_context = " ".join(
        [
            (
                "Function compute_total iterates over invoice items, adds "
                "tax_rate, and returns cents as an integer for API callers."
            )
        ]
        * repeats
    )
    summary_context = " ".join(
        [
            (
                "The service ingests long documents, extracts entities, "
                "builds section summaries, and returns a compact report."
            )
        ]
        * repeats
    )
    instruction_context = " ".join(
        [
            (
                "You are helping a user clean a deployment checklist: "
                "verify secrets, restart workers, confirm health checks, "
                "and record the rollout timestamp."
            )
        ]
        * repeats
    )
    return [
        {
            "prompt_name": "factual_recall",
            "prompt_text": (
                factual_context
                + "\nQuestion: Who leads Project Orion and when did it move "
                + "to Toronto?"
            ),
        },
        {
            "prompt_name": "code_context",
            "prompt_text": (
                code_context
                + "\nQuestion: Explain what compute_total returns and why "
                + "the result is stored in cents."
            ),
        },
        {
            "prompt_name": "summarization",
            "prompt_text": (
                summary_context
                + "\nTask: Summarize the pipeline in two short sentences."
            ),
        },
        {
            "prompt_name": "instruction_following",
            "prompt_text": (
                instruction_context
                + "\nTask: Produce a numbered checklist for the deployment "
                + "steps in order."
            ),
        },
    ]


def _mode_counter_export_file(output_path: str | Path, mode: str) -> str:
    output = Path(output_path)
    return str(output.with_name(f"{output.stem}.{mode}.counters.json"))


def build_mode_env(
    mode: str,
    *,
    args: argparse.Namespace,
    counter_export_file: str,
) -> dict[str, str]:
    base = {
        "KIVO_KV_DEMOTION_COUNTERS_ENABLE": "1",
        "KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE": counter_export_file,
    }
    if args.local_files_only:
        base["HF_HUB_OFFLINE"] = "1"
        base["TRANSFORMERS_OFFLINE"] = "1"
    if mode == BASELINE_MODE:
        return base
    base.update(
        {
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE": "1",
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION": "apply_block_table_only",
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY": RECENT_ONLY_MODE,
            "KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS": str(
                args.keep_recent_blocks
            ),
            "KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS": str(
                args.max_full_blocks
            ),
            "KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_TOPK": str(args.sketch_topk),
            "KIVO_KV_DEMOTION_TRANSPORT_ENABLE": "1",
            "KIVO_KV_DEMOTION_TRANSPORT_ACTION": "apply_core_mark_demoted",
            "KIVO_KV_CORE_DEMOTION_ENABLE": "1",
            "KIVO_KV_CORE_DEMOTION_ACTION": "mark_demoted_only",
            "KIVO_KV_OWNERSHIP_REMOVE_ENABLE": "1",
            "KIVO_KV_OWNERSHIP_REMOVE_ACTION": "remove_marked_demoted_only",
            "KIVO_KV_FREE_TO_POOL_ENABLE": "1",
            "KIVO_KV_FREE_TO_POOL_ACTION": "free_removed_demoted_only",
        }
    )
    if mode == RANDOM_PROJECTION_MODE:
        base.update(
            {
                "KIVO_KV_SKETCH_ENABLE": "1",
                "KIVO_KV_SKETCH_BACKEND": "random_projection",
                "KIVO_KV_SKETCH_DIM": str(args.sketch_dim),
                "KIVO_KV_SKETCH_SEED": str(args.sketch_seed),
            }
        )
    return base


def extract_output_text(output: Any) -> str:
    candidates = getattr(output, "outputs", None)
    if not candidates:
        return ""
    return str(candidates[0].text)


def extract_prompt_token_count(output: Any) -> int | None:
    token_ids = getattr(output, "prompt_token_ids", None)
    if token_ids is None:
        return None
    return len(token_ids)


def extract_output_token_count(output: Any, fallback: int) -> int | None:
    candidates = getattr(output, "outputs", None)
    if not candidates:
        return None
    token_ids = getattr(candidates[0], "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return fallback


def build_mode_summary(mode: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    success_results = [item for item in results if item["generation_success"]]
    latencies = [float(item["latency_seconds"]) for item in success_results]
    total_freed = sum(
        int(item["counter_summary"].get("freed_after_sketch_blocks_total", 0) or 0)
        for item in results
    )
    invariants_clean = all(
        bool(item["counter_summary"].get("invariants_clean", True))
        for item in results
    )
    sketch_success_count = sum(
        1
        for item in results
        if int(item["counter_summary"].get("sketch_build_succeeded", 0) or 0) > 0
    )
    warnings: list[str] = []
    if mode == RANDOM_PROJECTION_MODE and sketch_success_count == 0:
        warnings.append("random_projection_zero_sketch_success")
    if not invariants_clean:
        warnings.append("invariants_not_clean")
    return {
        "mode": mode,
        "prompt_count": len(results),
        "success_count": len(success_results),
        "average_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "total_freed_after_sketch_blocks": total_freed,
        "invariants_clean": invariants_clean,
        "random_projection_sketch_success_count": sketch_success_count,
        "warnings": warnings,
    }


def build_overall_summary(mode_reports: list[dict[str, Any]]) -> dict[str, Any]:
    per_mode = {item["mode"]: item["summary"] for item in mode_reports}
    warnings: list[str] = []
    random_projection = per_mode.get(RANDOM_PROJECTION_MODE, {})
    if random_projection.get("random_projection_sketch_success_count", 0) == 0:
        warnings.append("random_projection_zero_sketch_success")
    if not all(item.get("invariants_clean", True) for item in per_mode.values()):
        warnings.append("one_or_more_modes_have_invariant_failures")
    return {
        "per_mode_success_count": {
            mode: per_mode.get(mode, {}).get("success_count", 0)
            for mode in MODE_ORDER
        },
        "per_mode_average_latency_seconds": {
            mode: per_mode.get(mode, {}).get("average_latency_seconds")
            for mode in MODE_ORDER
        },
        "per_mode_total_freed_after_sketch_blocks": {
            mode: per_mode.get(mode, {}).get("total_freed_after_sketch_blocks", 0)
            for mode in MODE_ORDER
        },
        "per_mode_invariants_clean": {
            mode: per_mode.get(mode, {}).get("invariants_clean", True)
            for mode in MODE_ORDER
        },
        "random_projection_sketch_success_count": random_projection.get(
            "random_projection_sketch_success_count", 0
        ),
        "warnings": warnings,
    }


def _run_prompt(
    llm: Any,
    *,
    prompt_name: str,
    prompt_text: str,
    args: argparse.Namespace,
    counter_export_file: str,
    reset_counters_fn: Any,
    get_counters_fn: Any,
) -> dict[str, Any]:
    export_path = Path(counter_export_file)
    if export_path.exists():
        export_path.unlink()
    reset_counters_fn()
    try:
        from vllm import SamplingParams

        started = time.perf_counter()
        outputs = llm.generate(
            [prompt_text],
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
            ),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        first = outputs[0] if outputs else None
        exported_counters, file_found, export_pid = _load_exported_counters(
            counter_export_file
        )
        parent_counters = get_counters_fn()
        counters = (
            exported_counters if exported_counters is not None else parent_counters
        )
        return {
            "prompt_name": prompt_name,
            "generation_success": True,
            "prompt_token_count": (
                extract_prompt_token_count(first) if first is not None else None
            ),
            "output_text": extract_output_text(first) if first is not None else "",
            "output_token_count": (
                extract_output_token_count(first, args.max_tokens)
                if first is not None
                else None
            ),
            "latency_seconds": elapsed,
            "counter_export_file_found": file_found,
            "counter_export_pid": export_pid,
            "counters": counters,
            "counter_summary": summarize_sketch_counters(counters),
            "error": None,
        }
    except Exception as exc:
        exported_counters, file_found, export_pid = _load_exported_counters(
            counter_export_file
        )
        counters = exported_counters if exported_counters is not None else {}
        return {
            "prompt_name": prompt_name,
            "generation_success": False,
            "prompt_token_count": None,
            "output_text": None,
            "output_token_count": None,
            "latency_seconds": None,
            "counter_export_file_found": file_found,
            "counter_export_pid": export_pid,
            "counters": counters,
            "counter_summary": summarize_sketch_counters(counters),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_mode(
    mode: str,
    *,
    args: argparse.Namespace,
    resolved_model: str,
    model_is_local: bool,
    cached_model_candidates: list[str],
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
    counter_export_file = _mode_counter_export_file(args.output, mode)
    mode_env = build_mode_env(mode, args=args, counter_export_file=counter_export_file)
    llm = None
    try:
        with patched_environ(mode_env):
            from vllm import LLM
            from vllm.v1.core.kivo_demotion_counters import (
                get_kivo_demotion_counters_snapshot,
                reset_kivo_demotion_counters,
            )

            llm_kwargs = _build_llm_kwargs(args)
            llm_kwargs["model"] = resolved_model
            llm = LLM(**llm_kwargs)
            results = [
                _run_prompt(
                    llm,
                    prompt_name=item["prompt_name"],
                    prompt_text=item["prompt_text"],
                    args=args,
                    counter_export_file=counter_export_file,
                    reset_counters_fn=reset_kivo_demotion_counters,
                    get_counters_fn=get_kivo_demotion_counters_snapshot,
                )
                for item in prompts
            ]
    finally:
        if llm is not None:
            del llm
        gc.collect()
    return {
        "mode": mode,
        "model": args.model,
        "resolved_model": resolved_model,
        "model_is_local": model_is_local,
        "cached_model_candidates": cached_model_candidates,
        "env_flags": mode_env,
        "counter_export_file": counter_export_file,
        "results": results,
        "summary": build_mode_summary(mode, results),
    }


def run_quality_compare(args: argparse.Namespace) -> dict[str, Any]:
    resolved_model, model_is_local, cached_model_candidates = (
        resolve_model_reference(
            args.model,
            local_files_only=args.local_files_only,
        )
    )
    prompts = build_quality_prompts(repeats=args.prompt_repeats)
    mode_reports = [
        run_mode(
            mode,
            args=args,
            resolved_model=resolved_model,
            model_is_local=model_is_local,
            cached_model_candidates=cached_model_candidates,
            prompts=prompts,
        )
        for mode in MODE_ORDER
    ]
    return {
        "phase": "Sk-1.4",
        "model": args.model,
        "resolved_model": resolved_model,
        "model_is_local": model_is_local,
        "cached_model_candidates": cached_model_candidates,
        "settings": {
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "keep_recent_blocks": args.keep_recent_blocks,
            "max_full_blocks": args.max_full_blocks,
            "sketch_topk": args.sketch_topk,
            "sketch_dim": args.sketch_dim,
            "sketch_seed": args.sketch_seed,
            "prompt_repeats": args.prompt_repeats,
            "seed": args.seed,
        },
        "prompt_names": [item["prompt_name"] for item in prompts],
        "mode_reports": mode_reports,
        "summary": build_overall_summary(mode_reports),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_quality_compare(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
