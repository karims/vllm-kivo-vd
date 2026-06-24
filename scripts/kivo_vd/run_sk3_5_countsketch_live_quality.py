#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Integrated CountSketch live-retention quality/block-savings verdict runner."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kivo_vd.run_sk1_3c_live_sketch_smoke import (  # noqa: E402
    patched_environ,
    resolve_model_reference,
)
from scripts.kivo_vd.run_sk1_4_quality_compare import (  # noqa: E402
    extract_output_text,
    extract_output_token_count,
    extract_prompt_token_count,
)
from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (  # noqa: E402
    _build_llm_kwargs,
    _load_exported_counters,
)

BASELINE_MODE = "baseline"
SKETCH_BUILD_ONLY_MODE = "sketch_build_only_countsketch"
SAFE_MODE = "countsketch_span_safe"
DEFAULT_MODE = "countsketch_span_default"
AGGRESSIVE_MODE = "countsketch_span_aggressive"
COUNTSKETCH_POLICY = "countsketch_score_store_span_topk"
SCORING_SOURCE = "countsketch_score_store"

MODE_PRESETS: dict[str, dict[str, Any]] = {
    BASELINE_MODE: {},
    SKETCH_BUILD_ONLY_MODE: {
        "action": "plan_only",
        "keep_recent_blocks": 16,
        "sketch_topk": 8,
        "span_radius": 1,
        "max_full_blocks": 32,
        "free_enabled": False,
    },
    SAFE_MODE: {
        "action": "apply_block_table_only",
        "keep_recent_blocks": 24,
        "sketch_topk": 8,
        "span_radius": 1,
        "max_full_blocks": 40,
        "free_enabled": True,
    },
    DEFAULT_MODE: {
        "action": "apply_block_table_only",
        "keep_recent_blocks": 16,
        "sketch_topk": 8,
        "span_radius": 1,
        "max_full_blocks": 32,
        "free_enabled": True,
    },
    AGGRESSIVE_MODE: {
        "action": "apply_block_table_only",
        "keep_recent_blocks": 12,
        "sketch_topk": 6,
        "span_radius": 1,
        "max_full_blocks": 24,
        "free_enabled": True,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sk-3.5 CountSketch live quality/block-savings verdict."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="/tmp/sk3_5_countsketch_verdict.json")
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--counter-export-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--prompt-repeats", type=int, default=48)
    parser.add_argument("--sketch-backend", default="countsketch")
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--sketch-seed", type=int, default=123)
    parser.add_argument(
        "--modes",
        default=",".join(
            [
                BASELINE_MODE,
                SKETCH_BUILD_ONLY_MODE,
                SAFE_MODE,
                DEFAULT_MODE,
                AGGRESSIVE_MODE,
            ]
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def build_verdict_prompts(*, repeats: int) -> list[dict[str, Any]]:
    factual = " ".join(
        [
            (
                "Project Orion was led by Maya Chen and moved to Toronto in "
                "2020. This fact is important for the final question."
            )
        ]
        + [
            (
                "Distractor notes discuss launch timelines, archive formats, "
                "city permits, meeting agendas, and evaluation scaffolding."
            )
        ]
        * repeats
    )
    code = " ".join(
        [
            (
                "compute_total returns invoice total including tax in cents. "
                "The integer cents value avoids floating point drift."
            )
        ]
        + [
            (
                "Repository context mentions serializers, cache retries, HTTP "
                "handlers, schema migrations, deployment flags, and tests."
            )
        ]
        * repeats
    )
    summary = " ".join(
        [
            (
                "The system ingests documents, extracts entities, builds "
                "section summaries, and returns compact reports."
            )
        ]
        * repeats
    )
    instructions = " ".join(
        [
            (
                "Deployment checklist: verify secrets, restart workers, check "
                "health endpoints, watch logs, and record the rollout time."
            )
        ]
        * repeats
    )
    return [
        {
            "prompt_name": "factual_recall",
            "prompt_text": (
                factual
                + "\nQuestion: Who led Project Orion, where did it move, "
                + "and in what year?"
            ),
            "expected_terms": ["maya chen", "toronto", "2020"],
        },
        {
            "prompt_name": "code_context",
            "prompt_text": (
                code
                + "\nQuestion: What does compute_total return and why are "
                + "cents used?"
            ),
            "expected_terms": ["compute_total", "invoice", "tax", "cents"],
        },
        {
            "prompt_name": "summarization",
            "prompt_text": summary + "\nTask: Summarize the pipeline briefly.",
            "expected_terms": [],
        },
        {
            "prompt_name": "instruction_following",
            "prompt_text": instructions + "\nTask: Produce a numbered checklist.",
            "expected_terms": [],
        },
    ]


def expected_term_report(output_text: str | None, expected_terms: list[str]) -> dict[str, Any]:
    text = (output_text or "").lower()
    found = [term for term in expected_terms if term.lower() in text]
    missing = [term for term in expected_terms if term.lower() not in text]
    return {
        "expected_terms": expected_terms,
        "expected_terms_found": found,
        "expected_terms_missing": missing,
        "contains_expected_terms": len(missing) == 0,
    }


def degeneration_report(output_text: str | None) -> dict[str, Any]:
    text = output_text or ""
    words = re.findall(r"\w+", text.lower())
    adjacent_repeated = sum(
        1 for left, right in zip(words, words[1:]) if left == right
    )
    chars = list(text)
    char_count = max(1, len(chars))
    digit_ratio = sum(ch.isdigit() for ch in chars) / char_count
    non_ascii_ratio = sum(ord(ch) > 127 for ch in chars) / char_count
    junk_ratio = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?-'\"()[]") for ch in chars) / char_count
    repeated_word_limit = max(6, len(words) // 4)
    dominant_word_count = max(Counter(words).values()) if words else 0
    degeneration_detected = (
        len(words) < 3
        or adjacent_repeated > repeated_word_limit
        or (len(words) >= 6 and dominant_word_count >= len(words) * 0.75)
        or digit_ratio > 0.45
        or non_ascii_ratio > 0.20
        or junk_ratio > 0.30
    )
    return {
        "degeneration_detected": degeneration_detected,
        "adjacent_repeated_words": adjacent_repeated,
        "dominant_word_count": dominant_word_count,
        "digit_ratio": digit_ratio,
        "non_ascii_ratio": non_ascii_ratio,
        "junk_ratio": junk_ratio,
    }


def retention_ratio_from_counts(visible_after: int, visible_before: int) -> float | None:
    if visible_before <= 0:
        return None
    return float(visible_after / visible_before)


def span_diagnostics(block_ids: list[int]) -> dict[str, Any]:
    if not block_ids:
        return {
            "contiguous_span_count": 0,
            "max_gap_between_kept_blocks": 0,
            "kept_blocks_contiguous": True,
        }
    ordered = list(block_ids)
    spans = 1
    max_gap = 0
    for previous, current in zip(ordered, ordered[1:]):
        gap = max(0, current - previous - 1)
        if gap > 0:
            spans += 1
        max_gap = max(max_gap, gap)
    return {
        "contiguous_span_count": spans,
        "max_gap_between_kept_blocks": max_gap,
        "kept_blocks_contiguous": spans <= 1,
    }


def _mode_env(
    mode: str,
    *,
    args: argparse.Namespace,
    counter_file: str,
    trace_file: str,
) -> dict[str, str]:
    base = {
        "KIVO_KV_DEMOTION_COUNTERS_ENABLE": "1",
        "KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE": counter_file,
    }
    if args.local_files_only:
        base["HF_HUB_OFFLINE"] = "1"
        base["TRANSFORMERS_OFFLINE"] = "1"
    if mode == BASELINE_MODE:
        return base

    preset = MODE_PRESETS[mode]
    base.update(
        {
            "KIVO_KV_SKETCH_ENABLE": "1",
            "KIVO_KV_SKETCH_BACKEND": args.sketch_backend,
            "KIVO_KV_SKETCH_DIM": str(args.sketch_dim),
            "KIVO_KV_SKETCH_SEED": str(args.sketch_seed),
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE": "1",
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION": preset["action"],
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY": COUNTSKETCH_POLICY,
            "KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS": str(
                preset["keep_recent_blocks"]
            ),
            "KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_TOPK": str(preset["sketch_topk"]),
            "KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_SPAN_RADIUS": str(
                preset["span_radius"]
            ),
            "KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS": str(
                preset["max_full_blocks"]
            ),
            "KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_RETAINED_BLOCKS": "1",
            "KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_MAX_BLOCKS": "64",
            "KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_FILE": trace_file,
        }
    )
    if preset.get("free_enabled"):
        base.update(
            {
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
    return base


def _paths_for_mode_prompt(
    *,
    args: argparse.Namespace,
    mode: str,
    prompt_name: str,
) -> tuple[str, str]:
    output = Path(args.output)
    counter_dir = (
        Path(args.counter_export_dir)
        if args.counter_export_dir
        else output.with_suffix("").with_name(f"{output.stem}.counters")
    )
    trace_dir = (
        Path(args.trace_output)
        if args.trace_output
        else output.with_suffix("").with_name(f"{output.stem}.traces")
    )
    counter_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    return (
        str(counter_dir / f"{mode}.{prompt_name}.counters.json"),
        str(trace_dir / f"{mode}.{prompt_name}.trace.jsonl"),
    )


def _load_trace_rows(path: str, *, mode: str, prompt_name: str) -> list[dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    rows = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["mode"] = mode
        row["prompt_name"] = prompt_name
        rows.append(row)
    return rows


def _summarize_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["retention_ratio"])
        for row in rows
        if row.get("retention_ratio") is not None
    ]
    visible_before = [int(row.get("visible_before_count", 0) or 0) for row in rows]
    visible_after = [int(row.get("visible_after_count", 0) or 0) for row in rows]
    candidate_counts = [
        int(row.get("candidate_demote_count", 0) or 0) for row in rows
    ]
    return {
        "visible_before_count_max": max(visible_before) if visible_before else 0,
        "visible_after_count_min": min(visible_after) if visible_after else 0,
        "visible_before_count_last": visible_before[-1] if visible_before else 0,
        "visible_after_count_last": visible_after[-1] if visible_after else 0,
        "candidate_demote_count_total": sum(candidate_counts),
        "retention_ratio_min": min(ratios) if ratios else None,
        "retention_ratio_avg": sum(ratios) / len(ratios) if ratios else None,
        "retention_ratio_last": ratios[-1] if ratios else None,
        "retained_block_fraction_avg": sum(ratios) / len(ratios) if ratios else None,
    }


def _mode_counter_summary(counters: dict[str, Any] | None) -> dict[str, Any]:
    counters = counters or {}
    invariant_fields = {
        "ownership_remove_invariant_failed": int(
            counters.get("ownership_remove_invariant_failed", 0) or 0
        ),
        "free_to_pool_double_free_prevented": int(
            counters.get("free_to_pool_double_free_prevented", 0) or 0
        ),
        "block_pool_free_accounting_rejected": int(
            counters.get("block_pool_free_accounting_rejected", 0) or 0
        ),
    }
    return {
        "sketch_backend": counters.get("sketch_backend"),
        "last_policy": counters.get("last_policy"),
        "scoring_source": counters.get("last_scoring_source"),
        "sketch_build_attempted": int(counters.get("sketch_build_attempted", 0) or 0),
        "sketch_build_succeeded": int(counters.get("sketch_build_succeeded", 0) or 0),
        "sketch_build_failed": int(counters.get("sketch_build_failed", 0) or 0),
        "sketched_blocks_total": int(counters.get("sketched_blocks_total", 0) or 0),
        "sketch_bytes_total": int(counters.get("sketch_bytes_total", 0) or 0),
        "freed_after_sketch_blocks_total": int(
            counters.get("freed_after_sketch_blocks_total", 0) or 0
        ),
        "blocks_freed_total": int(counters.get("free_to_pool_blocks", 0) or 0),
        "last_sketch_span_anchor_ids_sample": list(
            counters.get("last_sketch_span_anchor_ids_sample", ()) or ()
        ),
        "last_sketch_span_keep_ids_sample": list(
            counters.get("last_sketch_span_keep_ids_sample", ()) or ()
        ),
        "last_contiguous_span_count": int(
            counters.get("last_contiguous_span_count", 0) or 0
        ),
        "last_max_gap_between_kept_blocks": int(
            counters.get("last_max_gap_between_kept_blocks", 0) or 0
        ),
        "invariant_counters": invariant_fields,
        "invariants_clean": all(value == 0 for value in invariant_fields.values()),
    }


def _run_prompt(
    llm: Any,
    *,
    mode: str,
    prompt: dict[str, Any],
    args: argparse.Namespace,
    env_values: dict[str, str],
    counter_file: str,
    trace_file: str,
) -> dict[str, Any]:
    for path in (Path(counter_file), Path(trace_file)):
        if path.exists():
            path.unlink()
    from vllm import SamplingParams
    from vllm.v1.core.kivo_demotion_counters import (
        get_kivo_demotion_counters_snapshot,
        reset_kivo_demotion_counters,
    )

    reset_kivo_demotion_counters()
    os.environ["KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE"] = counter_file
    if mode != BASELINE_MODE:
        os.environ["KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_FILE"] = trace_file
    started = time.perf_counter()
    try:
        outputs = llm.generate(
            [prompt["prompt_text"]],
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
            ),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        first = outputs[0] if outputs else None
        exported, found, pid = _load_exported_counters(counter_file)
        counters = exported if exported is not None else get_kivo_demotion_counters_snapshot()
        output_text = extract_output_text(first) if first is not None else ""
        terms = expected_term_report(output_text, prompt["expected_terms"])
        degeneration = degeneration_report(output_text)
        trace_rows = _load_trace_rows(trace_file, mode=mode, prompt_name=prompt["prompt_name"])
        return {
            "prompt_name": prompt["prompt_name"],
            "success": True,
            "output_text": output_text,
            "output_length": len(output_text),
            "output_token_count": (
                extract_output_token_count(first, args.max_tokens)
                if first is not None
                else None
            ),
            "prompt_token_count": (
                extract_prompt_token_count(first) if first is not None else None
            ),
            "latency_seconds": elapsed,
            **terms,
            **degeneration,
            "counter_export_file_found": found,
            "counter_export_pid": pid,
            "counter_summary": _mode_counter_summary(counters),
            "trace_file": trace_file,
            "trace_rows": trace_rows,
            "trace_summary": _summarize_trace(trace_rows),
            "env_flags": env_values,
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        exported, found, pid = _load_exported_counters(counter_file)
        counters = exported if exported is not None else {}
        trace_rows = _load_trace_rows(trace_file, mode=mode, prompt_name=prompt["prompt_name"])
        return {
            "prompt_name": prompt["prompt_name"],
            "success": False,
            "output_text": None,
            "output_length": 0,
            "output_token_count": None,
            "prompt_token_count": None,
            "latency_seconds": elapsed,
            **expected_term_report(None, prompt["expected_terms"]),
            **degeneration_report(None),
            "counter_export_file_found": found,
            "counter_export_pid": pid,
            "counter_summary": _mode_counter_summary(counters),
            "trace_file": trace_file,
            "trace_rows": trace_rows,
            "trace_summary": _summarize_trace(trace_rows),
            "env_flags": env_values,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _aggregate_mode(mode: str, prompt_results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    successes = [item for item in prompt_results if item["success"]]
    latencies = [
        float(item["latency_seconds"])
        for item in prompt_results
        if item.get("latency_seconds") is not None
    ]
    counter_summaries = [item["counter_summary"] for item in prompt_results]
    trace_summaries = [item["trace_summary"] for item in prompt_results]
    freed = sum(int(item.get("freed_after_sketch_blocks_total", 0) or 0) for item in counter_summaries)
    blocks_freed = sum(int(item.get("blocks_freed_total", 0) or 0) for item in counter_summaries)
    sketch_bytes = sum(int(item.get("sketch_bytes_total", 0) or 0) for item in counter_summaries)
    ratios = [
        item.get("retention_ratio_avg")
        for item in trace_summaries
        if item.get("retention_ratio_avg") is not None
    ]
    backend = next((item.get("sketch_backend") for item in counter_summaries if item.get("sketch_backend")), None)
    policy = next((item.get("last_policy") for item in counter_summaries if item.get("last_policy")), None)
    scoring_source = next((item.get("scoring_source") for item in counter_summaries if item.get("scoring_source")), None)
    expected_term_prompts = [
        item for item in prompt_results if item["expected_terms"]
    ]
    expected_pass = sum(1 for item in expected_term_prompts if item["contains_expected_terms"])
    degeneration_count = sum(1 for item in prompt_results if item["degeneration_detected"])
    invariants_clean = all(item["counter_summary"]["invariants_clean"] for item in prompt_results)
    backend_pass = (
        mode == BASELINE_MODE
        or (
            backend == args.sketch_backend
            and args.sketch_backend == "countsketch"
        )
    )
    block_savings_pass = freed > 0 and bool(ratios) and min(ratios) < 1.0
    warnings = []
    if mode != BASELINE_MODE and not backend_pass:
        warnings.append("countsketch_backend_not_observed")
    if degeneration_count > 0:
        warnings.append("degeneration_detected")
    if not invariants_clean:
        warnings.append("invariants_not_clean")
    block_bytes_estimate = 0
    block_equivalent_sketch_overhead = None
    if blocks_freed > 0:
        # A conservative block-equivalent estimate is not reliable without
        # dtype/layer/head metadata, so keep raw sketch bytes as the hard fact.
        block_equivalent_sketch_overhead = None
    return {
        "mode": mode,
        "generation_success_count": len(successes),
        "prompt_count": len(prompt_results),
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "total_latency_seconds": sum(latencies),
        "invariants_clean": invariants_clean,
        "warnings": warnings,
        "sketch_backend": backend,
        "last_policy": policy,
        "scoring_source": scoring_source,
        "sketch_dim": args.sketch_dim,
        "sketch_seed": args.sketch_seed,
        "sketch_build_attempted": sum(int(item.get("sketch_build_attempted", 0) or 0) for item in counter_summaries),
        "sketch_build_succeeded": sum(int(item.get("sketch_build_succeeded", 0) or 0) for item in counter_summaries),
        "sketch_build_failed": sum(int(item.get("sketch_build_failed", 0) or 0) for item in counter_summaries),
        "sketched_blocks_total": sum(int(item.get("sketched_blocks_total", 0) or 0) for item in counter_summaries),
        "sketch_bytes_total": sketch_bytes,
        "freed_after_sketch_blocks_total": freed,
        "visible_before_count_max": max((item.get("visible_before_count_max", 0) or 0) for item in trace_summaries) if trace_summaries else 0,
        "visible_after_count_min": min((item.get("visible_after_count_min", 0) or 0) for item in trace_summaries) if trace_summaries else 0,
        "visible_before_count_last": next((item.get("visible_before_count_last", 0) for item in reversed(trace_summaries) if item.get("visible_before_count_last", 0)), 0),
        "visible_after_count_last": next((item.get("visible_after_count_last", 0) for item in reversed(trace_summaries) if item.get("visible_after_count_last", 0)), 0),
        "candidate_demote_count_total": sum(int(item.get("candidate_demote_count_total", 0) or 0) for item in trace_summaries),
        "blocks_freed_total": blocks_freed,
        "retention_ratio_min": min(ratios) if ratios else None,
        "retention_ratio_avg": sum(ratios) / len(ratios) if ratios else None,
        "retention_ratio_last": ratios[-1] if ratios else None,
        "retained_block_fraction_avg": sum(ratios) / len(ratios) if ratios else None,
        "estimated_kv_blocks_saved": blocks_freed,
        "estimated_sketch_overhead_bytes": sketch_bytes,
        "approximate_net_block_savings": None,
        "block_equivalent_sketch_overhead": block_equivalent_sketch_overhead,
        "quality_pass_count": sum(
            1
            for item in prompt_results
            if item["success"]
            and not item["degeneration_detected"]
            and (not item["expected_terms"] or item["contains_expected_terms"])
        ),
        "expected_terms_pass_count": expected_pass,
        "degeneration_count": degeneration_count,
        "block_savings_pass": block_savings_pass,
        "backend_pass": backend_pass,
        "invariants_pass": invariants_clean,
        "gpu_memory_not_directly_measured": True,
        "block_bytes_estimate": block_bytes_estimate,
    }


def _verdict(mode_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    viable_modes = [
        item
        for item in mode_summaries.values()
        if item["backend_pass"]
        and item["invariants_pass"]
        and item["degeneration_count"] == 0
        and item["expected_terms_pass_count"] >= 2
    ]
    savings_modes = [
        item
        for item in mode_summaries.values()
        if item["block_savings_pass"] and item["invariants_pass"]
    ]
    best_quality = (
        max(viable_modes, key=lambda item: item["quality_pass_count"])["mode"]
        if viable_modes
        else None
    )
    best_savings = (
        min(
            savings_modes,
            key=lambda item: (
                item["retention_ratio_avg"]
                if item["retention_ratio_avg"] is not None
                else 1.0
            ),
        )["mode"]
        if savings_modes
        else None
    )
    recommended = best_quality if best_quality in {SAFE_MODE, DEFAULT_MODE} else best_savings
    viable = recommended is not None
    return {
        "best_quality_mode": best_quality,
        "best_savings_mode": best_savings,
        "recommended_next_mode": recommended,
        "whether_countsketch_live_baseline_is_viable": viable,
        "reason_if_not_viable": (
            None
            if viable
            else "No CountSketch mode simultaneously passed expected-term, degeneration, backend, savings, and invariant checks."
        ),
    }


def run_verdict(args: argparse.Namespace) -> dict[str, Any]:
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODE_PRESETS]
    if unknown:
        raise ValueError(f"Unsupported mode(s): {unknown}")
    resolved_model, model_is_local, cached_model_candidates = resolve_model_reference(
        args.model,
        local_files_only=args.local_files_only,
    )
    prompts = build_verdict_prompts(repeats=args.prompt_repeats)
    mode_reports = []
    for mode in modes:
        mode_results = []
        llm = None
        counter_file, trace_file = _paths_for_mode_prompt(
            args=args,
            mode=mode,
            prompt_name="mode_init",
        )
        env_values = _mode_env(
            mode,
            args=args,
            counter_file=counter_file,
            trace_file=trace_file,
        )
        try:
            with patched_environ(env_values):
                from vllm import LLM

                llm_kwargs = _build_llm_kwargs(args)
                llm_kwargs["model"] = resolved_model
                llm = LLM(**llm_kwargs)
                for prompt in prompts:
                    prompt_counter, prompt_trace = _paths_for_mode_prompt(
                        args=args,
                        mode=mode,
                        prompt_name=prompt["prompt_name"],
                    )
                    mode_results.append(
                        _run_prompt(
                            llm,
                            mode=mode,
                            prompt=prompt,
                            args=args,
                            env_values=env_values,
                            counter_file=prompt_counter,
                            trace_file=prompt_trace,
                        )
                    )
        finally:
            if llm is not None:
                del llm
            gc.collect()
        summary = _aggregate_mode(mode, mode_results, args)
        mode_reports.append(
            {
                "mode": mode,
                "preset": MODE_PRESETS[mode],
                "results": mode_results,
                "summary": summary,
            }
        )
    mode_summaries = {report["mode"]: report["summary"] for report in mode_reports}
    return {
        "phase": "Sk-3.5",
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
            "prompt_repeats": args.prompt_repeats,
            "sketch_backend": args.sketch_backend,
            "sketch_dim": args.sketch_dim,
            "sketch_seed": args.sketch_seed,
            "gpu_memory_not_directly_measured": True,
        },
        "mode_reports": mode_reports,
        "mode_summaries": mode_summaries,
        "verdict": _verdict(mode_summaries),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_verdict(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
