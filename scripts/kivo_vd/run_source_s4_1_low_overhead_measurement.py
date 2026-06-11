#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S4.1 low-overhead source measurement harness."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1
from vllm.v1.worker.kivo_runtime_counters import (
    S3_2B_SCHEMA,
    S3_3C_METADATA_SCHEMA,
    S3_3C_PLAN_SCHEMA,
    flatten_counters,
    get_and_reset_counters,
)

BASELINE_MODE = "baseline"
RECENT_WINDOW_VERBOSE_MODE = "recent_window_verbose"
RECENT_WINDOW_COUNTERS_MODE = "recent_window_counters"
SKETCH_ACTIVE_VERBOSE_MODE = "sketch_active_verbose"
SKETCH_ACTIVE_COUNTERS_MODE = "sketch_active_counters"

RECENT_WINDOW_POLICY = "active_recent_window_attention_metadata"
SKETCH_ACTIVE_POLICY = "active_sketch_kv_metadata_alias"

DEFAULT_PROMPTS = [
    "The quick brown fox",
    (
        "Machine learning systems can process long contexts by reusing "
        "cached key value states across generation steps."
    ),
    (
        "Kivo source experiments can compact recent-window attention "
        "metadata while staying in the observation-only boundary."
    ),
    (
        "Longer context validation helps test whether cloned metadata paths "
        "can travel through the existing vLLM attention stack without custom "
        "kernels. This prompt repeats the same idea in slightly varied wording "
        "so the token count grows past a few logical cache blocks while "
        "remaining comfortably inside the configured model length budget for a "
        "controlled quick measurement experiment."
    ),
]

MODE_ORDER = [
    BASELINE_MODE,
    RECENT_WINDOW_VERBOSE_MODE,
    RECENT_WINDOW_COUNTERS_MODE,
    SKETCH_ACTIVE_VERBOSE_MODE,
    SKETCH_ACTIVE_COUNTERS_MODE,
]

ACTIVE_MODES = [
    RECENT_WINDOW_VERBOSE_MODE,
    RECENT_WINDOW_COUNTERS_MODE,
    SKETCH_ACTIVE_VERBOSE_MODE,
    SKETCH_ACTIVE_COUNTERS_MODE,
]

MODE_TO_POLICY = {
    RECENT_WINDOW_VERBOSE_MODE: RECENT_WINDOW_POLICY,
    RECENT_WINDOW_COUNTERS_MODE: RECENT_WINDOW_POLICY,
    SKETCH_ACTIVE_VERBOSE_MODE: SKETCH_ACTIVE_POLICY,
    SKETCH_ACTIVE_COUNTERS_MODE: SKETCH_ACTIVE_POLICY,
}

MODE_TO_RECORD_MODE = {
    BASELINE_MODE: "off",
    RECENT_WINDOW_VERBOSE_MODE: "events",
    RECENT_WINDOW_COUNTERS_MODE: "counters",
    SKETCH_ACTIVE_VERBOSE_MODE: "events",
    SKETCH_ACTIVE_COUNTERS_MODE: "counters",
}

_SOURCE_ENV_KEYS = [
    "KIVO_SOURCE_ENABLE",
    "KIVO_SOURCE_OBSERVE_PATH",
    "KIVO_SOURCE_OBS_PATH",
    "KIVO_SOURCE_POLICY",
    "KIVO_SOURCE_ACTIVE",
    "KIVO_SOURCE_FAIL_CLOSED",
    "KIVO_SOURCE_KEEP_RECENT_BLOCKS",
    "KIVO_SOURCE_ACTIVE_FILTER_MODE",
    "KIVO_SOURCE_SKETCH_DIM",
    "KIVO_SOURCE_MAX_SKETCH_BLOCKS",
    "KIVO_SOURCE_BUDGET_RATIO",
    "KIVO_SOURCE_SKETCH_SEED",
    "KIVO_SOURCE_RECORD_MODE",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase S4.1 low-overhead source measurement."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=768)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--keep-recent-blocks", type=int, default=1)
    parser.add_argument("--sketch-dim", type=int, default=8)
    parser.add_argument("--max-sketch-blocks", type=int, default=4)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--sketch-seed", type=int, default=123)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement.md",
    )
    parser.add_argument(
        "--events-jsonl",
        default=(
            "outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement_events.jsonl"
        ),
    )
    parser.add_argument(
        "--force-inproc-engine-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable V1 multiprocessing before importing vLLM so counters stay in-process.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _build_generation_args(args: argparse.Namespace, prompt: str) -> SimpleNamespace:
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


def _clear_source_env() -> None:
    for key in _SOURCE_ENV_KEYS:
        os.environ.pop(key, None)


def _capture_source_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _SOURCE_ENV_KEYS}


def _restore_source_env(snapshot: dict[str, str | None]) -> None:
    _clear_source_env()
    for key, value in snapshot.items():
        if value is not None:
            os.environ[key] = value


def _set_mode_env(
    mode: str,
    *,
    event_path: Path | None,
    args: argparse.Namespace,
) -> None:
    _clear_source_env()
    record_mode = MODE_TO_RECORD_MODE[mode]
    if mode == BASELINE_MODE:
        return
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_RECORD_MODE"] = record_mode
    os.environ["KIVO_SOURCE_POLICY"] = MODE_TO_POLICY[mode]
    os.environ["KIVO_SOURCE_ACTIVE"] = "1"
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"
    if event_path is not None:
        os.environ["KIVO_SOURCE_OBSERVE_PATH"] = str(event_path)
        os.environ["KIVO_SOURCE_OBS_PATH"] = str(event_path)
    if mode.startswith("recent_window"):
        os.environ["KIVO_SOURCE_KEEP_RECENT_BLOCKS"] = str(
            args.keep_recent_blocks
        )
        os.environ["KIVO_SOURCE_ACTIVE_FILTER_MODE"] = (
            "compact_to_recent_window"
        )
    if mode.startswith("sketch_active"):
        os.environ["KIVO_SOURCE_SKETCH_DIM"] = str(args.sketch_dim)
        os.environ["KIVO_SOURCE_MAX_SKETCH_BLOCKS"] = str(
            args.max_sketch_blocks
        )
        os.environ["KIVO_SOURCE_BUDGET_RATIO"] = str(args.budget_ratio)
        os.environ["KIVO_SOURCE_SKETCH_SEED"] = str(args.sketch_seed)
        os.environ["KIVO_SOURCE_ACTIVE_FILTER_MODE"] = (
            "alias_excluded_blocks_to_sketch_selected"
        )


def _extract_text(outputs: list[Any]) -> str:
    if not outputs:
        return ""
    first = outputs[0]
    candidates = getattr(first, "outputs", None)
    if not candidates:
        return ""
    return str(candidates[0].text)


def _extract_prompt_token_length(outputs: list[Any]) -> int | None:
    if not outputs:
        return None
    first = outputs[0]
    token_ids = getattr(first, "prompt_token_ids", None)
    return len(token_ids) if token_ids is not None else None


def _extract_generated_token_count(outputs: list[Any], fallback: int) -> int | None:
    if not outputs:
        return None
    first = outputs[0]
    candidates = getattr(first, "outputs", None)
    if not candidates:
        return None
    token_ids = getattr(candidates[0], "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return fallback


def _run_generation(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        seed=args.seed,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    try:
        outputs = llm.generate(
            [args.prompt],
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
            ),
            use_tqdm=False,
        )
        text = _extract_text(outputs)
        prompt_token_length = _extract_prompt_token_length(outputs)
        generated_token_count = _extract_generated_token_count(
            outputs, fallback=args.max_tokens
        )
        return {
            "status": "succeeded",
            "output_text": text,
            "prompt_token_length": prompt_token_length,
            "generated_token_count": generated_token_count,
            "error": None,
        }
    finally:
        del llm
        gc.collect()


def _run_generation_safe(
    args: argparse.Namespace,
    generation_fn: Any,
    *,
    timer_fn: Any = time.perf_counter,
) -> dict[str, Any]:
    started = timer_fn()
    try:
        result = dict(generation_fn(args))
    except Exception as exc:
        result = {
            "status": "failed",
            "output_text": None,
            "prompt_token_length": None,
            "generated_token_count": None,
            "error": (
                f"{type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()[-4000:]}"
            ),
        }
    result.setdefault("status", "succeeded")
    elapsed = timer_fn() - started
    generated_token_count = result.get("generated_token_count")
    tokens_per_second = None
    if (
        result.get("status") == "succeeded"
        and isinstance(generated_token_count, int)
        and generated_token_count >= 0
        and elapsed > 0
    ):
        tokens_per_second = generated_token_count / elapsed
    result.update(
        {
            "latency_seconds": elapsed,
            "latency_ms": elapsed * 1000.0,
            "tokens_per_second": tokens_per_second,
        }
    )
    return result


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return source_s1.load_records(path)


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _concat_jsonl(
    paths: list[Path],
    output_path: Path,
    *,
    record_loader: Any | None = None,
) -> list[dict[str, Any]]:
    if record_loader is None:
        record_loader = _load_records
    records: list[dict[str, Any]] = []
    lines: list[str] = []
    for path in paths:
        records.extend(record_loader(path))
        if path.exists():
            lines.extend(
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return records


def _event_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    total_raw = len(records)
    total_s3_2b = sum(
        record.get("schema_version") == S3_2B_SCHEMA for record in records
    )
    total_s3_3c_plan = sum(
        record.get("schema_version") == S3_3C_PLAN_SCHEMA for record in records
    )
    total_s3_3c_metadata = sum(
        record.get("schema_version") == S3_3C_METADATA_SCHEMA
        for record in records
    )
    return {
        "total_raw_events": total_raw,
        "total_s3_2b_events": total_s3_2b,
        "total_s3_3c_sketch_plan_events": total_s3_3c_plan,
        "total_s3_3c_metadata_alias_events": total_s3_3c_metadata,
        "ignored_non_s3_events": (
            total_raw - total_s3_2b - total_s3_3c_plan - total_s3_3c_metadata
        ),
    }


def _max_int(records: list[dict[str, Any]], field: str) -> int:
    values: list[int] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values, default=0)


def _max_float(records: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def _unique_blocker_reasons(records: list[dict[str, Any]]) -> list[str]:
    reasons = {
        str(record[field])
        for record in records
        for field in ("mutation_blocker_reason", "sketch_plan_blocker_reason")
        if record.get(field)
    }
    return sorted(reasons)


def _summarize_recent_window_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    filtered = [
        record for record in records if record.get("schema_version") == S3_2B_SCHEMA
    ]
    blocker_reasons = _unique_blocker_reasons(filtered)
    return {
        "records_written": len(filtered),
        "total_s3_2b_events": len(filtered),
        "mutation_attempted_event_count": sum(
            record.get("mutation_attempted") is True for record in filtered
        ),
        "mutation_applied_event_count": sum(
            record.get("mutation_applied") is True for record in filtered
        ),
        "active_routing_event_count": sum(
            record.get("active_routing") is True for record in filtered
        ),
        "runtime_behavior_changed_event_count": sum(
            record.get("runtime_behavior_changed") is True for record in filtered
        ),
        "blocker_event_count": sum(
            bool(record.get("mutation_blocker_reason")) for record in filtered
        ),
        "blocker_reasons": blocker_reasons,
        "max_visible_block_count": max(
            [
                _max_int(filtered, "visible_block_count"),
                _max_int(filtered, "visible_block_count_estimate"),
                _max_int(filtered, "original_visible_block_count"),
            ],
            default=0,
        ),
        "max_original_visible_block_count": _max_int(
            filtered, "original_visible_block_count"
        ),
        "max_selected_block_count": _max_int(filtered, "selected_block_count"),
        "max_excluded_block_count": _max_int(filtered, "excluded_block_count"),
        "max_unselected_block_count": _max_int(filtered, "excluded_block_count"),
        "max_theoretical_attention_visible_block_reduction": _max_int(
            filtered, "theoretical_attention_visible_block_reduction"
        ),
        "max_theoretical_attention_visible_block_reduction_ratio": _max_float(
            filtered, "theoretical_attention_visible_block_reduction_ratio"
        ),
    }


def _summarize_sketch_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    plan_records = [
        record for record in records if record.get("schema_version") == S3_3C_PLAN_SCHEMA
    ]
    metadata_records = [
        record
        for record in records
        if record.get("schema_version") == S3_3C_METADATA_SCHEMA
    ]
    blocker_reasons = _unique_blocker_reasons(plan_records + metadata_records)
    return {
        "records_written": len(plan_records) + len(metadata_records),
        "total_s3_3c_sketch_plan_events": len(plan_records),
        "total_s3_3c_metadata_alias_events": len(metadata_records),
        "sketch_computed_event_count": sum(
            item.get("sketch_computed") is True for item in plan_records
        ),
        "sketch_plan_used_event_count": sum(
            item.get("sketch_plan_used") is True for item in metadata_records
        ),
        "mutation_attempted_event_count": sum(
            item.get("mutation_attempted") is True for item in metadata_records
        ),
        "mutation_applied_event_count": sum(
            item.get("mutation_applied") is True for item in metadata_records
        ),
        "active_routing_event_count": sum(
            item.get("active_routing") is True for item in metadata_records
        ),
        "runtime_behavior_changed_event_count": sum(
            item.get("runtime_behavior_changed") is True for item in metadata_records
        ),
        "blocker_event_count": sum(
            bool(item.get("sketch_plan_blocker_reason"))
            or bool(item.get("mutation_blocker_reason"))
            for item in metadata_records
        ),
        "blocker_reasons": blocker_reasons,
        "max_candidate_block_count": _max_int(plan_records, "candidate_block_count"),
        "max_selected_block_count": max(
            _max_int(plan_records, "selected_block_count"),
            _max_int(metadata_records, "selected_block_count"),
        ),
        "max_excluded_block_count": max(
            _max_int(plan_records, "excluded_block_count"),
            _max_int(metadata_records, "excluded_block_count"),
        ),
        "max_unselected_block_count": max(
            _max_int(plan_records, "excluded_block_count"),
            _max_int(metadata_records, "excluded_block_count"),
        ),
        "max_aliased_block_count": _max_int(metadata_records, "aliased_block_count"),
        "max_visible_block_count": _max_int(metadata_records, "visible_block_count"),
        "max_original_visible_block_count": _max_int(
            metadata_records, "original_visible_block_count"
        ),
        "max_theoretical_attention_visible_block_reduction_ratio": max(
            _max_float(
                plan_records, "theoretical_attention_visible_block_reduction_ratio"
            ),
            _max_float(
                metadata_records,
                "theoretical_attention_visible_block_reduction_ratio",
            ),
        ),
    }


def _merge_flattened_counter_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not summaries:
        return flatten_counters({})

    combined: dict[str, Any] = {
        "event_count": 0,
        "sketch_computed_count": 0,
        "sketch_blocked_count": 0,
        "mutation_attempted_count": 0,
        "mutation_applied_count": 0,
        "active_routing_count": 0,
        "runtime_behavior_changed_count": 0,
        "blocker_count": 0,
        "blocker_reason_counts": {},
        "max_candidate_block_count": 0,
        "max_selected_block_count": 0,
        "max_excluded_block_count": 0,
        "max_unselected_block_count": 0,
        "max_aliased_block_count": 0,
        "max_visible_block_count": 0,
        "max_original_visible_block_count": 0,
        "max_theoretical_attention_visible_block_reduction_ratio": 0.0,
        "sketch_plan_used_count": 0,
        "metadata_alias_count": 0,
        "recent_window_event_count": 0,
        "min_seq_len": None,
        "max_seq_len": None,
        "last_seq_len": None,
    }
    for summary in summaries:
        combined["event_count"] += int(summary.get("event_count", 0) or 0)
        combined["sketch_computed_count"] += int(
            summary.get("sketch_computed_count", 0) or 0
        )
        combined["sketch_blocked_count"] += int(
            summary.get("sketch_blocked_count", 0) or 0
        )
        combined["mutation_attempted_count"] += int(
            summary.get("mutation_attempted_count", 0) or 0
        )
        combined["mutation_applied_count"] += int(
            summary.get("mutation_applied_count", 0) or 0
        )
        combined["active_routing_count"] += int(
            summary.get("active_routing_count", 0) or 0
        )
        combined["runtime_behavior_changed_count"] += int(
            summary.get("runtime_behavior_changed_count", 0) or 0
        )
        combined["blocker_count"] += int(summary.get("blocker_count", 0) or 0)
        combined["sketch_plan_used_count"] += int(
            summary.get("sketch_plan_used_count", 0) or 0
        )
        combined["metadata_alias_count"] += int(
            summary.get("metadata_alias_count", 0) or 0
        )
        combined["recent_window_event_count"] += int(
            summary.get("recent_window_event_count", 0) or 0
        )
        for field in [
            "max_candidate_block_count",
            "max_selected_block_count",
            "max_excluded_block_count",
            "max_unselected_block_count",
            "max_aliased_block_count",
            "max_visible_block_count",
            "max_original_visible_block_count",
        ]:
            combined[field] = max(
                int(combined.get(field, 0) or 0),
                int(summary.get(field, 0) or 0),
            )
        try:
            combined["max_theoretical_attention_visible_block_reduction_ratio"] = max(
                float(
                    combined["max_theoretical_attention_visible_block_reduction_ratio"]
                ),
                float(
                    summary.get(
                        "max_theoretical_attention_visible_block_reduction_ratio",
                        0.0,
                    )
                    or 0.0
                ),
            )
        except (TypeError, ValueError):
            pass
        for key, value in (summary.get("blocker_reason_counts") or {}).items():
            reason_counts = combined.setdefault("blocker_reason_counts", {})
            reason_counts[str(key)] = int(reason_counts.get(str(key), 0)) + int(value)
        if summary.get("min_seq_len") is not None:
            current_min = combined.get("min_seq_len")
            combined["min_seq_len"] = (
                int(summary["min_seq_len"])
                if current_min is None
                else min(int(current_min), int(summary["min_seq_len"]))
            )
        if summary.get("max_seq_len") is not None:
            combined["max_seq_len"] = (
                int(summary["max_seq_len"])
                if combined.get("max_seq_len") is None
                else max(int(combined["max_seq_len"]), int(summary["max_seq_len"]))
            )
        if summary.get("last_seq_len") is not None:
            combined["last_seq_len"] = int(summary["last_seq_len"])
    return combined


def _mode_event_path(
    events_root: Path,
    *,
    mode: str,
    prompt_index: int,
    repeat_index: int,
    warmup: bool,
) -> Path:
    suffix = "warmup" if warmup else "repeat"
    return events_root.with_name(
        f"{events_root.stem}_{mode}_prompt_{prompt_index:02d}_{suffix}_{repeat_index:02d}.jsonl"
    )


def _run_mode_generation(
    *,
    mode: str,
    prompt_index: int,
    repeat_index: int,
    prompt: str,
    args: argparse.Namespace,
    generation_fn: Any,
    timer_fn: Any,
    warmup: bool,
    events_root: Path,
    record_loader: Any,
    counter_loader: Any,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Path | None,
    dict[str, Any],
    list[str],
]:
    generation_args = _build_generation_args(args, prompt)
    event_path: Path | None = None
    if mode != BASELINE_MODE:
        event_path = _mode_event_path(
            events_root,
            mode=mode,
            prompt_index=prompt_index,
            repeat_index=repeat_index,
            warmup=warmup,
        )
        if event_path.exists():
            event_path.unlink()
    _set_mode_env(mode, event_path=event_path, args=args)
    result = _run_generation_safe(
        generation_args,
        generation_fn,
        timer_fn=timer_fn,
    )
    records: list[dict[str, Any]] = []
    if event_path is not None:
        records = record_loader(event_path)
        if warmup and event_path.exists():
            event_path.unlink()
    try:
        counter_snapshot = dict(counter_loader())
        counter_loader_error = None
    except Exception as exc:  # pragma: no cover - defensive path
        counter_snapshot = {}
        counter_loader_error = f"{type(exc).__name__}: {exc}"
    counter_summary = flatten_counters(counter_snapshot)
    counter_schema_versions = sorted(counter_snapshot)
    return (
        result,
        records,
        event_path,
        counter_summary,
        counter_schema_versions if counter_loader_error is None else [],
    )


def _aggregate_mode_runs(run_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_mode: dict[str, dict[str, Any]] = {}
    for mode in MODE_ORDER:
        mode_runs = [item for item in run_records if item["mode"] == mode]
        successful_runs = [item for item in mode_runs if item["status"] == "succeeded"]
        latencies = [float(item["latency_seconds"]) for item in successful_runs]
        tokens_per_second = [
            float(item["tokens_per_second"])
            for item in successful_runs
            if item.get("tokens_per_second") is not None
        ]
        generated_token_total = sum(
            int(item.get("generated_token_count") or 0) for item in successful_runs
        )
        output_changed_count = sum(
            bool(item.get("output_changed_vs_baseline")) for item in successful_runs
        )
        verbose_event_records = [
            record for item in mode_runs for record in item.get("event_records", [])
        ]
        event_summary = (
            _summarize_recent_window_records(verbose_event_records)
            if mode.startswith("recent_window")
            else _summarize_sketch_records(verbose_event_records)
            if mode.startswith("sketch_active")
            else {
                "records_written": 0,
                "total_s3_2b_events": 0,
                "total_s3_3c_sketch_plan_events": 0,
                "total_s3_3c_metadata_alias_events": 0,
                "mutation_attempted_event_count": 0,
                "mutation_applied_event_count": 0,
                "active_routing_event_count": 0,
                "runtime_behavior_changed_event_count": 0,
                "blocker_event_count": 0,
                "blocker_reasons": [],
                "max_visible_block_count": 0,
                "max_original_visible_block_count": 0,
                "max_selected_block_count": 0,
                "max_excluded_block_count": 0,
                "max_unselected_block_count": 0,
                "max_theoretical_attention_visible_block_reduction": 0,
                "max_theoretical_attention_visible_block_reduction_ratio": 0.0,
                "max_candidate_block_count": 0,
                "max_aliased_block_count": 0,
                "sketch_computed_event_count": 0,
                "sketch_plan_used_event_count": 0,
            }
        )
        counter_summaries = [item.get("counter_summary", {}) for item in mode_runs]
        counter_summary = _merge_flattened_counter_summaries(counter_summaries)
        counter_schema_versions = sorted(
            {
                schema
                for item in mode_runs
                for schema in item.get("counter_schema_versions", [])
            }
        )
        verbose_event_record_count = sum(
            int(item.get("verbose_event_record_count") or 0) for item in mode_runs
        )
        per_mode[mode] = {
            "mode": mode,
            "attempted_runs": len(mode_runs),
            "success_count": len(successful_runs),
            "failure_count": len(mode_runs) - len(successful_runs),
            "measured_run_count": len(mode_runs),
            "mean_latency_seconds": (
                statistics.mean(latencies) if latencies else None
            ),
            "median_latency_seconds": (
                statistics.median(latencies) if latencies else None
            ),
            "min_latency_seconds": min(latencies) if latencies else None,
            "max_latency_seconds": max(latencies) if latencies else None,
            "mean_tokens_per_second": (
                statistics.mean(tokens_per_second) if tokens_per_second else None
            ),
            "generated_token_total": generated_token_total,
            "output_changed_count_vs_baseline": output_changed_count,
            "verbose_event_record_count": verbose_event_record_count,
            "counter_event_count": int(counter_summary.get("event_count", 0) or 0),
            "event_summary": event_summary,
            "counter_summary": counter_summary,
            "counter_schema_versions": counter_schema_versions,
            "blocker_reasons": sorted(
                set(event_summary.get("blocker_reasons", []))
                | set(
                    key
                    for key, value in (
                        counter_summary.get("blocker_reason_counts") or {}
                    ).items()
                    if int(value) > 0
                )
            ),
            "max_visible_block_count": max(
                int(event_summary.get("max_visible_block_count", 0) or 0),
                int(counter_summary.get("max_visible_block_count", 0) or 0),
            ),
            "max_original_visible_block_count": max(
                int(event_summary.get("max_original_visible_block_count", 0) or 0),
                int(counter_summary.get("max_original_visible_block_count", 0) or 0),
            ),
            "max_selected_block_count": max(
                int(event_summary.get("max_selected_block_count", 0) or 0),
                int(counter_summary.get("max_selected_block_count", 0) or 0),
            ),
            "max_excluded_block_count": max(
                int(event_summary.get("max_excluded_block_count", 0) or 0),
                int(counter_summary.get("max_excluded_block_count", 0) or 0),
            ),
            "max_unselected_block_count": max(
                int(event_summary.get("max_unselected_block_count", 0) or 0),
                int(counter_summary.get("max_unselected_block_count", 0) or 0),
            ),
            "max_candidate_block_count": max(
                int(event_summary.get("max_candidate_block_count", 0) or 0),
                int(counter_summary.get("max_candidate_block_count", 0) or 0),
            ),
            "max_aliased_block_count": max(
                int(event_summary.get("max_aliased_block_count", 0) or 0),
                int(counter_summary.get("max_aliased_block_count", 0) or 0),
            ),
            "max_theoretical_attention_visible_block_reduction_ratio": max(
                float(
                    event_summary.get(
                        "max_theoretical_attention_visible_block_reduction_ratio",
                        0.0,
                    )
                    or 0.0
                ),
                float(
                    counter_summary.get(
                        "max_theoretical_attention_visible_block_reduction_ratio",
                        0.0,
                    )
                    or 0.0
                ),
            ),
            "sketch_computed_count": int(
                counter_summary.get("sketch_computed_count", 0) or 0
            ),
            "sketch_blocked_count": int(
                counter_summary.get("sketch_blocked_count", 0) or 0
            ),
            "mutation_attempted_count": int(
                max(
                    event_summary.get("mutation_attempted_event_count", 0),
                    counter_summary.get("mutation_attempted_count", 0),
                )
                or 0
            ),
            "mutation_applied_count": int(
                max(
                    event_summary.get("mutation_applied_event_count", 0),
                    counter_summary.get("mutation_applied_count", 0),
                )
                or 0
            ),
            "active_routing_count": int(
                max(
                    event_summary.get("active_routing_event_count", 0),
                    counter_summary.get("active_routing_count", 0),
                )
                or 0
            ),
            "runtime_behavior_changed_count": int(
                max(
                    event_summary.get("runtime_behavior_changed_event_count", 0),
                    counter_summary.get("runtime_behavior_changed_count", 0),
                )
                or 0
            ),
            "sketch_plan_used_count": int(
                counter_summary.get("sketch_plan_used_count", 0) or 0
            ),
            "metadata_alias_count": int(
                counter_summary.get("metadata_alias_count", 0) or 0
            ),
            "recent_window_event_count": int(
                counter_summary.get("recent_window_event_count", 0) or 0
            ),
            "records_written": int(
                event_summary.get("records_written", 0) or 0
            ),
        }
    return per_mode


def build_report(
    args: argparse.Namespace,
    *,
    generation_fn: Any = _run_generation,
    record_loader: Any = _load_records,
    counter_loader: Any = get_and_reset_counters,
    timer_fn: Any = time.perf_counter,
) -> dict[str, Any]:
    prompts = [str(prompt) for prompt in DEFAULT_PROMPTS if str(prompt).strip()]
    if not prompts:
        raise ValueError("no prompts configured for S4.1 low-overhead measurement")

    try:
        counter_loader()
    except Exception:
        pass

    previous_env = _capture_source_env()
    events_root = Path(args.events_jsonl)
    if events_root.exists():
        events_root.unlink()

    run_records: list[dict[str, Any]] = []
    baseline_lookup: dict[tuple[int, int], str | None] = {}
    mode_event_paths: dict[str, list[Path]] = {mode: [] for mode in ACTIVE_MODES}
    mode_verbose_event_jsonl: dict[str, str | None] = {
        BASELINE_MODE: None,
        RECENT_WINDOW_VERBOSE_MODE: None,
        RECENT_WINDOW_COUNTERS_MODE: None,
        SKETCH_ACTIVE_VERBOSE_MODE: None,
        SKETCH_ACTIVE_COUNTERS_MODE: None,
    }

    try:
        for prompt_index, prompt in enumerate(prompts):
            for mode in MODE_ORDER:
                warmup_event = (
                    _mode_event_path(
                        events_root,
                        mode=mode,
                        prompt_index=prompt_index,
                        repeat_index=0,
                        warmup=True,
                    )
                    if mode != BASELINE_MODE
                    else None
                )
                if warmup_event is not None and warmup_event.exists():
                    warmup_event.unlink()
                _set_mode_env(mode, event_path=warmup_event, args=args)
                warmup_result = _run_generation_safe(
                    _build_generation_args(args, prompt),
                    generation_fn,
                    timer_fn=timer_fn,
                )
                try:
                    counter_loader()
                except Exception:
                    pass
                if warmup_event is not None and warmup_event.exists():
                    warmup_event.unlink()
                if (
                    not args.continue_on_error
                    and warmup_result.get("status") != "succeeded"
                ):
                    raise RuntimeError(
                        f"warmup measurement failed for prompt {prompt_index} "
                        f"mode {mode}"
                    )

            for repeat_index in range(args.repeats):
                for mode in MODE_ORDER:
                    event_path = (
                        _mode_event_path(
                            events_root,
                            mode=mode,
                            prompt_index=prompt_index,
                            repeat_index=repeat_index,
                            warmup=False,
                        )
                        if mode != BASELINE_MODE
                        else None
                    )
                    if event_path is not None and event_path.exists():
                        event_path.unlink()
                    _set_mode_env(mode, event_path=event_path, args=args)
                    result = _run_generation_safe(
                        _build_generation_args(args, prompt),
                        generation_fn,
                        timer_fn=timer_fn,
                    )
                    event_records = (
                        record_loader(event_path)
                        if event_path is not None
                        else []
                    )
                    if event_path is not None:
                        mode_event_paths[mode].append(event_path)
                    try:
                        counter_snapshot = counter_loader()
                        counter_loader_error = None
                    except Exception as exc:
                        counter_snapshot = {}
                        counter_loader_error = f"{type(exc).__name__}: {exc}"
                    counter_summary = flatten_counters(counter_snapshot)
                    counter_schema_versions = sorted(counter_snapshot)
                    if mode == BASELINE_MODE:
                        baseline_lookup[(prompt_index, repeat_index)] = result.get(
                            "output_text"
                        )
                    output_changed_vs_baseline = None
                    baseline_output = baseline_lookup.get((prompt_index, repeat_index))
                    if (
                        mode != BASELINE_MODE
                        and result.get("status") == "succeeded"
                        and baseline_output is not None
                    ):
                        output_changed_vs_baseline = (
                            result.get("output_text") != baseline_output
                        )
                    run_records.append(
                        {
                            "mode": mode,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "repeat_index": repeat_index,
                            "status": result.get("status"),
                            "output_text": result.get("output_text"),
                            "prompt_token_length": result.get("prompt_token_length"),
                            "generated_token_count": result.get(
                                "generated_token_count"
                            ),
                            "latency_seconds": result.get("latency_seconds"),
                            "latency_ms": result.get("latency_ms"),
                            "tokens_per_second": result.get("tokens_per_second"),
                            "output_changed_vs_baseline": output_changed_vs_baseline,
                            "error": result.get("error"),
                            "events_jsonl": str(event_path) if event_path else None,
                            "event_records": event_records,
                            "verbose_event_record_count": len(event_records),
                            "counter_summary": counter_summary,
                            "counter_schema_versions": counter_schema_versions,
                            "counter_loader_error": counter_loader_error,
                            "zero_event_debug": (
                                {
                                    "record_mode": MODE_TO_RECORD_MODE.get(mode),
                                    "env_policy_used": MODE_TO_POLICY.get(mode),
                                    "observe_path": str(event_path),
                                    "file_exists": event_path.exists()
                                    if event_path is not None
                                    else False,
                                    "file_size": (
                                        event_path.stat().st_size
                                        if event_path is not None
                                        and event_path.exists()
                                        else 0
                                    ),
                                }
                                if mode != BASELINE_MODE and len(event_records) == 0
                                else None
                            ),
                        }
                    )
                    if (
                        not args.continue_on_error
                        and result.get("status") != "succeeded"
                    ):
                        raise RuntimeError(
                            f"{mode} measurement failed for prompt {prompt_index} "
                            f"repeat {repeat_index}"
                        )

        for mode in ACTIVE_MODES:
            mode_output_path = events_root.with_name(
                f"{events_root.stem}_{mode}.jsonl"
            )
            _concat_jsonl(
                mode_event_paths[mode],
                mode_output_path,
                record_loader=record_loader,
            )
            if mode in {RECENT_WINDOW_VERBOSE_MODE, SKETCH_ACTIVE_VERBOSE_MODE}:
                mode_verbose_event_jsonl[mode] = str(mode_output_path)
        combined_events_records = _concat_jsonl(
            [
                events_root.with_name(f"{events_root.stem}_{mode}.jsonl")
                for mode in ACTIVE_MODES
            ],
            events_root,
            record_loader=record_loader,
        )
    finally:
        _restore_source_env(previous_env)

    per_mode = _aggregate_mode_runs(run_records)
    for mode, event_jsonl in mode_verbose_event_jsonl.items():
        per_mode[mode]["events_jsonl"] = event_jsonl

    total_measured_runs = len(prompts) * int(args.repeats)
    baseline_success_count = per_mode[BASELINE_MODE]["success_count"]
    recent_verbose_success_count = per_mode[RECENT_WINDOW_VERBOSE_MODE][
        "success_count"
    ]
    recent_counters_success_count = per_mode[RECENT_WINDOW_COUNTERS_MODE][
        "success_count"
    ]
    sketch_verbose_success_count = per_mode[SKETCH_ACTIVE_VERBOSE_MODE][
        "success_count"
    ]
    sketch_counters_success_count = per_mode[SKETCH_ACTIVE_COUNTERS_MODE][
        "success_count"
    ]

    baseline_mean = per_mode[BASELINE_MODE]["mean_latency_seconds"]
    recent_verbose_mean = per_mode[RECENT_WINDOW_VERBOSE_MODE][
        "mean_latency_seconds"
    ]
    recent_counters_mean = per_mode[RECENT_WINDOW_COUNTERS_MODE][
        "mean_latency_seconds"
    ]
    sketch_verbose_mean = per_mode[SKETCH_ACTIVE_VERBOSE_MODE]["mean_latency_seconds"]
    sketch_counters_mean = per_mode[SKETCH_ACTIVE_COUNTERS_MODE]["mean_latency_seconds"]

    def _ratio(numerator: float | None, denominator: float | None) -> float | None:
        if (
            numerator is None
            or denominator is None
            or denominator <= 0
        ):
            return None
        return float(numerator / denominator)

    comparisons = {
        "recent_window_verbose_vs_baseline_latency_ratio": _ratio(
            recent_verbose_mean, baseline_mean
        ),
        "recent_window_counters_vs_baseline_latency_ratio": _ratio(
            recent_counters_mean, baseline_mean
        ),
        "recent_window_counters_vs_verbose_latency_ratio": _ratio(
            recent_counters_mean, recent_verbose_mean
        ),
        "sketch_active_verbose_vs_baseline_latency_ratio": _ratio(
            sketch_verbose_mean, baseline_mean
        ),
        "sketch_active_counters_vs_baseline_latency_ratio": _ratio(
            sketch_counters_mean, baseline_mean
        ),
        "sketch_active_counters_vs_verbose_latency_ratio": _ratio(
            sketch_counters_mean, sketch_verbose_mean
        ),
    }

    all_success = bool(
        baseline_success_count == total_measured_runs
        and recent_verbose_success_count == total_measured_runs
        and recent_counters_success_count == total_measured_runs
        and sketch_verbose_success_count == total_measured_runs
        and sketch_counters_success_count == total_measured_runs
    )

    measured_runtime_reduction = bool(
        all_success
        and baseline_mean is not None
        and recent_counters_mean is not None
        and sketch_counters_mean is not None
        and recent_counters_mean <= baseline_mean * 0.95
        and sketch_counters_mean <= baseline_mean * 0.95
    )

    output_changed_count = sum(
        bool(item.get("output_changed_vs_baseline"))
        for item in run_records
        if item["mode"] in ACTIVE_MODES and item["status"] == "succeeded"
    )
    verbose_event_record_count_total = sum(
        int(item.get("verbose_event_record_count") or 0) for item in run_records
    )
    counter_event_count_total = sum(
        int(item.get("counter_summary", {}).get("event_count", 0) or 0)
        for item in run_records
    )

    recent_counter_summary = per_mode[RECENT_WINDOW_COUNTERS_MODE]["counter_summary"]
    sketch_counter_summary = per_mode[SKETCH_ACTIVE_COUNTERS_MODE]["counter_summary"]
    recent_verbose_event_summary = per_mode[RECENT_WINDOW_VERBOSE_MODE][
        "event_summary"
    ]
    sketch_verbose_event_summary = per_mode[SKETCH_ACTIVE_VERBOSE_MODE][
        "event_summary"
    ]

    recent_counter_pass = bool(
        int(recent_counter_summary.get("event_count", 0) or 0) > 0
        and (
            int(recent_counter_summary.get("mutation_applied_count", 0) or 0) > 0
            or int(recent_counter_summary.get("blocker_count", 0) or 0) > 0
        )
    )
    sketch_counter_pass = bool(
        int(sketch_counter_summary.get("event_count", 0) or 0) > 0
        and (
            int(sketch_counter_summary.get("sketch_computed_count", 0) or 0) > 0
            or int(sketch_counter_summary.get("blocker_count", 0) or 0) > 0
        )
        and (
            int(sketch_counter_summary.get("mutation_applied_count", 0) or 0) > 0
            or int(sketch_counter_summary.get("blocker_count", 0) or 0) > 0
        )
    )

    verbose_event_pass = bool(
        int(recent_verbose_event_summary.get("records_written", 0) or 0) > 0
        and int(sketch_verbose_event_summary.get("records_written", 0) or 0) > 0
    )

    passed = bool(
        total_measured_runs > 0
        and all_success
        and verbose_event_pass
        and recent_counter_pass
        and sketch_counter_pass
        and comparisons["recent_window_counters_vs_verbose_latency_ratio"] is not None
        and comparisons["sketch_active_counters_vs_verbose_latency_ratio"] is not None
        and verbose_event_record_count_total >= 0
        and counter_event_count_total >= 0
    )

    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "keep_recent_blocks": args.keep_recent_blocks,
        "sketch_dim": args.sketch_dim,
        "max_sketch_blocks": args.max_sketch_blocks,
        "budget_ratio": args.budget_ratio,
        "sketch_seed": args.sketch_seed,
        "modes": MODE_ORDER,
        "total_prompts": len(prompts),
        "measured_run_count": total_measured_runs * len(MODE_ORDER),
        "baseline_success_count": baseline_success_count,
        "recent_window_verbose_success_count": recent_verbose_success_count,
        "recent_window_counters_success_count": recent_counters_success_count,
        "sketch_active_verbose_success_count": sketch_verbose_success_count,
        "sketch_active_counters_success_count": sketch_counters_success_count,
        "output_changed_count": output_changed_count,
        "verbose_event_record_count": verbose_event_record_count_total,
        "counter_event_count": counter_event_count_total,
        **comparisons,
        "logging_overhead_reduction": {
            "recent_window_latency_improvement_from_counters": (
                None
                if recent_verbose_mean is None
                or recent_counters_mean is None
                or recent_verbose_mean <= 0
                else (recent_verbose_mean - recent_counters_mean) / recent_verbose_mean
            ),
            "sketch_active_latency_improvement_from_counters": (
                None
                if sketch_verbose_mean is None
                or sketch_counters_mean is None
                or sketch_verbose_mean <= 0
                else (sketch_verbose_mean - sketch_counters_mean)
                / sketch_verbose_mean
            ),
        },
        "per_mode": per_mode,
        "run_records": run_records,
        "mode_event_jsonl": {
            BASELINE_MODE: None,
            RECENT_WINDOW_VERBOSE_MODE: per_mode[RECENT_WINDOW_VERBOSE_MODE].get(
                "events_jsonl"
            ),
            RECENT_WINDOW_COUNTERS_MODE: None,
            SKETCH_ACTIVE_VERBOSE_MODE: per_mode[SKETCH_ACTIVE_VERBOSE_MODE].get(
                "events_jsonl"
            ),
            SKETCH_ACTIVE_COUNTERS_MODE: None,
        },
        "events_jsonl": str(events_root),
        "measured_runtime_reduction": measured_runtime_reduction,
        "memory_claim_allowed": False,
        "quality_claim_allowed": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "caveats": [
            "Verbose modes write per-event JSONL; counters modes only maintain in-memory aggregates.",
            "Counters mode suppresses block sketch samples and other verbose JSONL payloads.",
            "This measures logging overhead and source-level active-path overhead only.",
            "It does not prove memory reduction, quality preservation, or selected attention.",
            "Counters are only accessible when the engine core stays in-process.",
        ],
        "passed": bool(passed),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S4.1 Low-Overhead Measurement",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Repeats: `{report['repeats']}`",
        f"- Warmup: `{report['warmup']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        (
            "- Recent-window verbose success count: "
            f"`{report['recent_window_verbose_success_count']}`"
        ),
        (
            "- Recent-window counters success count: "
            f"`{report['recent_window_counters_success_count']}`"
        ),
        (
            "- Sketch-active verbose success count: "
            f"`{report['sketch_active_verbose_success_count']}`"
        ),
        (
            "- Sketch-active counters success count: "
            f"`{report['sketch_active_counters_success_count']}`"
        ),
        f"- Output changed count: `{report['output_changed_count']}`",
        (
            "- Baseline / recent-window verbose latency ratio: "
            f"`{report['recent_window_verbose_vs_baseline_latency_ratio']}`"
        ),
        (
            "- Baseline / recent-window counters latency ratio: "
            f"`{report['recent_window_counters_vs_baseline_latency_ratio']}`"
        ),
        (
            "- Recent-window counters / verbose latency ratio: "
            f"`{report['recent_window_counters_vs_verbose_latency_ratio']}`"
        ),
        (
            "- Baseline / sketch-active verbose latency ratio: "
            f"`{report['sketch_active_verbose_vs_baseline_latency_ratio']}`"
        ),
        (
            "- Baseline / sketch-active counters latency ratio: "
            f"`{report['sketch_active_counters_vs_baseline_latency_ratio']}`"
        ),
        (
            "- Sketch-active counters / verbose latency ratio: "
            f"`{report['sketch_active_counters_vs_verbose_latency_ratio']}`"
        ),
        f"- Measured runtime reduction: `{report['measured_runtime_reduction']}`",
        "",
        "## Per-Mode Summary",
        "",
        "| mode | success | mean latency (s) | mean tokens/sec | verbose records | counter events | output drift vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODE_ORDER:
        mode_report = report["per_mode"][mode]
        lines.append(
            "| "
            f"{mode} | {mode_report['success_count']} | "
            f"{mode_report['mean_latency_seconds']} | "
            f"{mode_report['mean_tokens_per_second']} | "
            f"{mode_report['verbose_event_record_count']} | "
            f"{mode_report['counter_event_count']} | "
            f"{mode_report['output_changed_count_vs_baseline']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Counters mode suppresses verbose JSONL and sample serialization but still exercises the source-level active path.",
            "- Verbose modes remain available as the comparison baseline for logging overhead.",
            "- This phase does not claim memory reduction, quality preservation, or selected-attention behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    _write(path, json.dumps(data, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.force_inproc_engine_core:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        report = build_report(args)
        _write_json(args.output_json, report)
        _write(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "baseline_success_count": report["baseline_success_count"],
                    "recent_window_verbose_success_count": report[
                        "recent_window_verbose_success_count"
                    ],
                    "recent_window_counters_success_count": report[
                        "recent_window_counters_success_count"
                    ],
                    "sketch_active_verbose_success_count": report[
                        "sketch_active_verbose_success_count"
                    ],
                    "sketch_active_counters_success_count": report[
                        "sketch_active_counters_success_count"
                    ],
                    "measured_runtime_reduction": report[
                        "measured_runtime_reduction"
                    ],
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "events_jsonl": args.events_jsonl,
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["passed"] else 1
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "dry_run_only": False,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
