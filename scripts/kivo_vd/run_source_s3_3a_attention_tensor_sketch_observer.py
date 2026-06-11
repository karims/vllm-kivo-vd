#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S3.3A attention tensor sketch-source observer."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1

SCHEMA = "kivo_source_s3_3a_attention_tensor_sketch_observer_v1"
POLICY = "observe_attention_tensors_for_sketch"

DEFAULT_PROMPTS = [
    "The quick brown fox",
    (
        "Machine learning systems can process long contexts by reusing "
        "cached key value states across generation steps."
    ),
    (
        "Kivo source experiments need real query key value tensor hooks for "
        "sketch construction. Kivo source experiments need real query key "
        "value tensor hooks for sketch construction."
    ),
]

_SOURCE_ENV_KEYS = [
    "KIVO_SOURCE_ENABLE",
    "KIVO_SOURCE_OBSERVE_PATH",
    "KIVO_SOURCE_OBS_PATH",
    "KIVO_SOURCE_POLICY",
    "KIVO_SOURCE_ACTIVE",
    "KIVO_SOURCE_FAIL_CLOSED",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase S3.3A attention tensor observation."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s3_3a_attention_tensor_sketch_observer.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s3_3a_attention_tensor_sketch_observer.md"
        ),
    )
    parser.add_argument(
        "--events-jsonl",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s3_3a_attention_tensor_sketch_observer_events.jsonl"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _clear_source_env() -> None:
    for key in _SOURCE_ENV_KEYS:
        os.environ.pop(key, None)


def _set_observer_env(event_path: str | Path) -> None:
    _clear_source_env()
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_OBSERVE_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_OBS_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_POLICY"] = POLICY
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"


def _capture_source_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _SOURCE_ENV_KEYS}


def _restore_source_env(snapshot: dict[str, str | None]) -> None:
    _clear_source_env()
    for key, value in snapshot.items():
        if value is not None:
            os.environ[key] = value


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


def filter_s3_3a_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [item for item in records if item.get("schema_version") == SCHEMA]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    filtered = filter_s3_3a_records(records)
    hook_points = sorted({
        str(item["hook_point"])
        for item in filtered
        if item.get("hook_point") is not None
    })
    source_counts: dict[str, int] = {}
    for item in filtered:
        source = str(item.get("recommended_sketch_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "records_written": len(filtered),
        "raw_records_written": len(records),
        "ignored_non_s3_events": len(records) - len(filtered),
        "observed_hook_points": hook_points,
        "query_observed_event_count": sum(
            item.get("query_present") is True for item in filtered
        ),
        "key_observed_event_count": sum(
            item.get("key_present") is True for item in filtered
        ),
        "value_observed_event_count": sum(
            item.get("value_present") is True for item in filtered
        ),
        "kv_cache_observed_event_count": sum(
            item.get("kv_cache_present") is True for item in filtered
        ),
        "can_build_query_sketch_event_count": sum(
            item.get("can_build_query_sketch") is True for item in filtered
        ),
        "can_build_key_sketch_event_count": sum(
            item.get("can_build_key_sketch") is True for item in filtered
        ),
        "can_build_value_sketch_event_count": sum(
            item.get("can_build_value_sketch") is True for item in filtered
        ),
        "can_build_kv_block_sketch_event_count": sum(
            item.get("can_build_kv_block_sketch") is True for item in filtered
        ),
        "recommended_sketch_source_counts": source_counts,
    }


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _concat_jsonl(paths: list[Path], output_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines: list[str] = []
    for path in paths:
        records.extend(_load_records(path))
        if path.exists():
            lines.extend(
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return records


def _zero_event_debug(path: Path) -> dict[str, Any]:
    return {
        "env_policy_used": POLICY,
        "observe_path": str(path),
        "file_exists": path.exists(),
        "file_size": path.stat().st_size if path.exists() else 0,
    }


def _merge_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    count_fields = [
        "query_observed_event_count",
        "key_observed_event_count",
        "value_observed_event_count",
        "kv_cache_observed_event_count",
        "can_build_query_sketch_event_count",
        "can_build_key_sketch_event_count",
        "can_build_value_sketch_event_count",
        "can_build_kv_block_sketch_event_count",
    ]
    merged: dict[str, Any] = {
        field: sum(int(item[field]) for item in summaries)
        for field in count_fields
    }
    merged["observed_hook_points"] = sorted({
        hook
        for item in summaries
        for hook in item["observed_hook_points"]
    })
    source_counts: dict[str, int] = {}
    for item in summaries:
        for source, count in item["recommended_sketch_source_counts"].items():
            source_counts[source] = source_counts.get(source, 0) + int(count)
    merged["recommended_sketch_source_counts"] = source_counts
    return merged


def build_report(
    args: argparse.Namespace,
    *,
    generation_fn: Any = _run_generation,
    record_loader: Any = _load_records,
) -> dict[str, Any]:
    prompts = [prompt for prompt in DEFAULT_PROMPTS if prompt.strip()]
    prompt_results: list[dict[str, Any]] = []
    event_paths: list[Path] = []
    previous_env = _capture_source_env()
    events_path = Path(args.events_jsonl)
    if events_path.exists():
        events_path.unlink()

    try:
        for index, prompt in enumerate(prompts):
            generation_args = _build_generation_args(args, prompt)
            _clear_source_env()
            baseline = _run_generation_safe(generation_args, generation_fn)

            prompt_path = events_path.with_name(
                f"{events_path.stem}_prompt_{index:02d}.jsonl"
            )
            if prompt_path.exists():
                prompt_path.unlink()
            _set_observer_env(prompt_path)
            observer = _run_generation_safe(generation_args, generation_fn)
            records = record_loader(prompt_path)
            event_paths.append(prompt_path)
            summary = summarize_records(records)
            output_changed = bool(
                baseline["status"] == "succeeded"
                and observer["status"] == "succeeded"
                and baseline["output_text"] != observer["output_text"]
            )
            prompt_results.append({
                "prompt_index": index,
                "prompt": prompt,
                "baseline_status": baseline["status"],
                "observer_status": observer["status"],
                "baseline_output": baseline["output_text"],
                "observer_output": observer["output_text"],
                "baseline_error": baseline.get("error"),
                "observer_error": observer.get("error"),
                "output_changed": output_changed,
                "events_jsonl": str(prompt_path),
                "zero_event_debug": (
                    _zero_event_debug(prompt_path)
                    if summary["records_written"] == 0
                    else None
                ),
                **summary,
            })
            if (
                not args.continue_on_error
                and (
                    baseline["status"] != "succeeded"
                    or observer["status"] != "succeeded"
                )
            ):
                break
    finally:
        _restore_source_env(previous_env)

    all_records = _concat_jsonl(event_paths, events_path)
    s3_records = filter_s3_3a_records(all_records)
    merged = _merge_summaries([summarize_records(all_records)])
    total_prompts = len(prompt_results)
    baseline_success_count = sum(
        item["baseline_status"] == "succeeded" for item in prompt_results
    )
    observer_success_count = sum(
        item["observer_status"] == "succeeded" for item in prompt_results
    )
    output_changed_count = sum(
        item["output_changed"] for item in prompt_results
    )
    any_tensor_observed = any(
        merged[field] > 0
        for field in [
            "query_observed_event_count",
            "key_observed_event_count",
            "value_observed_event_count",
            "kv_cache_observed_event_count",
        ]
    )
    passed = bool(
        total_prompts > 0
        and baseline_success_count == total_prompts
        and observer_success_count == total_prompts
        and output_changed_count == 0
        and len(s3_records) > 0
        and any_tensor_observed
    )
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "total_prompts": total_prompts,
        "baseline_success_count": baseline_success_count,
        "observer_success_count": observer_success_count,
        "output_changed_count": output_changed_count,
        "total_raw_events": len(all_records),
        "total_s3_3a_events": len(s3_records),
        "ignored_non_s3_events": len(all_records) - len(s3_records),
        **merged,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_3a_attention_tensor_observer_passed": passed,
        "prompt_results": prompt_results,
        "events_jsonl": str(events_path),
        "zero_event_debug": (
            _zero_event_debug(events_path) if not s3_records else None
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.3A Attention Tensor Sketch Observer",
        "",
        f"- Passed: `{report['s3_3a_attention_tensor_observer_passed']}`",
        f"- Baseline successes: `{report['baseline_success_count']}`",
        f"- Observer successes: `{report['observer_success_count']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        f"- Total raw events: `{report['total_raw_events']}`",
        f"- Total S3.3A events: `{report['total_s3_3a_events']}`",
        f"- Ignored non-S3 events: `{report['ignored_non_s3_events']}`",
        f"- Hook points: `{report['observed_hook_points']}`",
        f"- Query observations: `{report['query_observed_event_count']}`",
        f"- Key observations: `{report['key_observed_event_count']}`",
        f"- Value observations: `{report['value_observed_event_count']}`",
        f"- KV-cache observations: `{report['kv_cache_observed_event_count']}`",
        (
            "- Query-sketch-capable events: "
            f"`{report['can_build_query_sketch_event_count']}`"
        ),
        (
            "- Key-sketch-capable events: "
            f"`{report['can_build_key_sketch_event_count']}`"
        ),
        (
            "- KV-block-sketch-capable events: "
            f"`{report['can_build_kv_block_sketch_event_count']}`"
        ),
        (
            "- Recommended sketch sources: "
            f"`{report['recommended_sketch_source_counts']}`"
        ),
        "",
        "## Prompt Results",
        "",
    ]
    for item in report["prompt_results"]:
        lines.extend([
            f"### Prompt {item['prompt_index']}",
            "",
            f"- Baseline status: `{item['baseline_status']}`",
            f"- Observer status: `{item['observer_status']}`",
            f"- Output changed: `{item['output_changed']}`",
            f"- S3.3A records: `{item['records_written']}`",
            f"- Hook points: `{item['observed_hook_points']}`",
            f"- Zero-event debug: `{item['zero_event_debug']}`",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "- This observer records tensor metadata only.",
        "- It does not compute sketches or copy KV-cache contents.",
        "- It does not mutate attention, metadata, slot mappings, or KV cache.",
        (
            "- It does not establish memory, latency, quality, or "
            "selected-attention claims."
        ),
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "passed": report["s3_3a_attention_tensor_observer_passed"],
        "baseline_success_count": report["baseline_success_count"],
        "observer_success_count": report["observer_success_count"],
        "output_changed_count": report["output_changed_count"],
        "total_s3_3a_events": report["total_s3_3a_events"],
        "observed_hook_points": report["observed_hook_points"],
        "output_json": args.output_json,
        "output_md": args.output_md,
        "events_jsonl": args.events_jsonl,
    }, separators=(",", ":")))
    return 0 if report["s3_3a_attention_tensor_observer_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
