#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S3.1B shadow sketch-selected attention metadata probe."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1

SCHEMA = "kivo_source_s3_1b_shadow_sketch_selected_attention_metadata_v1"
POLICY_NAME = "shadow_sketch_selected_attention_metadata"
DEFAULT_PROMPTS = [
    "The quick brown fox",
    (
        "Machine learning systems can process long contexts by reusing "
        "cached key value states across generation steps."
    ),
    (
        "Kivo source experiments can plan sketch selected attention metadata "
        "without changing runtime behavior. Kivo source experiments can plan "
        "sketch selected attention metadata without changing runtime behavior."
    ),
]

_SOURCE_ENV_KEYS = [
    "KIVO_SOURCE_ENABLE",
    "KIVO_SOURCE_OBSERVE_PATH",
    "KIVO_SOURCE_OBS_PATH",
    "KIVO_SOURCE_POLICY",
    "KIVO_SOURCE_ACTIVE",
    "KIVO_SOURCE_FAIL_CLOSED",
    "KIVO_SOURCE_BUDGET_RATIO",
    "KIVO_SOURCE_KEEP_RECENT_BLOCKS",
    "KIVO_SOURCE_COVERAGE_WEIGHT",
    "KIVO_SOURCE_RECENCY_WEIGHT",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase S3.1B shadow sketch-selected attention planning."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--keep-recent-blocks", type=int, default=1)
    parser.add_argument("--coverage-weight", type=float, default=0.6)
    parser.add_argument("--recency-weight", type=float, default=0.4)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata.md",
    )
    parser.add_argument(
        "--events-jsonl",
        default="outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata_events.jsonl",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _clear_source_env() -> None:
    for key in _SOURCE_ENV_KEYS:
        os.environ.pop(key, None)


def _capture_source_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _SOURCE_ENV_KEYS}


def _restore_source_env(env_snapshot: dict[str, str | None]) -> None:
    _clear_source_env()
    for key, value in env_snapshot.items():
        if value is not None:
            os.environ[key] = value


def _set_shadow_env(
    event_path: str | Path,
    *,
    budget_ratio: float,
    keep_recent_blocks: int,
    coverage_weight: float,
    recency_weight: float,
) -> None:
    _clear_source_env()
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_OBSERVE_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_OBS_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_POLICY"] = POLICY_NAME
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"
    os.environ["KIVO_SOURCE_BUDGET_RATIO"] = str(budget_ratio)
    os.environ["KIVO_SOURCE_KEEP_RECENT_BLOCKS"] = str(keep_recent_blocks)
    os.environ["KIVO_SOURCE_COVERAGE_WEIGHT"] = str(coverage_weight)
    os.environ["KIVO_SOURCE_RECENCY_WEIGHT"] = str(recency_weight)


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


def _event_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    total_raw = len(records)
    total_s3 = sum(record.get("schema_version") == SCHEMA for record in records)
    return {
        "total_raw_events": total_raw,
        "total_s3_1b_events": total_s3,
        "ignored_non_s3_events": total_raw - total_s3,
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


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    filtered_records = [
        record for record in records if record.get("schema_version") == SCHEMA
    ]
    return {
        "records_written": len(filtered_records),
        "excluded_event_count": sum(
            int(record.get("excluded_block_count", 0) or 0) > 0
            for record in filtered_records
        ),
        "fallback_event_count": sum(
            record.get("fallback_used") is True for record in filtered_records
        ),
        "max_visible_block_count": _max_int(
            filtered_records, "visible_block_count_estimate"
        ),
        "max_selected_block_count": _max_int(
            filtered_records, "selected_block_count"
        ),
        "max_excluded_block_count": _max_int(
            filtered_records, "excluded_block_count"
        ),
        "max_theoretical_attention_visible_block_reduction": _max_int(
            filtered_records, "theoretical_attention_visible_block_reduction"
        ),
        "max_theoretical_attention_visible_block_reduction_ratio": max(
            (
                float(
                    record.get(
                        "theoretical_attention_visible_block_reduction_ratio", 0.0
                    )
                    or 0.0
                )
                for record in filtered_records
            ),
            default=0.0,
        ),
    }


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _zero_event_debug(event_path: Path) -> dict[str, Any]:
    return {
        "env_policy_used": os.environ.get("KIVO_SOURCE_POLICY"),
        "observe_path": str(event_path),
        "file_exists": event_path.exists(),
        "file_size": event_path.stat().st_size if event_path.exists() else 0,
    }


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
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return records


def build_report(
    args: argparse.Namespace,
    *,
    generation_fn: Any = _run_generation,
    record_loader: Any = _load_records,
) -> dict[str, Any]:
    prompts = [str(prompt) for prompt in DEFAULT_PROMPTS if str(prompt).strip()]
    prompt_results: list[dict[str, Any]] = []
    all_event_paths: list[Path] = []
    previous_env = _capture_source_env()

    events_path = Path(args.events_jsonl)
    if events_path.exists():
        events_path.unlink()
    try:
        for prompt_index, prompt in enumerate(prompts):
            generation_args = _build_generation_args(args, prompt)
            _clear_source_env()
            baseline = _run_generation_safe(generation_args, generation_fn)

            prompt_events_path = events_path.with_name(
                f"{events_path.stem}_prompt_{prompt_index:02d}.jsonl"
            )
            if prompt_events_path.exists():
                prompt_events_path.unlink()
            _set_shadow_env(
                prompt_events_path,
                budget_ratio=args.budget_ratio,
                keep_recent_blocks=args.keep_recent_blocks,
                coverage_weight=args.coverage_weight,
                recency_weight=args.recency_weight,
            )
            shadow = _run_generation_safe(generation_args, generation_fn)
            records = record_loader(prompt_events_path)
            all_event_paths.append(prompt_events_path)
            summary = summarize_records(records)
            output_changed = bool(
                baseline["status"] == "succeeded"
                and shadow["status"] == "succeeded"
                and baseline["output_text"] != shadow["output_text"]
            )
            zero_event_debug = (
                _zero_event_debug(prompt_events_path)
                if summary["records_written"] == 0
                else None
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
                    "events_jsonl": str(prompt_events_path),
                    "zero_event_debug": zero_event_debug,
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
                break
    finally:
        _restore_source_env(previous_env)

    all_records = _concat_jsonl(all_event_paths, events_path)
    event_counts = _event_counts(all_records)
    total_prompts = len(prompt_results)
    baseline_success_count = sum(
        result["baseline_status"] == "succeeded" for result in prompt_results
    )
    shadow_success_count = sum(
        result["shadow_status"] == "succeeded" for result in prompt_results
    )
    output_changed_count = sum(result["output_changed"] for result in prompt_results)
    shadow_plan_prompt_count = sum(
        result["records_written"] > 0 for result in prompt_results
    )
    excluded_block_prompt_count = sum(
        result["excluded_event_count"] > 0 for result in prompt_results
    )
    fallback_event_count = sum(
        result["fallback_event_count"] for result in prompt_results
    )
    max_visible_block_count = max(
        (result["max_visible_block_count"] for result in prompt_results),
        default=0,
    )
    max_selected_block_count = max(
        (result["max_selected_block_count"] for result in prompt_results),
        default=0,
    )
    max_excluded_block_count = max(
        (result["max_excluded_block_count"] for result in prompt_results),
        default=0,
    )
    max_reduction = max(
        (
            result["max_theoretical_attention_visible_block_reduction"]
            for result in prompt_results
        ),
        default=0,
    )
    max_reduction_ratio = max(
        (
            result["max_theoretical_attention_visible_block_reduction_ratio"]
            for result in prompt_results
        ),
        default=0.0,
    )
    s3_1b_shadow_sketch_plan_passed = bool(
        total_prompts > 0
        and baseline_success_count == total_prompts
        and shadow_success_count == total_prompts
        and output_changed_count == 0
        and event_counts["total_s3_1b_events"] > 0
        and shadow_plan_prompt_count > 0
    )
    zero_event_debug = None
    if event_counts["total_s3_1b_events"] == 0:
        zero_event_debug = {
            "env_policy_used": POLICY_NAME,
            "observe_path": str(events_path),
            "file_exists": events_path.exists(),
            "file_size": events_path.stat().st_size if events_path.exists() else 0,
        }
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "budget_ratio": args.budget_ratio,
        "keep_recent_blocks": args.keep_recent_blocks,
        "coverage_weight": args.coverage_weight,
        "recency_weight": args.recency_weight,
        "total_prompts": total_prompts,
        "baseline_success_count": baseline_success_count,
        "shadow_success_count": shadow_success_count,
        "output_changed_count": output_changed_count,
        "total_events": event_counts["total_raw_events"],
        **event_counts,
        "shadow_plan_prompt_count": shadow_plan_prompt_count,
        "excluded_block_prompt_count": excluded_block_prompt_count,
        "fallback_event_count": fallback_event_count,
        "max_visible_block_count": max_visible_block_count,
        "max_selected_block_count": max_selected_block_count,
        "max_excluded_block_count": max_excluded_block_count,
        "max_theoretical_attention_visible_block_reduction": max_reduction,
        "max_theoretical_attention_visible_block_reduction_ratio": (
            max_reduction_ratio
        ),
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_1b_shadow_sketch_plan_passed": s3_1b_shadow_sketch_plan_passed,
        "prompt_results": prompt_results,
        "events_jsonl": str(events_path),
        "zero_event_debug": zero_event_debug,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.1B Shadow Sketch Selected-Attention Metadata",
        "",
        f"- Passed: `{report['s3_1b_shadow_sketch_plan_passed']}`",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        f"- Shadow success count: `{report['shadow_success_count']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        f"- Total raw events: `{report['total_raw_events']}`",
        f"- Total S3.1B events: `{report['total_s3_1b_events']}`",
        f"- Ignored non-S3 events: `{report['ignored_non_s3_events']}`",
        f"- Shadow plan prompt count: `{report['shadow_plan_prompt_count']}`",
        f"- Excluded block prompt count: `{report['excluded_block_prompt_count']}`",
        f"- Fallback event count: `{report['fallback_event_count']}`",
        f"- Max visible block count: `{report['max_visible_block_count']}`",
        f"- Max selected block count: `{report['max_selected_block_count']}`",
        f"- Max excluded block count: `{report['max_excluded_block_count']}`",
        (
            "- Max theoretical attention-visible block reduction: "
            f"`{report['max_theoretical_attention_visible_block_reduction']}`"
        ),
        (
            "- Max theoretical reduction ratio: "
            f"`{report['max_theoretical_attention_visible_block_reduction_ratio']}`"
        ),
        "- Measured runtime reduction: `false`",
        "- Selected attention claim allowed: `false`",
        "- Performance claim allowed: `false`",
        f"- Events JSONL: `{report['events_jsonl']}`",
        "",
        "## Prompt Results",
        "",
    ]
    for item in report["prompt_results"]:
        lines.extend(
            [
                f"### Prompt {item['prompt_index']}",
                "",
                f"- Prompt: `{item['prompt']}`",
                f"- Baseline status: `{item['baseline_status']}`",
                f"- Shadow status: `{item['shadow_status']}`",
                f"- Output changed: `{item['output_changed']}`",
                f"- Records written: `{item['records_written']}`",
                f"- Excluded event count: `{item['excluded_event_count']}`",
                f"- Fallback event count: `{item['fallback_event_count']}`",
                f"- Max visible block count: `{item['max_visible_block_count']}`",
                f"- Max selected block count: `{item['max_selected_block_count']}`",
                f"- Max excluded block count: `{item['max_excluded_block_count']}`",
                (
                    "- Max theoretical reduction: "
                    f"`{item['max_theoretical_attention_visible_block_reduction']}`"
                ),
                (
                    "- Max theoretical reduction ratio: "
                    f"`{item['max_theoretical_attention_visible_block_reduction_ratio']}`"
                ),
                f"- Zero-event debug: `{item['zero_event_debug']}`",
                "",
            ]
        )
    if report.get("zero_event_debug") is not None:
        lines.extend(
            [
                "## Zero Event Debug",
                "",
                f"- Debug: `{report['zero_event_debug']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- This is a shadow planning hook at the metadata boundary.",
            "- It does not mutate block tables, slot mappings, or attention metadata.",
            "- It does not claim memory reduction, latency reduction, or selected attention.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "passed": report["s3_1b_shadow_sketch_plan_passed"],
                "baseline_success_count": report["baseline_success_count"],
                "shadow_success_count": report["shadow_success_count"],
                "output_changed_count": report["output_changed_count"],
                "total_events": report["total_events"],
                "total_raw_events": report["total_raw_events"],
                "total_s3_1b_events": report["total_s3_1b_events"],
                "ignored_non_s3_events": report["ignored_non_s3_events"],
                "shadow_plan_prompt_count": report["shadow_plan_prompt_count"],
                "output_json": args.output_json,
                "output_md": args.output_md,
                "events_jsonl": args.events_jsonl,
            },
            separators=(",", ":"),
        )
    )
    failed = report["baseline_success_count"] != report["total_prompts"]
    failed = failed or report["shadow_success_count"] != report["total_prompts"]
    failed = failed or report["output_changed_count"] != 0
    failed = failed or report["total_s3_1b_events"] <= 0
    failed = failed or report["shadow_plan_prompt_count"] <= 0
    if not args.continue_on_error:
        for result in report["prompt_results"]:
            if (
                result["baseline_status"] != "succeeded"
                or result["shadow_status"] != "succeeded"
            ):
                failed = True
                break
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
