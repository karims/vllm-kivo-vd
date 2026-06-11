#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S3.2B active recent-window attention metadata probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.kivo_vd import (
    run_source_s3_1b_shadow_sketch_selected_attention_metadata as shadow_runner,
)

SCHEMA = "kivo_source_s3_2b_active_recent_window_attention_metadata_v1"
POLICY_NAME = "active_recent_window_attention_metadata"
ACTIVE_FILTER_MODE = "compact_to_recent_window"
DEFAULT_PROMPTS = [
    "The quick brown fox",
    (
        "Machine learning systems can process long contexts by reusing "
        "cached key value states across generation steps."
    ),
    (
        "Kivo source experiments can actively compact recent-window "
        "attention metadata. Kivo source experiments can actively compact "
        "recent-window attention metadata."
    ),
    (
        "Longer context validation helps test whether a cloned block table "
        "and a reduced sequence length can travel through the existing vLLM "
        "attention metadata path without custom kernels. This prompt repeats "
        "the same idea in slightly varied wording so the token count grows "
        "past a few logical cache blocks while remaining comfortably inside "
        "the configured model length budget for a controlled recent-window "
        "compaction experiment."
    ),
]

_SOURCE_ENV_KEYS = [
    "KIVO_SOURCE_ENABLE",
    "KIVO_SOURCE_OBSERVE_PATH",
    "KIVO_SOURCE_OBS_PATH",
    "KIVO_SOURCE_POLICY",
    "KIVO_SOURCE_ACTIVE",
    "KIVO_SOURCE_FAIL_CLOSED",
    "KIVO_SOURCE_KEEP_RECENT_BLOCKS",
    "KIVO_SOURCE_ACTIVE_FILTER_MODE",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase S3.2B active recent-window metadata compaction."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--keep-recent-blocks", type=int, default=1)
    parser.add_argument(
        "--active-filter-mode",
        default=ACTIVE_FILTER_MODE,
        choices=[ACTIVE_FILTER_MODE],
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata.md",
    )
    parser.add_argument(
        "--events-jsonl",
        default="outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata_events.jsonl",
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


def _set_active_env(event_path: str | Path, args: argparse.Namespace) -> None:
    _clear_source_env()
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_OBSERVE_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_OBS_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_POLICY"] = POLICY_NAME
    os.environ["KIVO_SOURCE_ACTIVE"] = "1"
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"
    os.environ["KIVO_SOURCE_KEEP_RECENT_BLOCKS"] = str(args.keep_recent_blocks)
    os.environ["KIVO_SOURCE_ACTIVE_FILTER_MODE"] = args.active_filter_mode


def _event_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    total_raw = len(records)
    total_s3 = sum(record.get("schema_version") == SCHEMA for record in records)
    return {
        "total_raw_events": total_raw,
        "total_s3_2b_events": total_s3,
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
    filtered = [
        record for record in records if record.get("schema_version") == SCHEMA
    ]
    blocker_reasons = sorted({
        str(record["mutation_blocker_reason"])
        for record in filtered
        if record.get("mutation_blocker_reason")
    })
    return {
        "records_written": len(filtered),
        "mutation_attempted_event_count": sum(
            record.get("mutation_attempted") is True for record in filtered
        ),
        "mutation_applied_event_count": sum(
            record.get("mutation_applied") is True for record in filtered
        ),
        "active_routing_event_count": sum(
            record.get("active_routing") is True for record in filtered
        ),
        "blocker_event_count": sum(
            bool(record.get("mutation_blocker_reason")) for record in filtered
        ),
        "blocker_reasons": blocker_reasons,
        "max_original_visible_block_count": _max_int(
            filtered, "original_visible_block_count"
        ),
        "max_selected_block_count": _max_int(filtered, "selected_block_count"),
        "max_excluded_block_count": _max_int(filtered, "excluded_block_count"),
        "max_theoretical_attention_visible_block_reduction": _max_int(
            filtered, "theoretical_attention_visible_block_reduction"
        ),
        "max_theoretical_attention_visible_block_reduction_ratio": max(
            (
                float(
                    record.get(
                        "theoretical_attention_visible_block_reduction_ratio", 0.0
                    )
                    or 0.0
                )
                for record in filtered
            ),
            default=0.0,
        ),
    }


def _concat_jsonl(paths: list[Path], output_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines: list[str] = []
    for path in paths:
        records.extend(shadow_runner._load_records(path))
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
    generation_fn: Any = shadow_runner._run_generation,
    record_loader: Any = shadow_runner._load_records,
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
            generation_args = shadow_runner._build_generation_args(args, prompt)
            _clear_source_env()
            baseline = shadow_runner._run_generation_safe(
                generation_args, generation_fn
            )

            prompt_events_path = events_path.with_name(
                f"{events_path.stem}_prompt_{prompt_index:02d}.jsonl"
            )
            if prompt_events_path.exists():
                prompt_events_path.unlink()
            _set_active_env(prompt_events_path, args)
            active = shadow_runner._run_generation_safe(
                generation_args, generation_fn
            )
            records = record_loader(prompt_events_path)
            all_event_paths.append(prompt_events_path)
            summary = summarize_records(records)
            output_changed = bool(
                baseline["status"] == "succeeded"
                and active["status"] == "succeeded"
                and baseline["output_text"] != active["output_text"]
            )
            prompt_results.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "baseline_status": baseline["status"],
                    "active_status": active["status"],
                    "baseline_output": baseline["output_text"],
                    "active_output": active["output_text"],
                    "baseline_error": baseline["error"],
                    "active_error": active["error"],
                    "output_changed": output_changed,
                    "runtime_behavior_changed": output_changed,
                    "events_jsonl": str(prompt_events_path),
                    **summary,
                }
            )
            if (
                not args.continue_on_error
                and (
                    baseline["status"] != "succeeded"
                    or active["status"] != "succeeded"
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
    active_success_count = sum(
        result["active_status"] == "succeeded" for result in prompt_results
    )
    output_changed_count = sum(result["output_changed"] for result in prompt_results)
    mutation_attempted_event_count = sum(
        result["mutation_attempted_event_count"] for result in prompt_results
    )
    mutation_applied_event_count = sum(
        result["mutation_applied_event_count"] for result in prompt_results
    )
    active_routing_event_count = sum(
        result["active_routing_event_count"] for result in prompt_results
    )
    blocker_event_count = sum(
        result["blocker_event_count"] for result in prompt_results
    )
    s3_2b_active_recent_window_passed = bool(
        total_prompts > 0
        and baseline_success_count == total_prompts
        and active_success_count == total_prompts
        and event_counts["total_s3_2b_events"] > 0
        and mutation_attempted_event_count > 0
        and mutation_applied_event_count > 0
        and active_routing_event_count > 0
    )
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "keep_recent_blocks": args.keep_recent_blocks,
        "active_filter_mode": args.active_filter_mode,
        "total_prompts": total_prompts,
        "baseline_success_count": baseline_success_count,
        "active_success_count": active_success_count,
        "output_changed_count": output_changed_count,
        "runtime_behavior_changed_count": output_changed_count,
        "total_events": event_counts["total_raw_events"],
        **event_counts,
        "mutation_attempted_event_count": mutation_attempted_event_count,
        "mutation_applied_event_count": mutation_applied_event_count,
        "active_routing_event_count": active_routing_event_count,
        "blocker_event_count": blocker_event_count,
        "max_original_visible_block_count": max(
            (result["max_original_visible_block_count"] for result in prompt_results),
            default=0,
        ),
        "max_selected_block_count": max(
            (result["max_selected_block_count"] for result in prompt_results),
            default=0,
        ),
        "max_excluded_block_count": max(
            (result["max_excluded_block_count"] for result in prompt_results),
            default=0,
        ),
        "max_theoretical_attention_visible_block_reduction": max(
            (
                result["max_theoretical_attention_visible_block_reduction"]
                for result in prompt_results
            ),
            default=0,
        ),
        "max_theoretical_attention_visible_block_reduction_ratio": max(
            (
                result[
                    "max_theoretical_attention_visible_block_reduction_ratio"
                ]
                for result in prompt_results
            ),
            default=0.0,
        ),
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_2b_active_recent_window_passed": s3_2b_active_recent_window_passed,
        "prompt_results": prompt_results,
        "events_jsonl": str(events_path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.2B Active Recent-Window Metadata",
        "",
        f"- Passed: `{report['s3_2b_active_recent_window_passed']}`",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        f"- Active success count: `{report['active_success_count']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        (
            "- Runtime behavior changed count: "
            f"`{report['runtime_behavior_changed_count']}`"
        ),
        f"- Total raw events: `{report['total_raw_events']}`",
        f"- Total S3.2B events: `{report['total_s3_2b_events']}`",
        f"- Mutation attempted events: `{report['mutation_attempted_event_count']}`",
        f"- Mutation applied events: `{report['mutation_applied_event_count']}`",
        f"- Active routing events: `{report['active_routing_event_count']}`",
        f"- Blocker events: `{report['blocker_event_count']}`",
        f"- Max original visible block count: `{report['max_original_visible_block_count']}`",
        f"- Max selected block count: `{report['max_selected_block_count']}`",
        f"- Max excluded block count: `{report['max_excluded_block_count']}`",
        "- Measured runtime reduction: `false`",
        "",
        "## Prompt Results",
        "",
    ]
    for item in report["prompt_results"]:
        lines.extend(
            [
                f"### Prompt {item['prompt_index']}",
                "",
                f"- Baseline status: `{item['baseline_status']}`",
                f"- Active status: `{item['active_status']}`",
                f"- Output changed: `{item['output_changed']}`",
                f"- Mutation applied events: `{item['mutation_applied_event_count']}`",
                f"- Blocker reasons: `{item['blocker_reasons']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- This actively compacts to a recent contiguous window in cloned metadata.",
            "- It does not mutate scheduler-owned block tables or KV allocation.",
            "- It does not prove memory reduction or latency improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    shadow_runner._write(args.output_json, json.dumps(report, indent=2) + "\n")
    shadow_runner._write(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "passed": report["s3_2b_active_recent_window_passed"],
                "baseline_success_count": report["baseline_success_count"],
                "active_success_count": report["active_success_count"],
                "output_changed_count": report["output_changed_count"],
                "total_s3_2b_events": report["total_s3_2b_events"],
                "mutation_applied_event_count": (
                    report["mutation_applied_event_count"]
                ),
                "output_json": args.output_json,
                "output_md": args.output_md,
                "events_jsonl": args.events_jsonl,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["s3_2b_active_recent_window_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
