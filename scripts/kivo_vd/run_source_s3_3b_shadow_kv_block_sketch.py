#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S3.3B shadow KV-cache block sketch probe."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1

SCHEMA = "kivo_source_s3_3b_shadow_kv_block_sketch_v1"
POLICY = "shadow_kv_block_sketch"

DEFAULT_PROMPTS = [
    "The quick brown fox",
    (
        "Machine learning systems can process long contexts by reusing "
        "cached key value states across generation steps."
    ),
    (
        "Kivo source experiments now build real KV cache block sketches. "
        "Kivo source experiments now build real KV cache block sketches."
    ),
]

_SOURCE_ENV_KEYS = [
    "KIVO_SOURCE_ENABLE",
    "KIVO_SOURCE_OBSERVE_PATH",
    "KIVO_SOURCE_OBS_PATH",
    "KIVO_SOURCE_POLICY",
    "KIVO_SOURCE_FAIL_CLOSED",
    "KIVO_SOURCE_SKETCH_DIM",
    "KIVO_SOURCE_MAX_SKETCH_BLOCKS",
    "KIVO_SOURCE_BUDGET_RATIO",
    "KIVO_SOURCE_SKETCH_SEED",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase S3.3B shadow KV block sketching."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--sketch-dim", type=int, default=8)
    parser.add_argument("--max-sketch-blocks", type=int, default=4)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch.md"
        ),
    )
    parser.add_argument(
        "--events-jsonl",
        default=(
            "outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch_events.jsonl"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _clear_source_env() -> None:
    for key in _SOURCE_ENV_KEYS:
        os.environ.pop(key, None)


def _set_shadow_env(
    event_path: str | Path,
    *,
    sketch_dim: int,
    max_sketch_blocks: int,
    budget_ratio: float,
    seed: int,
) -> None:
    _clear_source_env()
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_OBSERVE_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_OBS_PATH"] = str(event_path)
    os.environ["KIVO_SOURCE_POLICY"] = POLICY
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"
    os.environ["KIVO_SOURCE_SKETCH_DIM"] = str(sketch_dim)
    os.environ["KIVO_SOURCE_MAX_SKETCH_BLOCKS"] = str(max_sketch_blocks)
    os.environ["KIVO_SOURCE_BUDGET_RATIO"] = str(budget_ratio)
    os.environ["KIVO_SOURCE_SKETCH_SEED"] = str(seed)


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


def filter_s3_3b_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in records if item.get("schema_version") == SCHEMA]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    filtered = filter_s3_3b_records(records)
    return {
        "records_written": len(filtered),
        "raw_records_written": len(records),
        "ignored_non_s3_events": len(records) - len(filtered),
        "sketch_computed_event_count": sum(
            item.get("sketch_computed") is True for item in filtered
        ),
        "sketch_blocked_event_count": sum(
            item.get("sketch_computed") is not True for item in filtered
        ),
        "kv_cache_observed_event_count": sum(
            item.get("kv_cache_present") is True for item in filtered
        ),
        "slot_mapping_observed_event_count": sum(
            item.get("slot_mapping_present") is True for item in filtered
        ),
        "max_candidate_block_count": max(
            (int(item.get("candidate_block_count", 0) or 0) for item in filtered),
            default=0,
        ),
        "max_selected_block_count": max(
            (int(item.get("selected_block_count", 0) or 0) for item in filtered),
            default=0,
        ),
        "max_excluded_block_count": max(
            (int(item.get("excluded_block_count", 0) or 0) for item in filtered),
            default=0,
        ),
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
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return records


def _zero_event_debug(path: Path) -> dict[str, Any]:
    return {
        "env_policy_used": POLICY,
        "observe_path": str(path),
        "file_exists": path.exists(),
        "file_size": path.stat().st_size if path.exists() else 0,
    }


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
            _set_shadow_env(
                prompt_path,
                sketch_dim=args.sketch_dim,
                max_sketch_blocks=args.max_sketch_blocks,
                budget_ratio=args.budget_ratio,
                seed=args.seed,
            )
            shadow = _run_generation_safe(generation_args, generation_fn)
            records = record_loader(prompt_path)
            event_paths.append(prompt_path)
            summary = summarize_records(records)
            output_changed = bool(
                baseline["status"] == "succeeded"
                and shadow["status"] == "succeeded"
                and baseline["output_text"] != shadow["output_text"]
            )
            prompt_results.append(
                {
                    "prompt_index": index,
                    "prompt": prompt,
                    "baseline_status": baseline["status"],
                    "shadow_status": shadow["status"],
                    "baseline_output": baseline["output_text"],
                    "shadow_output": shadow["output_text"],
                    "baseline_error": baseline.get("error"),
                    "shadow_error": shadow.get("error"),
                    "output_changed": output_changed,
                    "events_jsonl": str(prompt_path),
                    "zero_event_debug": (
                        _zero_event_debug(prompt_path)
                        if summary["records_written"] == 0
                        else None
                    ),
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

    all_records = _concat_jsonl(event_paths, events_path)
    summary = summarize_records(all_records)
    total_prompts = len(prompt_results)
    baseline_success_count = sum(
        item["baseline_status"] == "succeeded" for item in prompt_results
    )
    shadow_success_count = sum(
        item["shadow_status"] == "succeeded" for item in prompt_results
    )
    output_changed_count = sum(
        item["output_changed"] for item in prompt_results
    )
    passed = bool(
        total_prompts > 0
        and baseline_success_count == total_prompts
        and shadow_success_count == total_prompts
        and output_changed_count == 0
        and summary["records_written"] > 0
        and summary["sketch_computed_event_count"] > 0
    )
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "sketch_dim": args.sketch_dim,
        "max_sketch_blocks": args.max_sketch_blocks,
        "budget_ratio": args.budget_ratio,
        "total_prompts": total_prompts,
        "baseline_success_count": baseline_success_count,
        "shadow_success_count": shadow_success_count,
        "output_changed_count": output_changed_count,
        "total_raw_events": len(all_records),
        "total_s3_3b_events": summary["records_written"],
        "ignored_non_s3_events": summary["ignored_non_s3_events"],
        "sketch_computed_event_count": summary["sketch_computed_event_count"],
        "sketch_blocked_event_count": summary["sketch_blocked_event_count"],
        "kv_cache_observed_event_count": summary["kv_cache_observed_event_count"],
        "slot_mapping_observed_event_count": (
            summary["slot_mapping_observed_event_count"]
        ),
        "max_candidate_block_count": summary["max_candidate_block_count"],
        "max_selected_block_count": summary["max_selected_block_count"],
        "max_excluded_block_count": summary["max_excluded_block_count"],
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_3b_shadow_kv_block_sketch_passed": passed,
        "prompt_results": prompt_results,
        "events_jsonl": str(events_path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.3B Shadow KV Block Sketch",
        "",
        f"- Passed: `{report['s3_3b_shadow_kv_block_sketch_passed']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        f"- Shadow success count: `{report['shadow_success_count']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        f"- Total raw events: `{report['total_raw_events']}`",
        f"- Total S3.3B events: `{report['total_s3_3b_events']}`",
        f"- Sketch computed events: `{report['sketch_computed_event_count']}`",
        f"- Sketch blocked events: `{report['sketch_blocked_event_count']}`",
        f"- Max candidate block count: `{report['max_candidate_block_count']}`",
        f"- Max selected block count: `{report['max_selected_block_count']}`",
        f"- Max excluded block count: `{report['max_excluded_block_count']}`",
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
                f"- Shadow status: `{item['shadow_status']}`",
                f"- Output changed: `{item['output_changed']}`",
                f"- S3.3B records: `{item['records_written']}`",
                f"- Sketch computed events: `{item['sketch_computed_event_count']}`",
                f"- Sketch blocked events: `{item['sketch_blocked_event_count']}`",
                f"- Zero-event debug: `{item['zero_event_debug']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- This phase computes real tiny KV-cache block sketches in shadow mode.",
            "- It does not mutate attention metadata, KV cache, or outputs.",
            (
                "- It does not support memory, latency, quality, or "
                "selected-attention claims."
            ),
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
                "passed": report["s3_3b_shadow_kv_block_sketch_passed"],
                "baseline_success_count": report["baseline_success_count"],
                "shadow_success_count": report["shadow_success_count"],
                "output_changed_count": report["output_changed_count"],
                "total_s3_3b_events": report["total_s3_3b_events"],
                "sketch_computed_event_count": (
                    report["sketch_computed_event_count"]
                ),
                "output_json": args.output_json,
                "output_md": args.output_md,
                "events_jsonl": args.events_jsonl,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["s3_3b_shadow_kv_block_sketch_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
