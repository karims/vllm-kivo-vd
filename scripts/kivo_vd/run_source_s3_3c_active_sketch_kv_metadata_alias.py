#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run Phase S3.3C active sketch-driven metadata aliasing."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1

PLAN_SCHEMA = "kivo_source_s3_3c_active_sketch_plan_v1"
METADATA_SCHEMA = "kivo_source_s3_3c_active_sketch_metadata_alias_v1"
POLICY = "active_sketch_kv_metadata_alias"
ACTIVE_FILTER_MODE = "alias_excluded_blocks_to_sketch_selected"

DEFAULT_PROMPTS = [
    "The quick brown fox",
    (
        "Machine learning systems can process long contexts by reusing "
        "cached key value states across generation steps."
    ),
    (
        "Kivo source experiments now use real KV cache block sketches to "
        "drive active metadata behavior. Kivo source experiments now use "
        "real KV cache block sketches to drive active metadata behavior."
    ),
    (
        "Longer source-level Kivo validation prompt. "
        "This prompt repeats enough material to make several decode steps "
        "touch a wider range of visible context blocks while still staying "
        "well within the conservative test budget for GPT-2. "
        "Longer source-level Kivo validation prompt. "
        "This prompt repeats enough material to make several decode steps "
        "touch a wider range of visible context blocks while still staying "
        "well within the conservative test budget for GPT-2."
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
    "KIVO_SOURCE_ACTIVE_FILTER_MODE",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase S3.3C active sketch metadata aliasing."
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
        "--active-filter-mode",
        default=ACTIVE_FILTER_MODE,
    )
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias.md"
        ),
    )
    parser.add_argument(
        "--events-jsonl",
        default=(
            "outputs/kivo_vd/runs/"
            "source_s3_3c_active_sketch_kv_metadata_alias_events.jsonl"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _clear_source_env() -> None:
    for key in _SOURCE_ENV_KEYS:
        os.environ.pop(key, None)


def _set_active_env(
    event_path: str | Path,
    *,
    sketch_dim: int,
    max_sketch_blocks: int,
    budget_ratio: float,
    seed: int,
    active_filter_mode: str,
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
    os.environ["KIVO_SOURCE_ACTIVE_FILTER_MODE"] = str(active_filter_mode)


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


def filter_plan_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in records if item.get("schema_version") == PLAN_SCHEMA]


def filter_metadata_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in records if item.get("schema_version") == METADATA_SCHEMA]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    plan_records = filter_plan_records(records)
    metadata_records = filter_metadata_records(records)
    return {
        "records_written": len(plan_records) + len(metadata_records),
        "total_s3_3c_sketch_plan_events": len(plan_records),
        "total_s3_3c_metadata_alias_events": len(metadata_records),
        "ignored_non_s3_events": (
            len(records) - len(plan_records) - len(metadata_records)
        ),
        "sketch_computed_event_count": sum(
            item.get("sketch_computed") is True for item in plan_records
        ),
        "sketch_plan_used_event_count": sum(
            item.get("sketch_plan_used") is True for item in metadata_records
        ),
        "sketch_plan_blocked_event_count": sum(
            item.get("sketch_plan_used") is not True for item in metadata_records
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
        "runtime_behavior_changed_count": sum(
            item.get("runtime_behavior_changed") is True for item in metadata_records
        ),
        "blocker_event_count": sum(
            bool(item.get("sketch_plan_blocker_reason"))
            or bool(item.get("mutation_blocker_reason"))
            for item in metadata_records
        ),
        "max_candidate_block_count": max(
            (int(item.get("candidate_block_count", 0) or 0) for item in plan_records),
            default=0,
        ),
        "max_selected_block_count": max(
            (
                int(item.get("selected_block_count", 0) or 0)
                for item in metadata_records + plan_records
            ),
            default=0,
        ),
        "max_excluded_block_count": max(
            (
                int(item.get("excluded_block_count", 0) or 0)
                for item in metadata_records + plan_records
            ),
            default=0,
        ),
        "max_aliased_block_count": max(
            (int(item.get("aliased_block_count", 0) or 0) for item in metadata_records),
            default=0,
        ),
    }


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
            _set_active_env(
                prompt_path,
                sketch_dim=args.sketch_dim,
                max_sketch_blocks=args.max_sketch_blocks,
                budget_ratio=args.budget_ratio,
                seed=args.seed,
                active_filter_mode=args.active_filter_mode,
            )
            active = _run_generation_safe(generation_args, generation_fn)
            records = record_loader(prompt_path)
            event_paths.append(prompt_path)
            summary = summarize_records(records)
            output_changed = bool(
                baseline["status"] == "succeeded"
                and active["status"] == "succeeded"
                and baseline["output_text"] != active["output_text"]
            )
            prompt_results.append(
                {
                    "prompt_index": index,
                    "prompt": prompt,
                    "baseline_status": baseline["status"],
                    "active_status": active["status"],
                    "baseline_output": baseline["output_text"],
                    "active_output": active["output_text"],
                    "baseline_error": baseline.get("error"),
                    "active_error": active.get("error"),
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
                    or active["status"] != "succeeded"
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
    active_success_count = sum(
        item["active_status"] == "succeeded" for item in prompt_results
    )
    output_changed_count = sum(
        item["output_changed"] for item in prompt_results
    )
    passed = bool(
        total_prompts > 0
        and baseline_success_count == total_prompts
        and active_success_count == total_prompts
        and summary["sketch_computed_event_count"] > 0
        and summary["total_s3_3c_metadata_alias_events"] > 0
        and summary["sketch_plan_used_event_count"] > 0
        and summary["mutation_attempted_event_count"] > 0
        and summary["mutation_applied_event_count"] > 0
        and summary["active_routing_event_count"] > 0
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
        "active_filter_mode": args.active_filter_mode,
        "total_prompts": total_prompts,
        "baseline_success_count": baseline_success_count,
        "active_success_count": active_success_count,
        "output_changed_count": output_changed_count,
        "runtime_behavior_changed_count": summary["runtime_behavior_changed_count"],
        "total_raw_events": len(all_records),
        **summary,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_3c_active_sketch_metadata_alias_passed": passed,
        "prompt_results": prompt_results,
        "events_jsonl": str(events_path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.3C Active Sketch KV Metadata Alias",
        "",
        f"- Passed: `{report['s3_3c_active_sketch_metadata_alias_passed']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        f"- Active success count: `{report['active_success_count']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        (
            "- Sketch plan events: "
            f"`{report['total_s3_3c_sketch_plan_events']}`"
        ),
        (
            "- Metadata alias events: "
            f"`{report['total_s3_3c_metadata_alias_events']}`"
        ),
        (
            "- Mutation applied events: "
            f"`{report['mutation_applied_event_count']}`"
        ),
        (
            "- Active routing events: "
            f"`{report['active_routing_event_count']}`"
        ),
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
                (
                    "- Sketch plan used events: "
                    f"`{item['sketch_plan_used_event_count']}`"
                ),
                (
                    "- Mutation applied events: "
                    f"`{item['mutation_applied_event_count']}`"
                ),
                f"- Zero-event debug: `{item['zero_event_debug']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- This is the first active sketch-controlled metadata phase.",
            "- It does not reduce KV allocation or prove latency improvement.",
            "- It does not prove quality preservation or final selected attention.",
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
                "passed": report["s3_3c_active_sketch_metadata_alias_passed"],
                "baseline_success_count": report["baseline_success_count"],
                "active_success_count": report["active_success_count"],
                "output_changed_count": report["output_changed_count"],
                "mutation_applied_event_count": (
                    report["mutation_applied_event_count"]
                ),
                "active_routing_event_count": report["active_routing_event_count"],
                "output_json": args.output_json,
                "output_md": args.output_md,
                "events_jsonl": args.events_jsonl,
            },
            separators=(",", ":"),
        )
    )
    return 0 if report["s3_3c_active_sketch_metadata_alias_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
