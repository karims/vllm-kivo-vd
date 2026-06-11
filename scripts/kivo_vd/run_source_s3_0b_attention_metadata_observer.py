#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S3.0B attention metadata observer probe."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1

DEFAULT_PROMPTS = [
    "The quick brown fox",
    (
        "Machine learning systems can process long contexts by reusing "
        "cached key value states across generation steps."
    ),
    (
        "Kivo source experiments can observe attention metadata without "
        "changing runtime behavior. Kivo source experiments can observe "
        "attention metadata without changing runtime behavior."
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
        description="Run Phase S3.0B attention metadata observation."
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
        default="outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer.md",
    )
    parser.add_argument(
        "--events-jsonl",
        default="outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer_events.jsonl",
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
    os.environ["KIVO_SOURCE_POLICY"] = "observe_attention_metadata"
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"


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


def _max_shape_component(
    records: list[dict[str, Any]],
    field: str,
    axis: int,
) -> int:
    values = []
    for record in records:
        shape = record.get(field)
        if isinstance(shape, list) and len(shape) > axis:
            try:
                values.append(int(shape[axis]))
            except (TypeError, ValueError):
                continue
    return max(values, default=0)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records_written": len(records),
        "max_block_table_rows": _max_shape_component(
            records, "block_table_tensor_shape", 0
        ),
        "max_block_table_cols": _max_shape_component(
            records, "block_table_tensor_shape", 1
        ),
        "max_slot_mapping_len": _max_shape_component(
            records, "slot_mapping_shape", 0
        ),
        "mutation_attempted": any(
            item.get("mutation_attempted") is True for item in records
        ),
        "mutation_applied": any(
            item.get("mutation_applied") is True for item in records
        ),
        "active_routing": any(item.get("active_routing") is True for item in records),
        "runtime_behavior_changed": any(
            item.get("runtime_behavior_changed") is True for item in records
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
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
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
    stop_early = False

    events_path = Path(args.events_jsonl)
    for prompt_index, prompt in enumerate(prompts):
        generation_args = _build_generation_args(args, prompt)
        _clear_source_env()
        baseline = _run_generation_safe(generation_args, generation_fn)

        prompt_events_path = events_path.with_name(
            f"{events_path.stem}_prompt_{prompt_index:02d}.jsonl"
        )
        if prompt_events_path.exists():
            prompt_events_path.unlink()
        _set_observer_env(prompt_events_path)
        observer = _run_generation_safe(generation_args, generation_fn)
        records = record_loader(prompt_events_path)
        all_event_paths.append(prompt_events_path)
        summary = summarize_records(records)
        output_changed = bool(
            baseline["status"] == "succeeded"
            and observer["status"] == "succeeded"
            and baseline["output_text"] != observer["output_text"]
        )
        prompt_results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "baseline_status": baseline["status"],
                "observer_status": observer["status"],
                "baseline_output": baseline["output_text"],
                "observer_output": observer["output_text"],
                "baseline_error": baseline["error"],
                "observer_error": observer["error"],
                "output_changed": output_changed,
                "events_jsonl": str(prompt_events_path),
                **summary,
            }
        )
        if (
            not args.continue_on_error
            and (
                baseline["status"] != "succeeded"
                or observer["status"] != "succeeded"
            )
        ):
            stop_early = True
            break

    _clear_source_env()
    all_records = _concat_jsonl(all_event_paths, events_path)
    total_prompts = len(prompt_results)
    baseline_success_count = sum(
        result["baseline_status"] == "succeeded" for result in prompt_results
    )
    observer_success_count = sum(
        result["observer_status"] == "succeeded" for result in prompt_results
    )
    output_changed_count = sum(
        result["output_changed"] for result in prompt_results
    )
    metadata_observed_prompt_count = sum(
        result["records_written"] > 0 for result in prompt_results
    )
    max_block_table_rows = max(
        (result["max_block_table_rows"] for result in prompt_results),
        default=0,
    )
    max_block_table_cols = max(
        (result["max_block_table_cols"] for result in prompt_results),
        default=0,
    )
    max_slot_mapping_len = max(
        (result["max_slot_mapping_len"] for result in prompt_results),
        default=0,
    )
    s3_0b_observer_passed = bool(
        total_prompts > 0
        and baseline_success_count == total_prompts
        and observer_success_count == total_prompts
        and output_changed_count == 0
        and len(all_records) > 0
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
        "total_events": len(all_records),
        "metadata_observed_prompt_count": metadata_observed_prompt_count,
        "max_block_table_rows": max_block_table_rows,
        "max_block_table_cols": max_block_table_cols,
        "max_slot_mapping_len": max_slot_mapping_len,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "s3_0b_observer_passed": s3_0b_observer_passed,
        "prompt_results": prompt_results,
        "events_jsonl": str(events_path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S3.0B Attention Metadata Observer",
        "",
        f"- Passed: `{report['s3_0b_observer_passed']}`",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        f"- Observer success count: `{report['observer_success_count']}`",
        f"- Output changed count: `{report['output_changed_count']}`",
        f"- Total events: `{report['total_events']}`",
        (
            "- Metadata observed prompt count: "
            f"`{report['metadata_observed_prompt_count']}`"
        ),
        f"- Max block table rows: `{report['max_block_table_rows']}`",
        f"- Max block table cols: `{report['max_block_table_cols']}`",
        f"- Max slot mapping len: `{report['max_slot_mapping_len']}`",
        "- Measured runtime reduction: `false`",
        "- Selected attention claim allowed: `false`",
        "- Performance claim allowed: `false`",
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
                f"- Observer status: `{item['observer_status']}`",
                f"- Output changed: `{item['output_changed']}`",
                f"- Records written: `{item['records_written']}`",
                f"- Max block table rows: `{item['max_block_table_rows']}`",
                f"- Max block table cols: `{item['max_block_table_cols']}`",
                f"- Max slot mapping len: `{item['max_slot_mapping_len']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- This is an observation-only hook at the metadata boundary.",
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
                "passed": report["s3_0b_observer_passed"],
                "baseline_success_count": report["baseline_success_count"],
                "observer_success_count": report["observer_success_count"],
                "output_changed_count": report["output_changed_count"],
                "total_events": report["total_events"],
                "output_json": args.output_json,
                "output_md": args.output_md,
                "events_jsonl": args.events_jsonl,
            },
            separators=(",", ":"),
        )
    )
    failed = report["baseline_success_count"] != report["total_prompts"]
    failed = failed or report["observer_success_count"] != report["total_prompts"]
    failed = failed or report["output_changed_count"] != 0
    failed = failed or report["total_events"] <= 0
    if not args.continue_on_error:
        for result in report["prompt_results"]:
            if (
                result["baseline_status"] != "succeeded"
                or result["observer_status"] != "succeeded"
            ):
                failed = True
                break
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
