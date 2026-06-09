#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the guarded Phase 12.8/12.9 active mutation ladder."""

from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline, metadata, and selected-slot generations."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--prompt",
        default="Kivo Phase 12.8/12.9 active ladder probe.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--baseline-obs-jsonl",
        default="outputs/kivo_vd/runs/phase12_8_9_baseline.jsonl",
    )
    parser.add_argument(
        "--metadata-obs-jsonl",
        default="outputs/kivo_vd/runs/phase12_8_9_metadata.jsonl",
    )
    parser.add_argument(
        "--selected-slot-obs-jsonl",
        default="outputs/kivo_vd/runs/phase12_8_9_selected_slot.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/phase12_8_9_active_ladder.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/phase12_8_9_active_ladder.md",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    records = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = any(record.get("mutation_attempted") is True for record in records)
    applied = any(record.get("mutation_applied") is True for record in records)
    removed = next(
        (record.get("removed_key") for record in records if record.get("removed_key")),
        None,
    )
    blockers = sorted({
        str(record["blocker_reason"])
        for record in records
        if record.get("blocker_reason")
    })
    return {
        "observation_count": len(records),
        "mutation_attempted": attempted,
        "mutation_applied": applied,
        "removed_key": removed,
        "blocker_reason": (
            None if applied else "; ".join(blockers) if blockers else None
        ),
    }


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
        candidates = getattr(outputs[0], "outputs", None) if outputs else None
        return {
            "status": "succeeded",
            "output_text": str(candidates[0].text) if candidates else "",
            "error": None,
        }
    finally:
        del llm
        gc.collect()


def _run_stage(
    args: argparse.Namespace,
    stage: str,
    observation_path: str,
    generation_fn: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(observation_path)
    if path.exists():
        path.unlink()
    os.environ["KIVO_PHASE12_8_9_ENABLE"] = "1"
    os.environ["KIVO_PHASE12_8_9_STAGE"] = stage
    os.environ["KIVO_PHASE12_8_9_OBS_PATH"] = str(path)
    os.environ["KIVO_PHASE12_8_9_MAX_MUTATIONS"] = "1"
    if stage == "baseline":
        os.environ.pop("KIVO_PHASE12_8_9_ACTIVE", None)
    else:
        os.environ["KIVO_PHASE12_8_9_ACTIVE"] = "1"
    try:
        generation = generation_fn(args)
    except Exception as exc:
        generation = {
            "status": "failed",
            "output_text": None,
            "error": (
                f"{type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()[-4000:]}"
            ),
        }
    return generation, summarize_records(load_records(path))


def build_report(
    args: argparse.Namespace,
    *,
    generation_fn: Any = _run_generation,
) -> dict[str, Any]:
    baseline, baseline_summary = _run_stage(
        args, "baseline", args.baseline_obs_jsonl, generation_fn
    )
    metadata, metadata_summary = _run_stage(
        args, "metadata", args.metadata_obs_jsonl, generation_fn
    )
    selected = {
        "status": "skipped",
        "output_text": None,
        "error": None,
    }
    selected_summary = {
        "observation_count": 0,
        "mutation_attempted": False,
        "mutation_applied": False,
        "removed_key": None,
        "blocker_reason": "metadata stage did not succeed with mutation",
    }
    if metadata["status"] == "succeeded" and metadata_summary["mutation_applied"]:
        selected, selected_summary = _run_stage(
            args,
            "selected_slot",
            args.selected_slot_obs_jsonl,
            generation_fn,
        )

    baseline_text = baseline["output_text"]
    metadata_changed = bool(
        metadata["status"] == "succeeded"
        and metadata["output_text"] != baseline_text
    )
    selected_changed = bool(
        selected["status"] == "succeeded"
        and selected["output_text"] != baseline_text
    )
    selected_candidate = bool(
        selected["status"] == "succeeded"
        and selected_summary["mutation_applied"]
    )
    runtime_changed = bool(
        metadata_summary["mutation_applied"]
        or selected_summary["mutation_applied"]
    )
    return {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "baseline_generation_status": baseline["status"],
        "baseline_output_text": baseline_text,
        "baseline_observation_count": baseline_summary["observation_count"],
        "metadata_generation_status": metadata["status"],
        "metadata_output_text": metadata["output_text"],
        "metadata_error": metadata["error"],
        "metadata_mutation_attempted": metadata_summary["mutation_attempted"],
        "metadata_mutation_applied": metadata_summary["mutation_applied"],
        "metadata_removed_key": metadata_summary["removed_key"],
        "metadata_output_changed": metadata_changed,
        "selected_slot_generation_status": selected["status"],
        "selected_slot_output_text": selected["output_text"],
        "selected_slot_error": selected["error"],
        "selected_slot_mutation_attempted": selected_summary[
            "mutation_attempted"
        ],
        "selected_slot_mutation_applied": selected_summary["mutation_applied"],
        "selected_slot_blocker_reason": selected_summary["blocker_reason"],
        "selected_slot_output_changed": selected_changed,
        "active_routing": selected_summary["mutation_applied"],
        "runtime_behavior_changed": runtime_changed,
        "measured_runtime_reduction": False,
        "phase13_selected_attention_candidate": selected_candidate,
        "caveat": (
            "This is an experimental installed-wheel metadata mutation. "
            "It is not a production selected-attention or memory claim."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Kivo-VD Phase 12.8/12.9 Active Ladder",
        "",
        "## Baseline",
        "",
        f"- Status: `{report['baseline_generation_status']}`",
        f"- Output: `{report['baseline_output_text']}`",
        "",
        "## Metadata Mutation",
        "",
        f"- Status: `{report['metadata_generation_status']}`",
        f"- Attempted: `{report['metadata_mutation_attempted']}`",
        f"- Applied: `{report['metadata_mutation_applied']}`",
        f"- Removed key: `{report['metadata_removed_key']}`",
        f"- Output changed: `{report['metadata_output_changed']}`",
        f"- Error: `{report['metadata_error']}`",
        "",
        "## Selected-Slot Mutation",
        "",
        f"- Status: `{report['selected_slot_generation_status']}`",
        f"- Attempted: `{report['selected_slot_mutation_attempted']}`",
        f"- Applied: `{report['selected_slot_mutation_applied']}`",
        f"- Blocker: `{report['selected_slot_blocker_reason']}`",
        f"- Output changed: `{report['selected_slot_output_changed']}`",
        f"- Error: `{report['selected_slot_error']}`",
        "",
        "## Boundary",
        "",
        f"- Active routing attempted: `{report['active_routing']}`",
        f"- Runtime behavior changed: `{report['runtime_behavior_changed']}`",
        "- Measured runtime reduction: `false`",
        (
            "- Phase 13 selected-attention candidate: "
            f"`{report['phase13_selected_attention_candidate']}`"
        ),
        "",
        report["caveat"],
    ]) + "\n"


def _write(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "baseline_generation_status": report["baseline_generation_status"],
        "metadata_generation_status": report["metadata_generation_status"],
        "selected_slot_generation_status": report[
            "selected_slot_generation_status"
        ],
        "active_routing": report["active_routing"],
        "phase13_selected_attention_candidate": report[
            "phase13_selected_attention_candidate"
        ],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    failed = report["baseline_generation_status"] != "succeeded"
    if not args.continue_on_error:
        failed = failed or report["metadata_generation_status"] == "failed"
        failed = failed or report["selected_slot_generation_status"] == "failed"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
