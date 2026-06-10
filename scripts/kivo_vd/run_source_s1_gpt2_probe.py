#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S1 source-level GPT-2 mutation probe."""

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
        description="Run source-level baseline, observation, and active probes."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--prompt",
        default="Kivo Phase S1 source-level block table probe.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--baseline-obs-jsonl",
        default="outputs/kivo_vd/runs/source_s1_baseline.jsonl",
    )
    parser.add_argument(
        "--observation-obs-jsonl",
        default="outputs/kivo_vd/runs/source_s1_observation.jsonl",
    )
    parser.add_argument(
        "--active-obs-jsonl",
        default="outputs/kivo_vd/runs/source_s1_active.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s1_probe.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s1_probe.md",
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
    record = records[0] if records else {}
    return {
        "records_written": len(records),
        "mutation_attempted": any(
            item.get("mutation_attempted") is True for item in records
        ),
        "mutation_applied": any(
            item.get("mutation_applied") is True for item in records
        ),
        "mutation_blocker_reason": record.get("mutation_blocker_reason"),
        "old_value": record.get("old_value"),
        "new_value": record.get("new_value"),
        "mutation_index": record.get("mutation_index"),
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


def _set_stage_env(observation_path: str, *, active: bool) -> None:
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_OBS_PATH"] = str(observation_path)
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"
    if active:
        os.environ["KIVO_SOURCE_ACTIVE"] = "1"
        os.environ["KIVO_SOURCE_POLICY"] = "mask_last_slot"
        os.environ["KIVO_SOURCE_MAX_MUTATIONS"] = "1"
    else:
        os.environ.pop("KIVO_SOURCE_ACTIVE", None)
        os.environ.pop("KIVO_SOURCE_POLICY", None)
        os.environ.pop("KIVO_SOURCE_MAX_MUTATIONS", None)


def _run_stage(
    args: argparse.Namespace,
    observation_path: str,
    *,
    active: bool,
    generation_fn: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(observation_path)
    if path.exists():
        path.unlink()
    _set_stage_env(str(path), active=active)
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
    os.environ.pop("KIVO_SOURCE_ENABLE", None)
    os.environ.pop("KIVO_SOURCE_OBS_PATH", None)
    os.environ.pop("KIVO_SOURCE_ACTIVE", None)
    os.environ.pop("KIVO_SOURCE_POLICY", None)
    os.environ.pop("KIVO_SOURCE_MAX_MUTATIONS", None)
    os.environ.pop("KIVO_SOURCE_FAIL_CLOSED", None)

    baseline, baseline_summary = _run_stage(
        args,
        args.baseline_obs_jsonl,
        active=False,
        generation_fn=generation_fn,
    )
    observation, observation_summary = _run_stage(
        args,
        args.observation_obs_jsonl,
        active=False,
        generation_fn=generation_fn,
    )
    active, active_summary = _run_stage(
        args,
        args.active_obs_jsonl,
        active=True,
        generation_fn=generation_fn,
    )
    output_changed = bool(
        baseline["status"] == "succeeded"
        and active["status"] == "succeeded"
        and baseline["output_text"] != active["output_text"]
    )
    return {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "baseline_status": baseline["status"],
        "baseline_output": baseline["output_text"],
        "baseline_records_written": baseline_summary["records_written"],
        "observation_status": observation["status"],
        "observation_output": observation["output_text"],
        "observation_error": observation["error"],
        "observation_records_written": observation_summary["records_written"],
        "active_status": active["status"],
        "active_output": active["output_text"],
        "active_error": active["error"],
        "active_records_written": active_summary["records_written"],
        "mutation_attempted": active_summary["mutation_attempted"],
        "mutation_applied": active_summary["mutation_applied"],
        "mutation_blocker_reason": active_summary["mutation_blocker_reason"],
        "old_value": active_summary["old_value"],
        "new_value": active_summary["new_value"],
        "mutation_index": active_summary["mutation_index"],
        "output_changed": output_changed,
        "runtime_behavior_changed": active_summary["mutation_applied"],
        "active_routing": active_summary["mutation_applied"],
        "measured_runtime_reduction": False,
        "source_selected_block_candidate": bool(
            active_summary["mutation_applied"] and active["status"] == "succeeded"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Kivo-VD Phase S1 Source-Level GPT-2 Probe",
        "",
        "## Baseline",
        "",
        f"- Status: `{report['baseline_status']}`",
        f"- Output: `{report['baseline_output']}`",
        (
            "- Records written: "
            f"`{report['baseline_records_written']}`"
        ),
        "",
        "## Observation",
        "",
        f"- Status: `{report['observation_status']}`",
        f"- Output: `{report['observation_output']}`",
        f"- Error: `{report['observation_error']}`",
        (
            "- Records written: "
            f"`{report['observation_records_written']}`"
        ),
        "",
        "## Active",
        "",
        f"- Status: `{report['active_status']}`",
        f"- Output: `{report['active_output']}`",
        f"- Error: `{report['active_error']}`",
        f"- Mutation attempted: `{report['mutation_attempted']}`",
        f"- Mutation applied: `{report['mutation_applied']}`",
        f"- Blocker reason: `{report['mutation_blocker_reason']}`",
        f"- Old value: `{report['old_value']}`",
        f"- New value: `{report['new_value']}`",
        f"- Mutation index: `{report['mutation_index']}`",
        f"- Records written: `{report['active_records_written']}`",
        "",
        "## Boundary",
        "",
        f"- Output changed: `{report['output_changed']}`",
        f"- Runtime behavior changed: `{report['runtime_behavior_changed']}`",
        f"- Active routing: `{report['active_routing']}`",
        "- Measured runtime reduction: `false`",
        (
            "- Source-selected-block candidate: "
            f"`{report['source_selected_block_candidate']}`"
        ),
    ]) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "baseline_status": report["baseline_status"],
        "observation_status": report["observation_status"],
        "active_status": report["active_status"],
        "mutation_applied": report["mutation_applied"],
        "source_selected_block_candidate": report[
            "source_selected_block_candidate"
        ],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    failed = report["baseline_status"] != "succeeded"
    if not args.continue_on_error:
        failed = failed or report["observation_status"] == "failed"
        failed = failed or report["active_status"] == "failed"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
