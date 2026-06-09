#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase 12.10 BlockTable slot-mapping mutation probe."""

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
        description="Run baseline and active BlockTable mutation generation."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--prompt",
        default="Kivo Phase 12.10 block table mutation probe.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--baseline-obs-jsonl",
        default="outputs/kivo_vd/runs/phase12_10_baseline.jsonl",
    )
    parser.add_argument(
        "--active-obs-jsonl",
        default="outputs/kivo_vd/runs/phase12_10_active.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/phase12_10_probe.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/phase12_10_probe.md",
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
        "observations_written": len(records),
        "mutation_attempted": any(
            item.get("mutation_attempted") is True for item in records
        ),
        "mutation_applied": any(
            item.get("mutation_applied") is True for item in records
        ),
        "mutation_policy": record.get("mutation_policy"),
        "blocker_reason": record.get("blocker_reason"),
        "tensor_like_result_found": any(
            item.get("tensor_like_result_found") is True
            for item in records
        ),
        "python_mutable_result_found": any(
            item.get("python_mutable_result_found") is True
            for item in records
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
    observation_path: str,
    *,
    active: bool,
    generation_fn: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(observation_path)
    if path.exists():
        path.unlink()
    os.environ["KIVO_PHASE12_10_ENABLE"] = "1"
    os.environ["KIVO_PHASE12_10_OBS_PATH"] = str(path)
    if active:
        os.environ["KIVO_PHASE12_10_ACTIVE"] = "1"
    else:
        os.environ.pop("KIVO_PHASE12_10_ACTIVE", None)
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
        args,
        args.baseline_obs_jsonl,
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
        "baseline_generation_status": baseline["status"],
        "baseline_output_text": baseline["output_text"],
        "baseline_observations_written": baseline_summary[
            "observations_written"
        ],
        "active_generation_status": active["status"],
        "active_output_text": active["output_text"],
        "active_error": active["error"],
        "active_observations_written": active_summary[
            "observations_written"
        ],
        "mutation_attempted": active_summary["mutation_attempted"],
        "mutation_applied": active_summary["mutation_applied"],
        "mutation_policy": active_summary["mutation_policy"],
        "blocker_reason": active_summary["blocker_reason"],
        "tensor_like_result_found": active_summary[
            "tensor_like_result_found"
        ],
        "python_mutable_result_found": active_summary[
            "python_mutable_result_found"
        ],
        "output_changed": output_changed,
        "active_routing": active_summary["mutation_applied"],
        "runtime_behavior_changed": active_summary["mutation_applied"],
        "measured_runtime_reduction": False,
        "phase13_selected_attention_candidate": bool(
            active_summary["mutation_applied"]
            and active["status"] == "succeeded"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Kivo-VD Phase 12.10 BlockTable Mutation Probe",
        "",
        "## Baseline",
        "",
        f"- Status: `{report['baseline_generation_status']}`",
        f"- Output: `{report['baseline_output_text']}`",
        (
            "- Observations written: "
            f"`{report['baseline_observations_written']}`"
        ),
        "",
        "## Active",
        "",
        f"- Status: `{report['active_generation_status']}`",
        f"- Output: `{report['active_output_text']}`",
        f"- Error: `{report['active_error']}`",
        (
            "- Observations written: "
            f"`{report['active_observations_written']}`"
        ),
        f"- Mutation attempted: `{report['mutation_attempted']}`",
        f"- Mutation applied: `{report['mutation_applied']}`",
        f"- Mutation policy: `{report['mutation_policy']}`",
        f"- Blocker reason: `{report['blocker_reason']}`",
        (
            "- Tensor-like result found: "
            f"`{report['tensor_like_result_found']}`"
        ),
        (
            "- Python mutable result found: "
            f"`{report['python_mutable_result_found']}`"
        ),
        f"- Output changed: `{report['output_changed']}`",
        "",
        "## Boundary",
        "",
        f"- Active routing: `{report['active_routing']}`",
        (
            "- Runtime behavior changed: "
            f"`{report['runtime_behavior_changed']}`"
        ),
        "- Measured runtime reduction: `false`",
        (
            "- Phase 13 selected-attention candidate: "
            f"`{report['phase13_selected_attention_candidate']}`"
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
        "baseline_generation_status": report["baseline_generation_status"],
        "active_generation_status": report["active_generation_status"],
        "mutation_applied": report["mutation_applied"],
        "phase13_selected_attention_candidate": report[
            "phase13_selected_attention_candidate"
        ],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    failed = report["baseline_generation_status"] != "succeeded"
    if not args.continue_on_error:
        failed = failed or report["active_generation_status"] == "failed"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
