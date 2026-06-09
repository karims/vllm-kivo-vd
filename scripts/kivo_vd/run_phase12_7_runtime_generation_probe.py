#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run generation against an installed vLLM wheel patched for Phase 12.7."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 12.7 installed-runtime generation probe."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--prompt",
        default="Kivo Phase 12.7 runtime integration probe.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--active", action="store_true")
    parser.add_argument(
        "--observations-jsonl",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_7_runtime_observations.jsonl"
        ),
    )
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_7_runtime_generation_probe.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_7_runtime_generation_probe.md"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def load_observations(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    records = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    active_records = [
        record for record in records if record.get("active_enabled") is True
    ]
    attempted = any(
        record.get("mutation_attempted") is True for record in records
    )
    applied = any(
        record.get("mutation_applied") is True for record in records
    )
    blocked_records = [
        record
        for record in records
        if record.get("active_experiment_blocked") is True
    ]
    blockers = sorted({
        str(record.get("blocker_reason"))
        for record in blocked_records
        if record.get("blocker_reason")
    })
    would_select = [
        record.get("would_select_blocks")
        for record in active_records
        if record.get("would_select_blocks")
    ]
    return {
        "observations_written": len(records),
        "active_decision_records": len(active_records),
        "mutation_attempted": attempted,
        "mutation_applied": applied,
        "active_experiment_blocked": bool(blocked_records),
        "blocker_reason": "; ".join(blockers) if blockers else None,
        "would_select_blocks_preview": would_select[:8],
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
        "error_type": None,
        "error": None,
    }


def build_report(
    args: argparse.Namespace,
    *,
    generation_fn: Any = _run_generation,
) -> dict[str, Any]:
    observations_path = Path(args.observations_jsonl)
    if observations_path.exists():
        observations_path.unlink()
    os.environ["KIVO_PHASE12_7_ENABLE"] = "1"
    os.environ["KIVO_PHASE12_7_OBS_PATH"] = str(observations_path)
    if args.active:
        os.environ["KIVO_PHASE12_7_ACTIVE"] = "1"
    else:
        os.environ.pop("KIVO_PHASE12_7_ACTIVE", None)

    try:
        generation = generation_fn(args)
    except Exception as exc:
        generation = {
            "status": "failed",
            "output_text": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not args.continue_on_error:
            raise
    records = load_observations(observations_path)
    summary = summarize_records(records)
    candidate = bool(
        generation["status"] == "succeeded"
        and args.active
        and summary["observations_written"] > 0
        and summary["mutation_attempted"]
        and not summary["mutation_applied"]
        and summary["active_experiment_blocked"]
    )
    return {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "generation_status": generation["status"],
        "output_text": generation["output_text"],
        "generation_error": generation["error"],
        "observations_jsonl": str(observations_path),
        "active_enabled": args.active,
        **summary,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "runtime_behavior_changed": False,
        "phase12_8_active_selected_attention_candidate": candidate,
        "caveat": (
            "Candidate means design review only; no active selected attention "
            "or runtime mutation was performed."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Kivo-VD Phase 12.7 Runtime Generation Probe",
        "",
        f"- Generation status: `{report['generation_status']}`",
        f"- Output text: `{report['output_text']}`",
        f"- Generation error: `{report['generation_error']}`",
        f"- Observations written: `{report['observations_written']}`",
        f"- Active enabled: `{str(report['active_enabled']).lower()}`",
        (
            "- Mutation attempted: "
            f"`{str(report['mutation_attempted']).lower()}`"
        ),
        f"- Mutation applied: `{str(report['mutation_applied']).lower()}`",
        (
            "- Active experiment blocked: "
            f"`{str(report['active_experiment_blocked']).lower()}`"
        ),
        f"- Blocker reason: `{report['blocker_reason']}`",
        "- Active routing: `false`",
        "- Measured runtime reduction: `false`",
        "- Runtime behavior changed: `false`",
        (
            "- Phase 12.8 design-review candidate: "
            f"`{str(report['phase12_8_active_selected_attention_candidate']).lower()}`"
        ),
        "",
        "The active mode computes a side-channel decision only. It does not",
        "modify runtime-consumed metadata, KV tensors, or attention.",
    ]) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(args)
        exit_code = 0
    except Exception as exc:
        report = {
            "generation_status": "failed",
            "output_text": None,
            "generation_error": f"{type(exc).__name__}: {exc}",
            "observations_written": 0,
            "active_enabled": args.active,
            "mutation_attempted": False,
            "mutation_applied": False,
            "active_experiment_blocked": False,
            "blocker_reason": None,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "runtime_behavior_changed": False,
            "phase12_8_active_selected_attention_candidate": False,
        }
        exit_code = 0 if args.continue_on_error else 1
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "generation_status": report["generation_status"],
        "observations_written": report["observations_written"],
        "active_enabled": report["active_enabled"],
        "mutation_applied": report["mutation_applied"],
        "phase12_8_active_selected_attention_candidate": report[
            "phase12_8_active_selected_attention_candidate"
        ],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
