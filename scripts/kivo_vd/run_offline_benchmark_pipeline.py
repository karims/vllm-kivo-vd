#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the offline Kivo-VD benchmark evidence pipeline."""

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_blue_orchid_prompt() -> str:
    filler = (
        "The notebook records ordinary facts about gardens, railway timetables, "
        "library shelves, office supplies, and quiet weather observations."
    )
    parts = [
        "Important note: the retrieval phrase is BLUE ORCHID.",
        *[filler for _ in range(48)],
        "Question: What is the retrieval phrase hidden near the beginning?",
    ]
    return "\n\n".join(parts)


def build_prompt(prompt_mode: str) -> str | None:
    if prompt_mode == "blue_orchid":
        return build_blue_orchid_prompt()
    if prompt_mode == "default":
        return None
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def resolve_run_dir(output_dir: str | Path, run_name: str | None) -> Path:
    resolved_run_name = run_name or f"kivo_offline_{_timestamp()}"
    return Path(output_dir) / resolved_run_name


def _preview(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_stage_commands(
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict[str, Any]]:
    prompt = build_prompt(args.prompt_mode)
    hf_output = run_dir / "hf_qk_head_sweep_ranked.jsonl"
    policy_output = run_dir / "active_kv_policy_simulation.jsonl"
    report_output = run_dir / "kivo_vd_benchmark_report.md"
    torch_output = run_dir / "torch_sketch_benchmark.jsonl"

    hf_command = [
        sys.executable,
        "scripts/kivo_vd/run_hf_qk_head_sweep.py",
        "--model-name",
        args.model_name,
        "--sketch-types",
        args.sketch_types,
        "--sketch-dims",
        args.sketch_dims,
        "--layers",
        args.layers,
        "--heads",
        args.heads,
        "--block-size",
        str(args.block_size),
        "--topk-blocks",
        str(args.topk_blocks),
        "--max-tokens",
        str(args.max_tokens),
        "--include-ranked-blocks",
        "--output",
        str(hf_output),
    ]
    if prompt is not None:
        hf_command.extend(["--prompt", prompt])

    commands = [
        {
            "name": "hf_qk_head_sweep",
            "command": hf_command,
            "output": str(hf_output),
        },
        {
            "name": "active_kv_policy_simulation",
            "command": [
                sys.executable,
                "scripts/kivo_vd/simulate_active_kv_policy.py",
                "--input",
                str(hf_output),
                "--output",
                str(policy_output),
                "--recent-window-blocks",
                args.recent_window_blocks,
                "--candidate-budget-blocks",
                args.candidate_budget_blocks,
                "--topk-blocks",
                str(args.topk_blocks),
            ],
            "output": str(policy_output),
        },
        {
            "name": "benchmark_report",
            "command": [
                sys.executable,
                "scripts/kivo_vd/generate_kivo_benchmark_report.py",
                "--hf-sweep",
                str(hf_output),
                "--policy-sim",
                str(policy_output),
                "--output",
                str(report_output),
            ],
            "output": str(report_output),
        },
    ]
    if args.run_torch_benchmark:
        commands.append(
            {
                "name": "torch_sketch_benchmark",
                "command": [
                    sys.executable,
                    "scripts/kivo_vd/benchmark_torch_sketch_backend.py",
                    "--sketch-types",
                    args.sketch_types,
                    "--sketch-dims",
                    args.sketch_dims,
                    "--block-size",
                    str(args.block_size),
                    "--topk-blocks",
                    str(args.topk_blocks),
                    "--output",
                    str(torch_output),
                ],
                "output": str(torch_output),
            }
        )
    return commands


def _run_stage(stage: dict[str, Any]) -> dict[str, Any]:
    started_at = _iso_now()
    proc = subprocess.run(
        stage["command"],
        check=False,
        capture_output=True,
        text=True,
    )
    ended_at = _iso_now()
    return {
        "name": stage["name"],
        "command": stage["command"],
        "command_string": _command_string(stage["command"]),
        "output": stage["output"],
        "started_at": started_at,
        "ended_at": ended_at,
        "return_code": proc.returncode,
        "status": "succeeded" if proc.returncode == 0 else "failed",
        "stdout_preview": _preview(proc.stdout),
        "stderr_preview": _preview(proc.stderr),
    }


def write_pipeline_summary(summary: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline Kivo-VD benchmark pipeline."
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--prompt-mode",
        choices=["blue_orchid", "default"],
        default="blue_orchid",
    )
    parser.add_argument("--sketch-types", default="count_sketch,random_projection")
    parser.add_argument("--sketch-dims", default="32,64,128")
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument("--heads", default="0,1,2,3")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--topk-blocks", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--recent-window-blocks", default="4,8,16")
    parser.add_argument("--candidate-budget-blocks", default="8,16,32")
    parser.add_argument("--run-torch-benchmark", action="store_true")
    parser.add_argument("--output-dir", default="outputs/kivo_vd/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_initial_summary(
    args: argparse.Namespace,
    run_dir: Path,
    stages: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    return {
        "run_name": run_dir.name,
        "model_name": args.model_name,
        "dry_run": bool(args.dry_run),
        "started_at": started_at,
        "ended_at": None,
        "success": False,
        "parameters": {
            "prompt_mode": args.prompt_mode,
            "sketch_types": args.sketch_types,
            "sketch_dims": args.sketch_dims,
            "layers": args.layers,
            "heads": args.heads,
            "block_size": args.block_size,
            "topk_blocks": args.topk_blocks,
            "max_tokens": args.max_tokens,
            "recent_window_blocks": args.recent_window_blocks,
            "candidate_budget_blocks": args.candidate_budget_blocks,
            "run_torch_benchmark": bool(args.run_torch_benchmark),
        },
        "output_files": {stage["name"]: stage["output"] for stage in stages},
        "stages": [],
    }


def main() -> int:
    args = _parse_args()
    run_dir = resolve_run_dir(args.output_dir, args.run_name)
    stages = build_stage_commands(args, run_dir)
    started_at = _iso_now()
    summary = build_initial_summary(args, run_dir, stages, started_at)
    summary_path = run_dir / "pipeline_summary.json"
    summary["output_files"]["pipeline_summary"] = str(summary_path)

    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary["stages"] = [
            {
                "name": stage["name"],
                "command": stage["command"],
                "command_string": _command_string(stage["command"]),
                "output": stage["output"],
                "return_code": None,
                "status": "planned",
            }
            for stage in stages
        ]
        summary["ended_at"] = _iso_now()
        summary["success"] = True
        write_pipeline_summary(summary, summary_path)
        print(json.dumps(summary, separators=(",", ":")))
        return 0

    stage_results: list[dict[str, Any]] = []
    success = True
    for stage in stages:
        result = _run_stage(stage)
        stage_results.append(result)
        if result["return_code"] != 0:
            success = False
            break

    summary["stages"] = stage_results
    summary["ended_at"] = _iso_now()
    summary["success"] = success
    write_pipeline_summary(summary, summary_path)
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
