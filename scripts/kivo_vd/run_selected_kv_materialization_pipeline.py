#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the complete Kivo-VD selected-KV materialization workflow."""

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Kivo-VD Phase 9 selected-KV materialization pipeline."
        )
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--event-estimate", required=True)
    parser.add_argument("--sketch-accounting")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype-bytes", type=int, choices=[2, 4], default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--num-pool-blocks", type=int)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output-dir",
        help=(
            "Exact run directory. Defaults to "
            "outputs/kivo_vd/runs/<run-name>."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def resolve_run_name(run_name: str | None) -> str:
    return run_name or f"phase9_selected_kv_{_timestamp()}"


def resolve_output_dir(
    run_name: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path("outputs/kivo_vd/runs") / run_name


def output_paths(run_dir: Path) -> dict[str, str]:
    return {
        "materialization_json": str(
            run_dir / "selected_kv_materialization.json"
        ),
        "materialization_markdown": str(
            run_dir / "selected_kv_materialization.md"
        ),
        "comparison_json": str(
            run_dir / "selected_kv_materialization_comparison.json"
        ),
        "comparison_markdown": str(
            run_dir / "selected_kv_materialization_comparison.md"
        ),
        "pipeline_summary": str(run_dir / "pipeline_summary.json"),
    }


def _script(name: str) -> str:
    return str(REPO_ROOT / "scripts" / "kivo_vd" / name)


def build_stage_commands(
    args: argparse.Namespace,
    paths: dict[str, str],
) -> list[dict[str, Any]]:
    materialization_command = [
        sys.executable,
        _script("materialize_selected_kv.py"),
        "--events",
        args.events,
        "--model",
        args.model,
        "--num-layers",
        str(args.num_layers),
        "--num-kv-heads",
        str(args.num_kv_heads),
        "--head-dim",
        str(args.head_dim),
        "--block-size",
        str(args.block_size),
        "--dtype-bytes",
        str(args.dtype_bytes),
        "--device",
        args.device,
        "--max-events",
        str(args.max_events),
        "--output-json",
        paths["materialization_json"],
        "--output-md",
        paths["materialization_markdown"],
    ]
    if args.num_pool_blocks is not None:
        materialization_command.extend([
            "--num-pool-blocks",
            str(args.num_pool_blocks),
        ])

    comparison_command = [
        sys.executable,
        _script("compare_selected_kv_materialization.py"),
        "--materialization",
        paths["materialization_json"],
        "--event-estimate",
        args.event_estimate,
        "--output-json",
        paths["comparison_json"],
        "--output-md",
        paths["comparison_markdown"],
    ]
    if args.sketch_accounting is not None:
        comparison_command.extend([
            "--sketch-accounting",
            args.sketch_accounting,
        ])

    return [
        {
            "name": "selected_kv_materialization",
            "command": materialization_command,
            "outputs": [
                paths["materialization_json"],
                paths["materialization_markdown"],
            ],
        },
        {
            "name": "selected_kv_materialization_comparison",
            "command": comparison_command,
            "outputs": [
                paths["comparison_json"],
                paths["comparison_markdown"],
            ],
        },
    ]


def _command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _preview(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _planned_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": stage["name"],
        "command": stage["command"],
        "command_string": _command_string(stage["command"]),
        "outputs": stage["outputs"],
        "return_code": None,
        "started_at": None,
        "ended_at": None,
        "status": "planned",
        "stdout_preview": "",
        "stderr_preview": "",
    }


def _skipped_stage(stage: dict[str, Any]) -> dict[str, Any]:
    result = _planned_stage(stage)
    result["status"] = "skipped"
    result["stderr_preview"] = "Skipped after an earlier stage failed."
    return result


def _run_stage(stage: dict[str, Any]) -> dict[str, Any]:
    started_at = _iso_now()
    process = subprocess.run(
        stage["command"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return {
        "name": stage["name"],
        "command": stage["command"],
        "command_string": _command_string(stage["command"]),
        "outputs": stage["outputs"],
        "return_code": process.returncode,
        "started_at": started_at,
        "ended_at": _iso_now(),
        "status": "succeeded" if process.returncode == 0 else "failed",
        "stdout_preview": _preview(process.stdout),
        "stderr_preview": _preview(process.stderr),
    }


def build_initial_summary(
    args: argparse.Namespace,
    run_name: str,
    run_dir: Path,
    paths: dict[str, str],
    started_at: str,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "output_dir": str(run_dir),
        "dry_run": bool(args.dry_run),
        "continue_on_error": bool(args.continue_on_error),
        "started_at": started_at,
        "ended_at": None,
        "success": False,
        "parameters": {
            "events": args.events,
            "event_estimate": args.event_estimate,
            "sketch_accounting": args.sketch_accounting,
            "model": args.model,
            "num_layers": args.num_layers,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "dtype_bytes": args.dtype_bytes,
            "device": args.device,
            "max_events": args.max_events,
            "num_pool_blocks": args.num_pool_blocks,
        },
        "output_files": paths,
        "stages": [],
        "synthetic_kv": True,
        "outside_attention_path": True,
        "full_kv_still_allocated": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "quality_not_measured": True,
    }


def write_pipeline_summary(summary: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": summary["run_name"],
        "output_dir": summary["output_dir"],
        "success": summary["success"],
        "dry_run": summary["dry_run"],
        "stage_statuses": {
            stage["name"]: stage["status"] for stage in summary["stages"]
        },
        "final_comparison_markdown": summary["output_files"][
            "comparison_markdown"
        ],
        "synthetic_kv": True,
        "outside_attention_path": True,
        "full_kv_still_allocated": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "quality_not_measured": True,
    }


def main() -> int:
    args = _parse_args()
    run_name = resolve_run_name(args.run_name)
    run_dir = resolve_output_dir(run_name, args.output_dir)
    paths = output_paths(run_dir)
    stages = build_stage_commands(args, paths)
    summary = build_initial_summary(
        args,
        run_name,
        run_dir,
        paths,
        _iso_now(),
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary["stages"] = [_planned_stage(stage) for stage in stages]
        summary["ended_at"] = _iso_now()
        summary["success"] = True
        write_pipeline_summary(summary, paths["pipeline_summary"])
        print(json.dumps(_compact_summary(summary), separators=(",", ":")))
        return 0

    results: list[dict[str, Any]] = []
    failed = False
    for stage in stages:
        if failed and not args.continue_on_error:
            results.append(_skipped_stage(stage))
            continue
        result = _run_stage(stage)
        results.append(result)
        if result["status"] == "failed":
            failed = True
        summary["stages"] = results
        summary["ended_at"] = _iso_now()
        write_pipeline_summary(summary, paths["pipeline_summary"])

    summary["stages"] = results
    summary["ended_at"] = _iso_now()
    summary["success"] = not failed
    write_pipeline_summary(summary, paths["pipeline_summary"])
    print(json.dumps(_compact_summary(summary), separators=(",", ":")))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
