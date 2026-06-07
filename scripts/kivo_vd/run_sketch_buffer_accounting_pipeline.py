#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the complete Kivo-VD Phase 8 sketch-buffer accounting workflow."""

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
            "Run the Kivo-VD Phase 8 sketch-buffer accounting pipeline."
        )
    )
    parser.add_argument("--event-estimate", required=True)
    parser.add_argument("--memory-comparison")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--dtype-bytes", type=int, choices=[2, 4], default=2)
    parser.add_argument(
        "--sketch-types",
        default=(
            "count_sketch,random_projection,bidiagonal_sign_subsample"
        ),
    )
    parser.add_argument("--sketch-dims", default="16,32,64")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
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
    return run_name or f"phase8_sketch_buffers_{_timestamp()}"


def resolve_output_dir(
    run_name: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path("outputs/kivo_vd/runs") / run_name


def output_paths(run_dir: Path) -> dict[str, str]:
    return {
        "sketch_overhead_json": str(
            run_dir / "sketch_buffer_overhead.json"
        ),
        "sketch_overhead_markdown": str(
            run_dir / "sketch_buffer_overhead.md"
        ),
        "overhead_vs_savings_json": str(
            run_dir / "sketch_overhead_vs_savings.json"
        ),
        "overhead_vs_savings_markdown": str(
            run_dir / "sketch_overhead_vs_savings.md"
        ),
        "event_accounting_json": str(
            run_dir / "event_aware_sketch_buffer_accounting.json"
        ),
        "event_accounting_markdown": str(
            run_dir / "event_aware_sketch_buffer_accounting.md"
        ),
        "pipeline_summary": str(run_dir / "pipeline_summary.json"),
    }


def _script(name: str) -> str:
    return str(REPO_ROOT / "scripts" / "kivo_vd" / name)


def _optional_memory_comparison(
    command: list[str],
    path: str | None,
) -> None:
    if path is not None:
        command.extend(["--memory-comparison", path])


def build_stage_commands(
    args: argparse.Namespace,
    paths: dict[str, str],
) -> list[dict[str, Any]]:
    overhead_command = [
        sys.executable,
        _script("measure_sketch_buffer_overhead.py"),
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
        "--num-blocks",
        str(args.num_blocks),
        "--dtype-bytes",
        str(args.dtype_bytes),
        "--sketch-types",
        args.sketch_types,
        "--sketch-dims",
        args.sketch_dims,
        "--device",
        args.device,
        "--output-json",
        paths["sketch_overhead_json"],
        "--output-md",
        paths["sketch_overhead_markdown"],
    ]
    comparison_command = [
        sys.executable,
        _script("compare_sketch_overhead_to_savings.py"),
        "--event-estimate",
        args.event_estimate,
        "--sketch-overhead",
        paths["sketch_overhead_json"],
        "--output-json",
        paths["overhead_vs_savings_json"],
        "--output-md",
        paths["overhead_vs_savings_markdown"],
    ]
    accounting_command = [
        sys.executable,
        _script("model_sketch_buffer_accounting.py"),
        "--event-estimate",
        args.event_estimate,
        "--sketch-overhead",
        paths["sketch_overhead_json"],
        "--output-json",
        paths["event_accounting_json"],
        "--output-md",
        paths["event_accounting_markdown"],
    ]
    _optional_memory_comparison(
        comparison_command, args.memory_comparison
    )
    _optional_memory_comparison(
        accounting_command, args.memory_comparison
    )
    return [
        {
            "name": "sketch_buffer_overhead_measurement",
            "command": overhead_command,
            "outputs": [
                paths["sketch_overhead_json"],
                paths["sketch_overhead_markdown"],
            ],
        },
        {
            "name": "overhead_vs_savings_comparison",
            "command": comparison_command,
            "outputs": [
                paths["overhead_vs_savings_json"],
                paths["overhead_vs_savings_markdown"],
            ],
        },
        {
            "name": "event_aware_sketch_buffer_accounting",
            "command": accounting_command,
            "outputs": [
                paths["event_accounting_json"],
                paths["event_accounting_markdown"],
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
            "event_estimate": args.event_estimate,
            "memory_comparison": args.memory_comparison,
            "model": args.model,
            "num_layers": args.num_layers,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "dtype_bytes": args.dtype_bytes,
            "sketch_types": args.sketch_types,
            "sketch_dims": args.sketch_dims,
            "device": args.device,
        },
        "output_files": paths,
        "stages": [],
        "savings_are_theoretical_only": True,
        "measured_runtime_reduction": False,
        "active_routing": False,
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
        "final_accounting_markdown": summary["output_files"][
            "event_accounting_markdown"
        ],
        "savings_are_theoretical_only": True,
        "measured_runtime_reduction": False,
        "active_routing": False,
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
