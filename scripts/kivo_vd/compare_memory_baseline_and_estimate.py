#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare measured vLLM memory with theoretical Kivo event estimates."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Phase 7.0 measured memory with Phase 7.1 theoretical "
            "active-KV estimates."
        )
    )
    parser.add_argument("--baseline-memory", required=True)
    parser.add_argument("--kivo-memory")
    parser.add_argument("--event-estimate", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase7_2_memory_comparison.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase7_2_memory_comparison.md",
    )
    return parser.parse_args(argv)


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} file is missing: {input_path}")
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} contains invalid JSON: {input_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {input_path}")
    return value


def _checkpoint_map(memory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checkpoints = memory.get("memory_checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("memory JSON lacks a memory_checkpoints list")
    result: dict[str, dict[str, Any]] = {}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        name = checkpoint.get("name")
        if isinstance(name, str):
            result[name] = checkpoint
    return result


def _metric(checkpoint: dict[str, Any], name: str) -> int:
    value = checkpoint.get(name)
    if not isinstance(value, int | float):
        raise ValueError(f"memory checkpoint lacks numeric {name}")
    return int(value)


def _required_checkpoint(
    checkpoints: dict[str, dict[str, Any]], name: str
) -> dict[str, Any]:
    checkpoint = checkpoints.get(name)
    if checkpoint is None:
        raise ValueError(f"memory JSON lacks required checkpoint: {name}")
    return checkpoint


def summarize_measured_memory(memory: dict[str, Any]) -> dict[str, Any]:
    checkpoints = _checkpoint_map(memory)
    before_init = _required_checkpoint(checkpoints, "before_llm_init")
    after_init = _required_checkpoint(checkpoints, "after_llm_init")
    before_generate = _required_checkpoint(checkpoints, "before_generate")
    after_generate = _required_checkpoint(checkpoints, "after_generate")
    cleanup = checkpoints.get("after_request_or_cleanup")

    peak_allocated = max(
        _metric(checkpoint, "max_memory_allocated_bytes")
        for checkpoint in checkpoints.values()
    )
    peak_reserved = max(
        _metric(checkpoint, "max_memory_reserved_bytes")
        for checkpoint in checkpoints.values()
    )
    summary: dict[str, Any] = {
        "model": memory.get("config", {}).get("model"),
        "kivo_enabled": bool(memory.get("kivo_enabled", False)),
        "before_llm_init_allocated_bytes": _metric(
            before_init, "memory_allocated_bytes"
        ),
        "after_llm_init_allocated_bytes": _metric(
            after_init, "memory_allocated_bytes"
        ),
        "model_init_allocated_delta_bytes": (
            _metric(after_init, "memory_allocated_bytes")
            - _metric(before_init, "memory_allocated_bytes")
        ),
        "model_init_reserved_delta_bytes": (
            _metric(after_init, "memory_reserved_bytes")
            - _metric(before_init, "memory_reserved_bytes")
        ),
        "before_generate_allocated_bytes": _metric(
            before_generate, "memory_allocated_bytes"
        ),
        "after_generate_allocated_bytes": _metric(
            after_generate, "memory_allocated_bytes"
        ),
        "generation_allocated_delta_bytes": (
            _metric(after_generate, "memory_allocated_bytes")
            - _metric(before_generate, "memory_allocated_bytes")
        ),
        "generation_reserved_delta_bytes": (
            _metric(after_generate, "memory_reserved_bytes")
            - _metric(before_generate, "memory_reserved_bytes")
        ),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "cleanup_allocated_bytes": None,
        "cleanup_reserved_bytes": None,
    }
    if cleanup is not None:
        summary["cleanup_allocated_bytes"] = _metric(
            cleanup, "memory_allocated_bytes"
        )
        summary["cleanup_reserved_bytes"] = _metric(
            cleanup, "memory_reserved_bytes"
        )
    return summary


def summarize_event_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    aggregate = estimate.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("event estimate JSON lacks an aggregate object")
    bytes_per_block = estimate.get("bytes_per_block")
    if not isinstance(bytes_per_block, int | float):
        raise ValueError("event estimate JSON lacks numeric bytes_per_block")

    return {
        "bytes_per_block": int(bytes_per_block),
        "average_selected_blocks": aggregate.get("average_selected_blocks"),
        "average_skipped_blocks": aggregate.get("average_skipped_blocks"),
        "average_active_kv_bytes": aggregate.get("average_active_kv_bytes"),
        "average_skipped_kv_bytes": aggregate.get("average_skipped_kv_bytes"),
        "average_estimated_reduction_ratio": aggregate.get(
            "average_estimated_reduction_ratio"
        ),
        "total_routing_events": aggregate.get("total_routing_events"),
        "estimated_routing_events": aggregate.get("estimated_routing_events"),
        "source_estimated_only": bool(estimate.get("estimated_only", False)),
        "source_measured_runtime_reduction": bool(
            estimate.get("measured_runtime_reduction", False)
        ),
    }


def compare_measured_runs(
    baseline: dict[str, Any],
    kivo: dict[str, Any],
) -> dict[str, Any]:
    peak_allocated_delta = (
        kivo["peak_allocated_bytes"] - baseline["peak_allocated_bytes"]
    )
    peak_reserved_delta = (
        kivo["peak_reserved_bytes"] - baseline["peak_reserved_bytes"]
    )
    return {
        "kivo_minus_baseline_model_init_allocated_bytes": (
            kivo["model_init_allocated_delta_bytes"]
            - baseline["model_init_allocated_delta_bytes"]
        ),
        "kivo_minus_baseline_generation_allocated_bytes": (
            kivo["generation_allocated_delta_bytes"]
            - baseline["generation_allocated_delta_bytes"]
        ),
        "kivo_minus_baseline_peak_allocated_bytes": peak_allocated_delta,
        "kivo_minus_baseline_peak_reserved_bytes": peak_reserved_delta,
        "peak_allocated_drop_observed": peak_allocated_delta < 0,
        "peak_reserved_drop_observed": peak_reserved_delta < 0,
    }


def build_comparison(
    *,
    baseline_path: str | Path,
    event_estimate_path: str | Path,
    kivo_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline_json = _load_json(baseline_path, "baseline memory")
    estimate_json = _load_json(event_estimate_path, "event estimate")
    baseline = summarize_measured_memory(baseline_json)
    theoretical = summarize_event_estimate(estimate_json)

    kivo = None
    measured_comparison = None
    measured_drop_observed = False
    if kivo_path is not None:
        kivo_json = _load_json(kivo_path, "Kivo memory")
        kivo = summarize_measured_memory(kivo_json)
        measured_comparison = compare_measured_runs(baseline, kivo)
        measured_drop_observed = bool(
            measured_comparison["peak_allocated_drop_observed"]
        )

    theoretical_available = (
        theoretical["average_estimated_reduction_ratio"] is not None
    )
    return {
        "input_paths": {
            "baseline_memory": str(baseline_path),
            "kivo_memory": str(kivo_path) if kivo_path is not None else None,
            "event_estimate": str(event_estimate_path),
        },
        "measured_memory_summary": {
            "baseline": baseline,
            "kivo_dry_run": kivo,
        },
        "theoretical_event_estimate_summary": theoretical,
        "baseline_vs_kivo_measured_comparison": measured_comparison,
        "conclusion": {
            "measured_runtime_drop_observed": measured_drop_observed,
            "measured_runtime_reduction_observed": measured_drop_observed,
            "theoretical_active_kv_reduction_available": theoretical_available,
            "dry_run_gap_explanation": (
                "vLLM still allocates the full KV cache and Kivo only records "
                "candidate selections. The event estimate is counterfactual "
                "active-KV accounting, not memory released by this run."
            ),
        },
        "caveats": {
            "estimated_only_for_savings": True,
            "measured_runtime_reduction": False,
            "observed_drop_is_not_attributed_to_kivo": True,
            "latency_improvement_claimed": False,
        },
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(lines: list[str], rows: list[tuple[str, Any]]) -> None:
    lines.extend(["| metric | value |", "| --- | ---: |"])
    lines.extend(
        f"| {name} | `{_format_value(value)}` |" for name, value in rows
    )


def render_markdown(result: dict[str, Any]) -> str:
    measured = result["measured_memory_summary"]
    theoretical = result["theoretical_event_estimate_summary"]
    comparison = result["baseline_vs_kivo_measured_comparison"]
    conclusion = result["conclusion"]

    lines = [
        "# Kivo-VD Memory Baseline Vs Event Estimate",
        "",
        "**Status:** Measured runtime memory is reported separately from "
        "theoretical event-based active-KV savings.",
        "",
        "## Baseline Measured Memory",
        "",
    ]
    _table(lines, _measured_rows(measured["baseline"]))

    if measured["kivo_dry_run"] is not None:
        lines.extend(["", "## Kivo Dry-Run Measured Memory", ""])
        _table(lines, _measured_rows(measured["kivo_dry_run"]))

    lines.extend(["", "## Theoretical Event Estimate", ""])
    _table(lines, [
        ("Bytes per KV block", theoretical["bytes_per_block"]),
        ("Average selected blocks", theoretical["average_selected_blocks"]),
        ("Average skipped blocks", theoretical["average_skipped_blocks"]),
        ("Average active KV bytes", theoretical["average_active_kv_bytes"]),
        ("Average skipped KV bytes", theoretical["average_skipped_kv_bytes"]),
        (
            "Average estimated reduction ratio",
            theoretical["average_estimated_reduction_ratio"],
        ),
        ("Estimated routing events", theoretical["estimated_routing_events"]),
    ])

    if comparison is not None:
        lines.extend(["", "## Baseline Vs Kivo Dry-Run", ""])
        _table(lines, [
            (
                "Kivo - baseline init allocated bytes",
                comparison[
                    "kivo_minus_baseline_model_init_allocated_bytes"
                ],
            ),
            (
                "Kivo - baseline generation allocated bytes",
                comparison[
                    "kivo_minus_baseline_generation_allocated_bytes"
                ],
            ),
            (
                "Kivo - baseline peak allocated bytes",
                comparison["kivo_minus_baseline_peak_allocated_bytes"],
            ),
            (
                "Kivo - baseline peak reserved bytes",
                comparison["kivo_minus_baseline_peak_reserved_bytes"],
            ),
            (
                "Peak allocated drop observed",
                comparison["peak_allocated_drop_observed"],
            ),
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        conclusion["dry_run_gap_explanation"],
        "",
        "An observed lower peak in one Kivo dry-run is recorded as a run-to-run "
        "measurement only. It is not attributed to Kivo because no allocation "
        "or attention behavior changes in dry-run mode.",
        "",
        "## Proven Vs Not Proven",
        "",
        "Proven by this report:",
        "",
        "- Phase 7.0 CUDA measurements and Phase 7.1 theoretical accounting "
        "can be compared without conflating them;",
        "- any measured baseline/Kivo differences are visible explicitly.",
        "",
        "Not proven by this report:",
        "",
        "- measured runtime KV memory reduction;",
        "- active KV routing or candidate-block attention;",
        "- latency improvement or quality preservation.",
        "",
        "## Next Steps",
        "",
        "- Repeat baseline and dry-run measurements under identical conditions.",
        "- Add dry-run event-based memory trends across longer requests.",
        "- Do not attempt active routing until memory accounting and quality "
        "baselines are stable.",
        "",
        "## Caveats",
        "",
        "- `estimated_only_for_savings` is `true`.",
        "- `measured_runtime_reduction` is `false`.",
        "- vLLM still allocates and attends over the normal/full KV cache.",
    ])
    return "\n".join(lines) + "\n"


def _measured_rows(summary: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("Model", summary["model"]),
        ("Kivo enabled", summary["kivo_enabled"]),
        (
            "Model/init allocated delta bytes",
            summary["model_init_allocated_delta_bytes"],
        ),
        (
            "Model/init reserved delta bytes",
            summary["model_init_reserved_delta_bytes"],
        ),
        (
            "Generation allocated delta bytes",
            summary["generation_allocated_delta_bytes"],
        ),
        (
            "Generation reserved delta bytes",
            summary["generation_reserved_delta_bytes"],
        ),
        ("Peak allocated bytes", summary["peak_allocated_bytes"]),
        ("Peak reserved bytes", summary["peak_reserved_bytes"]),
        ("Cleanup allocated bytes", summary["cleanup_allocated_bytes"]),
        ("Cleanup reserved bytes", summary["cleanup_reserved_bytes"]),
    ]


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        args = _parse_args()
        result = build_comparison(
            baseline_path=args.baseline_memory,
            kivo_path=args.kivo_memory,
            event_estimate_path=args.event_estimate,
        )
        _write_text(
            args.output_json,
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
        _write_text(args.output_md, render_markdown(result))
        summary = {
            "output_json": args.output_json,
            "output_md": args.output_md,
            "conclusion": result["conclusion"],
            "caveats": result["caveats"],
        }
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "estimated_only_for_savings": True,
                    "measured_runtime_reduction": False,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
