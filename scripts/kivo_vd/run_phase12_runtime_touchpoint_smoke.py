#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test Phase 12 runtime-facing touchpoint helpers."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from phase12_shadow_observer import DEFAULT_RATIO_POLICY
from phase12_vllm_runtime_touchpoint import (
    observe_phase12_decode_shadow_metadata,
)
from validate_phase12_shadow_event import load_events, validate_events


def _parse_int_list(spec: str, name: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integers")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 12 runtime touchpoint smoke test."
    )
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--num-events", type=int, default=4)
    parser.add_argument("--request-id-prefix", default="runtime-smoke")
    parser.add_argument("--layers", default="0,5,8,11")
    parser.add_argument("--context-blocks", default="64")
    parser.add_argument(
        "--output-jsonl",
        default=(
            "outputs/kivo_vd/"
            "phase12_runtime_touchpoint_smoke_events.jsonl"
        ),
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase12_runtime_touchpoint_smoke.md",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _env(enabled: bool, output_jsonl: str) -> dict[str, str]:
    return {
        "KIVO_PHASE12_SHADOW_ENABLED": "1" if enabled else "0",
        "KIVO_PHASE12_SHADOW_OUTPUT": output_jsonl,
        "KIVO_PHASE12_RATIO_POLICY": DEFAULT_RATIO_POLICY,
        "KIVO_PHASE12_PREVIEW_ONLY": "0",
    }


def run_smoke(
    *,
    enabled: bool,
    num_events: int,
    request_id_prefix: str,
    layers: list[int],
    context_blocks: list[int],
    output_jsonl: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if num_events < 0:
        raise ValueError("num_events must be non-negative")
    if not request_id_prefix:
        raise ValueError("request-id-prefix is required")
    output_path = Path(output_jsonl)
    if output_path.exists():
        output_path.unlink()

    rng = random.Random(seed)
    env = _env(enabled, output_jsonl)
    results: list[dict[str, Any]] = []
    for event_idx in range(num_events):
        total_blocks = context_blocks[event_idx % len(context_blocks)]
        scores = {
            block_id: rng.uniform(-1.0, 1.0)
            for block_id in range(total_blocks)
        }
        results.append(observe_phase12_decode_shadow_metadata(
            request_id=f"{request_id_prefix}-{event_idx:04d}",
            sequence_id=f"sequence-{event_idx:04d}",
            layer_idx=layers[event_idx % len(layers)],
            step_idx=event_idx,
            context_token_count=total_blocks * 16,
            total_context_blocks=total_blocks,
            block_ids=list(range(total_blocks)),
            scores=scores,
            metadata={"source": "runtime_touchpoint_smoke"},
            env=env,
        ))

    validation = (
        validate_events(load_events(output_path))
        if output_path.exists()
        else {
            "validation_passed": not enabled,
            "total_events": 0,
            "valid_events": 0,
            "invalid_events": 0,
            "warnings": [],
            "errors": [],
        }
    )
    return results, validation


def render_markdown(
    *,
    enabled: bool,
    results: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    written = sum(result["event_written"] for result in results)
    errors = sum(int(result["errors"]) for result in results)
    lines = [
        "# Kivo-VD Phase 12 Runtime Touchpoint Smoke",
        "",
        "## Summary",
        "",
        f"- Touchpoint enabled: `{str(enabled).lower()}`",
        f"- Calls: `{len(results)}`",
        f"- Events written: `{written}`",
        f"- Errors: `{errors}`",
        (
            "- Validation passed: "
            f"`{str(validation['validation_passed']).lower()}`"
        ),
        f"- Valid events: `{validation['valid_events']}`",
        "- Shadow only: `true`",
        "- Active routing: `false`",
        "- Measured runtime reduction: `false`",
        "",
        "## Caveats",
        "",
        "- Inputs and scores are synthetic vLLM-like metadata.",
        "- No core vLLM runtime file is modified by this smoke.",
        "- Full KV allocation and normal attention remain unchanged.",
        "- No memory, latency, or generation-quality claim is implied.",
    ]
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        results, validation = run_smoke(
            enabled=args.enabled,
            num_events=args.num_events,
            request_id_prefix=args.request_id_prefix,
            layers=_parse_int_list(args.layers, "layers"),
            context_blocks=_parse_int_list(
                args.context_blocks,
                "context-blocks",
            ),
            output_jsonl=args.output_jsonl,
            seed=args.seed,
        )
        _write(
            args.output_md,
            render_markdown(
                enabled=args.enabled,
                results=results,
                validation=validation,
            ),
        )
        summary = {
            "enabled": args.enabled,
            "calls": len(results),
            "events_written": sum(
                result["event_written"] for result in results
            ),
            "validation_passed": validation["validation_passed"],
            "output_jsonl": args.output_jsonl,
            "output_md": args.output_md,
            "shadow_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
        }
        print(json.dumps(summary, separators=(",", ":")))
        return 0 if validation["validation_passed"] else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
