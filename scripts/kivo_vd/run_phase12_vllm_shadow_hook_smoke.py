#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Exercise the manual Phase 12 vLLM shadow-hook API."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from phase12_shadow_observer import DEFAULT_RATIO_POLICY
from phase12_vllm_shadow_hook import (
    Phase12VllmShadowHook,
    Phase12VllmShadowHookConfig,
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
        description="Run the opt-in Phase 12 vLLM shadow-hook smoke test."
    )
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--num-events", type=int, default=4)
    parser.add_argument("--layers", default="0,5,8,11")
    parser.add_argument("--context-blocks", default="64")
    parser.add_argument(
        "--output-jsonl",
        default=(
            "outputs/kivo_vd/"
            "phase12_vllm_shadow_hook_smoke_events.jsonl"
        ),
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase12_vllm_shadow_hook_smoke.md",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run_smoke(
    *,
    enabled: bool,
    num_events: int,
    layers: list[int],
    context_blocks: list[int],
    output_jsonl: str,
    seed: int,
) -> tuple[Phase12VllmShadowHook, list[dict[str, Any]], dict[str, Any]]:
    if num_events < 0:
        raise ValueError("num_events must be non-negative")
    output_path = Path(output_jsonl)
    if output_path.exists():
        output_path.unlink()

    hook = Phase12VllmShadowHook(Phase12VllmShadowHookConfig(
        enabled=enabled,
        output_jsonl=output_jsonl,
        ratio_policy=DEFAULT_RATIO_POLICY,
        preview_only=False,
    ))
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for event_idx in range(num_events):
        total_blocks = context_blocks[event_idx % len(context_blocks)]
        scores = {
            block_id: rng.uniform(-1.0, 1.0)
            for block_id in range(total_blocks)
        }
        results.append(hook.observe_decode_metadata(
            request_id=f"hook-smoke-{event_idx:04d}",
            sequence_id=f"sequence-{event_idx:04d}",
            layer_idx=layers[event_idx % len(layers)],
            step_idx=event_idx,
            context_token_count=total_blocks * 16,
            total_context_blocks=total_blocks,
            block_ids=list(range(total_blocks)),
            scores=scores,
            metadata={"source": "synthetic_smoke"},
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
    return hook, results, validation


def render_markdown(
    hook: Phase12VllmShadowHook,
    results: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    written = sum(result["event_written"] for result in results)
    lines = [
        "# Kivo-VD Phase 12 vLLM Shadow Hook Smoke",
        "",
        "## Summary",
        "",
        f"- Hook enabled: `{str(hook.config.enabled).lower()}`",
        f"- Calls: `{len(results)}`",
        f"- Events written: `{written}`",
        f"- Hook errors: `{hook.errors}`",
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
        "- No automatic vLLM runtime hook is installed.",
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
        hook, results, validation = run_smoke(
            enabled=args.enabled,
            num_events=args.num_events,
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
            render_markdown(hook, results, validation),
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
