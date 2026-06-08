#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Exercise the passive Phase 12 observer with synthetic observations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from phase12_shadow_observer import (
    DEFAULT_RATIO_POLICY,
    Phase12ShadowObservation,
    Phase12ShadowObserver,
    Phase12ShadowObserverConfig,
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
        description="Run the passive Phase 12 shadow observer smoke test."
    )
    parser.add_argument("--num-events", type=int, default=8)
    parser.add_argument("--layers", default="0,5,8,11")
    parser.add_argument("--context-blocks", default="32,64")
    parser.add_argument("--ratio-policy", default=DEFAULT_RATIO_POLICY)
    parser.add_argument(
        "--selector-policy",
        default="query_key_block_score",
    )
    parser.add_argument(
        "--output-jsonl",
        default=(
            "outputs/kivo_vd/phase12_shadow_observer_smoke_events.jsonl"
        ),
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase12_shadow_observer_smoke.md",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run_smoke(
    *,
    num_events: int,
    layers: list[int],
    context_blocks: list[int],
    ratio_policy: str,
    selector_policy: str,
    output_jsonl: str,
    seed: int,
) -> tuple[Phase12ShadowObserver, list[dict[str, Any]], dict[str, Any]]:
    if num_events < 0:
        raise ValueError("num_events must be non-negative")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    observer = Phase12ShadowObserver(Phase12ShadowObserverConfig(
        enabled=True,
        output_jsonl=output_jsonl,
        ratio_policy=ratio_policy,
        selector_policy=selector_policy,
        preview_only=False,
    ))
    rng = random.Random(seed)
    for event_idx in range(num_events):
        total_blocks = context_blocks[event_idx % len(context_blocks)]
        scores = [rng.uniform(-1.0, 1.0) for _ in range(total_blocks)]
        observer.observe(Phase12ShadowObservation(
            request_id=f"observer-smoke-{event_idx:04d}",
            sequence_id=f"sequence-{event_idx:04d}",
            layer_idx=layers[event_idx % len(layers)],
            step_idx=event_idx,
            context_token_count=total_blocks * observer.config.block_size,
            total_context_blocks=total_blocks,
            scores=scores,
        ))
    events = load_events(output_path)
    return observer, events, validate_events(events)


def render_markdown(
    observer: Phase12ShadowObserver,
    events: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    counters = observer.get_counters()
    ratios = [event["selected_ratio"] for event in events]
    average_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    ordering_valid = all(event["ordering_valid"] for event in events)
    lines = [
        "# Kivo-VD Phase 12 Shadow Observer Smoke",
        "",
        "## Summary",
        "",
        f"- Events seen: `{counters['events_seen']}`",
        f"- Events written: `{counters['events_written']}`",
        f"- Invalid events: `{counters['invalid_events']}`",
        f"- Warnings: `{counters['warnings']}`",
        f"- Average selected ratio: `{average_ratio:.6f}`",
        f"- Ordering valid: `{str(ordering_valid).lower()}`",
        (
            "- Validator passed: "
            f"`{str(validation['validation_passed']).lower()}`"
        ),
        "- Active routing: `false`",
        "- Measured runtime reduction: `false`",
        "",
        "## Caveats",
        "",
        "- Observations and scores are synthetic.",
        "- No vLLM runtime hook is installed.",
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
        observer, events, validation = run_smoke(
            num_events=args.num_events,
            layers=_parse_int_list(args.layers, "layers"),
            context_blocks=_parse_int_list(
                args.context_blocks,
                "context-blocks",
            ),
            ratio_policy=args.ratio_policy,
            selector_policy=args.selector_policy,
            output_jsonl=args.output_jsonl,
            seed=args.seed,
        )
        _write(
            args.output_md,
            render_markdown(observer, events, validation),
        )
        summary = {
            **observer.get_counters(),
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
