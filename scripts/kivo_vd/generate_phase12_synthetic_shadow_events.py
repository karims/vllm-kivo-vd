#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic synthetic Kivo-VD Phase 12 shadow events."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from phase12_shadow_events import (
    build_phase12_shadow_event,
    derive_layer_budget,
    parse_ratio_policy,
)

DEFAULT_RATIO_POLICY = "balanced=0:0.60,5:0.45,8:0.45,11:0.60"


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
        description="Generate synthetic Phase 12 shadow-selection events."
    )
    parser.add_argument("--num-events", type=int, default=8)
    parser.add_argument("--layers", default="0,5,8,11")
    parser.add_argument("--context-blocks", default="32,64")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--ratio-policy", default=DEFAULT_RATIO_POLICY)
    parser.add_argument(
        "--selector-policy",
        default="query_key_block_score",
    )
    parser.add_argument("--request-id-prefix", default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-jsonl",
        default="outputs/kivo_vd/phase12_synthetic_shadow_events.jsonl",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase12_synthetic_shadow_events.md",
    )
    return parser.parse_args(argv)


def generate_events(
    *,
    num_events: int,
    layers: list[int],
    context_blocks: list[int],
    block_size: int,
    ratio_policy_spec: str,
    selector_policy: str,
    request_id_prefix: str,
    seed: int,
) -> list[dict[str, Any]]:
    if num_events < 0:
        raise ValueError("num_events must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not request_id_prefix:
        raise ValueError("request_id_prefix is required")

    policy = parse_ratio_policy(ratio_policy_spec)
    missing_layers = sorted(set(layers) - set(policy.layer_ratios))
    if missing_layers:
        missing = ", ".join(str(layer) for layer in missing_layers)
        raise ValueError(f"ratio policy is missing configured layers: {missing}")

    rng = random.Random(seed)
    events: list[dict[str, Any]] = []
    for event_idx in range(num_events):
        layer_idx = layers[event_idx % len(layers)]
        total_blocks = context_blocks[event_idx % len(context_blocks)]
        budget = derive_layer_budget(
            total_context_blocks=total_blocks,
            layer_idx=layer_idx,
            ratio_policy=policy,
        )
        scores = [rng.uniform(-1.0, 1.0) for _ in range(total_blocks)]
        ranked_ids = sorted(
            range(total_blocks),
            key=lambda block_id: (-scores[block_id], block_id),
        )
        selected_ids = ranked_ids[:budget]
        event = build_phase12_shadow_event(
            request_id=f"{request_id_prefix}-{event_idx:04d}",
            sequence_id=f"sequence-{event_idx:04d}",
            layer_idx=layer_idx,
            step_idx=event_idx,
            context_token_count=total_blocks * block_size,
            block_size=block_size,
            total_context_blocks=total_blocks,
            ratio_policy=policy,
            candidate_budget_blocks=budget,
            selected_block_ids_by_score=selected_ids,
            selector_policy=selector_policy,
            scores=scores,
        )
        events.append(event.to_dict())
    return events


def render_markdown(
    events: list[dict[str, Any]],
    *,
    seed: int,
) -> str:
    lines = [
        "# Kivo-VD Phase 12 Synthetic Shadow Events",
        "",
        "## Summary",
        "",
        f"- Events: `{len(events)}`",
        f"- Seed: `{seed}`",
        "- Shadow only: `true`",
        "- Active routing: `false`",
        "- Measured runtime reduction: `false`",
        "",
        "## Events",
        "",
        "| request | layer | blocks | budget | selected ratio | ordering |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for event in events:
        lines.append(
            "| `{request_id}` | `{layer_idx}` | `{total_context_blocks}` | "
            "`{candidate_budget_blocks}` | `{selected_ratio:.6f}` | "
            "`{ordering_valid}` |".format(**event)
        )
    lines.extend([
        "",
        "## Caveats",
        "",
        "- These events are deterministic synthetic fixtures.",
        "- They do not read a vLLM KV cache or run inference.",
        "- Full KV allocation and normal attention remain unchanged.",
        "- No measured memory, latency, or quality claim is implied.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        layers = _parse_int_list(args.layers, "layers")
        context_blocks = _parse_int_list(
            args.context_blocks,
            "context-blocks",
        )
        events = generate_events(
            num_events=args.num_events,
            layers=layers,
            context_blocks=context_blocks,
            block_size=args.block_size,
            ratio_policy_spec=args.ratio_policy,
            selector_policy=args.selector_policy,
            request_id_prefix=args.request_id_prefix,
            seed=args.seed,
        )
        _write(
            args.output_jsonl,
            "".join(
                json.dumps(event, sort_keys=True) + "\n" for event in events
            ),
        )
        _write(args.output_md, render_markdown(events, seed=args.seed))
        print(json.dumps({
            "num_events": len(events),
            "output_jsonl": args.output_jsonl,
            "output_md": args.output_md,
            "shadow_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
        }, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
