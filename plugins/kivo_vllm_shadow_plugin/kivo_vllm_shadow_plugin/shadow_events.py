# SPDX-License-Identifier: Apache-2.0

"""Self-contained preview event emission for the public generate boundary."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_LAYERS = (0, 5, 8, 11)
DEFAULT_BLOCK_SIZE = 16
DEFAULT_RATIO_POLICY = "balanced=0:0.60,5:0.45,8:0.45,11:0.60"


def parse_layers(value: str | None) -> tuple[int, ...]:
    if not value:
        return DEFAULT_LAYERS
    layers = tuple(int(item.strip()) for item in value.split(","))
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("Kivo shadow layers must be non-negative integers")
    return layers


def parse_ratio_policy(value: str | None) -> tuple[str, dict[int, float]]:
    raw = value or DEFAULT_RATIO_POLICY
    name, separator, mapping_text = raw.partition("=")
    if not separator or not name.strip():
        raise ValueError("ratio policy must use name=layer:ratio syntax")
    ratios: dict[int, float] = {}
    for item in mapping_text.split(","):
        layer_text, item_separator, ratio_text = item.partition(":")
        if not item_separator:
            raise ValueError("ratio policy entries must use layer:ratio syntax")
        layer = int(layer_text.strip())
        ratio = float(ratio_text.strip())
        if layer < 0 or not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio policy layers and ratios are out of range")
        ratios[layer] = ratio
    if not ratios:
        raise ValueError("ratio policy must contain at least one layer")
    return name.strip(), ratios


def _prompt_token_count(output: Any, prompt: Any) -> tuple[int, str]:
    token_ids = getattr(output, "prompt_token_ids", None)
    if token_ids is not None:
        try:
            return max(0, len(token_ids)), "output.prompt_token_ids"
        except TypeError:
            pass

    if isinstance(prompt, str):
        return max(1, math.ceil(len(prompt) / 4)), "prompt_length_estimate"
    if isinstance(prompt, Sequence) and not isinstance(
        prompt, (str, bytes, bytearray)
    ):
        return max(0, len(prompt)), "prompt_sequence_length"
    return 1, "minimum_fallback"


def _as_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _request_id(output: Any, index: int) -> str:
    value = getattr(output, "request_id", None)
    return str(value) if value is not None else f"generate-preview-{index}"


def build_preview_events(
    *,
    prompts: Any,
    result: Any,
    layers: Iterable[int] = DEFAULT_LAYERS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    ratio_policy: str = DEFAULT_RATIO_POLICY,
) -> list[dict[str, Any]]:
    """Build deterministic preview events without reading vLLM internals."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    policy_name, layer_ratios = parse_ratio_policy(ratio_policy)
    prompt_items = _as_items(prompts)
    output_items = _as_items(result)
    request_count = max(len(prompt_items), len(output_items), 1)
    events: list[dict[str, Any]] = []

    for request_index in range(request_count):
        output = (
            output_items[request_index]
            if request_index < len(output_items)
            else None
        )
        prompt = (
            prompt_items[request_index]
            if request_index < len(prompt_items)
            else None
        )
        context_tokens, count_source = _prompt_token_count(output, prompt)
        total_blocks = (
            math.ceil(context_tokens / block_size) if context_tokens else 0
        )
        request_id = _request_id(output, request_index)

        for layer in layers:
            ratio = layer_ratios.get(layer, 0.5)
            budget = (
                min(total_blocks, max(1, math.ceil(total_blocks * ratio)))
                if total_blocks
                else 0
            )
            by_score = list(range(total_blocks - 1, total_blocks - budget - 1, -1))
            for_gather = sorted(by_score)
            selected_ratio = budget / total_blocks if total_blocks else 0.0
            events.append({
                "event_type": "kivo_vd_shadow_selection",
                "version": "12.0",
                "request_id": request_id,
                "sequence_id": f"generate-output-{request_index}",
                "layer_idx": layer,
                "step_idx": None,
                "context_token_count": context_tokens,
                "context_token_count_source": count_source,
                "block_size": block_size,
                "total_context_blocks": total_blocks,
                "ratio_policy": {
                    "name": policy_name,
                    "layer_ratio": ratio,
                    "rounding": "ceil",
                    "preview_only": True,
                },
                "candidate_budget_blocks": budget,
                "selected_block_ids_by_score": by_score,
                "selected_block_ids_for_gather": for_gather,
                "selected_block_count": budget,
                "selected_ratio": selected_ratio,
                "estimated_active_block_reduction_ratio": 1.0 - selected_ratio,
                "selector_policy": "plugin_generate_boundary_preview",
                "ordering_valid": True,
                "causal_valid": True,
                "preview_only": True,
                "shadow_only": True,
                "active_routing": False,
                "measured_runtime_reduction": False,
                "caveats": [
                    "generate-boundary preview only; no block-table access",
                    "selected block IDs are deterministic synthetic previews",
                    "full KV allocation and attention behavior are unchanged",
                ],
            })
    return events


def write_events_jsonl(
    path: str | Path,
    events: Iterable[dict[str, Any]],
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(events)
    with output_path.open("a", encoding="utf-8") as output:
        for event in rows:
            output.write(json.dumps(event, sort_keys=True) + "\n")
    return len(rows)


def emit_shadow_events_from_generate_call(
    *,
    prompts: Any,
    result: Any,
    events_path: str | Path,
    layers: Iterable[int] = DEFAULT_LAYERS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    ratio_policy: str = DEFAULT_RATIO_POLICY,
) -> int:
    events = build_preview_events(
        prompts=prompts,
        result=result,
        layers=layers,
        block_size=block_size,
        ratio_policy=ratio_policy,
    )
    return write_events_jsonl(events_path, events)
