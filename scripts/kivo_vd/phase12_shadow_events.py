#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Reusable builders for synthetic Kivo-VD Phase 12 shadow events."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ShadowRatioPolicy:
    """Layer-specific ratios used to derive shadow candidate budgets."""

    name: str
    layer_ratios: dict[int, float]

    def ratio_for_layer(self, layer_idx: int) -> float:
        if layer_idx not in self.layer_ratios:
            available = ", ".join(str(idx) for idx in sorted(self.layer_ratios))
            raise ValueError(
                f"ratio policy {self.name!r} has no ratio for layer "
                f"{layer_idx}; configured layers: {available}"
            )
        return self.layer_ratios[layer_idx]

    def to_dict(self, layer_idx: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_ratio": self.ratio_for_layer(layer_idx),
            "layer_ratios": {
                str(idx): ratio
                for idx, ratio in sorted(self.layer_ratios.items())
            },
        }


@dataclass(frozen=True)
class ShadowSelectionResult:
    """Score-ranked and sequence-ordered views of one shadow selection."""

    selected_block_ids_by_score: tuple[int, ...]
    selected_block_ids_for_gather: tuple[int, ...]
    selected_count: int
    selected_ratio: float
    estimated_active_block_reduction_ratio: float
    ordering_valid: bool


@dataclass(frozen=True)
class Phase12ShadowEvent:
    """Serializable Phase 12 shadow-selection event."""

    request_id: str
    layer_idx: int
    context_token_count: int
    block_size: int
    total_context_blocks: int
    ratio_policy_name: str
    ratio_policy: dict[str, Any]
    candidate_budget_blocks: int
    selected_block_ids_by_score: list[int]
    selected_block_ids_for_gather: list[int]
    selected_block_count: int
    selected_ratio: float
    estimated_active_block_reduction_ratio: float
    selector_policy: str
    selector_scores_summary: dict[str, float] | None
    ordering_valid: bool
    causal_valid: bool = True
    preview_only: bool = False
    shadow_only: bool = True
    active_routing: bool = False
    measured_runtime_reduction: bool = False
    caveats: list[str] = field(default_factory=list)
    event_type: str = "kivo_vd_shadow_selection"
    version: str = "12.0"
    sequence_id: str | None = None
    step_idx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_ratio_policy(spec: str) -> ShadowRatioPolicy:
    """Parse ``name=layer:ratio,...`` into a validated ratio policy."""

    if "=" not in spec:
        raise ValueError(
            "ratio policy must use name=layer:ratio,... syntax"
        )
    name, raw_entries = (part.strip() for part in spec.split("=", 1))
    if not name or not raw_entries:
        raise ValueError("ratio policy name and layer ratios are required")

    layer_ratios: dict[int, float] = {}
    for raw_entry in raw_entries.split(","):
        entry = raw_entry.strip()
        if ":" not in entry:
            raise ValueError(
                f"invalid ratio policy entry {entry!r}; expected layer:ratio"
            )
        raw_layer, raw_ratio = (part.strip() for part in entry.split(":", 1))
        try:
            layer_idx = int(raw_layer)
            ratio = float(raw_ratio)
        except ValueError as exc:
            raise ValueError(
                f"invalid ratio policy entry {entry!r}"
            ) from exc
        if layer_idx < 0:
            raise ValueError("ratio policy layer indices must be non-negative")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio policy values must be between 0 and 1")
        if layer_idx in layer_ratios:
            raise ValueError(
                f"ratio policy repeats layer index {layer_idx}"
            )
        layer_ratios[layer_idx] = ratio

    return ShadowRatioPolicy(name=name, layer_ratios=layer_ratios)


def derive_layer_budget(
    total_context_blocks: int,
    layer_idx: int,
    ratio_policy: ShadowRatioPolicy,
    min_budget: int = 1,
    max_budget: int | None = None,
) -> int:
    """Derive a ceil-rounded, clamped shadow candidate budget."""

    if total_context_blocks < 0:
        raise ValueError("total_context_blocks must be non-negative")
    if min_budget < 0:
        raise ValueError("min_budget must be non-negative")
    if max_budget is not None and max_budget < 0:
        raise ValueError("max_budget must be non-negative")
    if max_budget is not None and min_budget > max_budget:
        raise ValueError("min_budget must not exceed max_budget")
    if total_context_blocks == 0:
        return 0

    budget = math.ceil(
        total_context_blocks * ratio_policy.ratio_for_layer(layer_idx)
    )
    budget = max(min_budget, budget)
    if max_budget is not None:
        budget = min(max_budget, budget)
    return min(total_context_blocks, budget)


def sort_selected_ids_for_gather(
    selected_block_ids_by_score: Iterable[int],
) -> list[int]:
    """Restore original block order for later gather simulation."""

    return sorted(selected_block_ids_by_score)


def validate_ordering(
    selected_block_ids_by_score: Sequence[int],
    selected_block_ids_for_gather: Sequence[int],
) -> bool:
    """Check uniqueness, set equality, and ascending gather order."""

    score_ids = list(selected_block_ids_by_score)
    gather_ids = list(selected_block_ids_for_gather)
    return (
        len(score_ids) == len(set(score_ids))
        and len(gather_ids) == len(set(gather_ids))
        and set(score_ids) == set(gather_ids)
        and gather_ids == sorted(gather_ids)
    )


def compute_selected_ratio(
    selected_count: int,
    total_context_blocks: int,
) -> float:
    if selected_count < 0 or total_context_blocks < 0:
        raise ValueError("block counts must be non-negative")
    if selected_count > total_context_blocks:
        raise ValueError("selected_count must not exceed total blocks")
    if total_context_blocks == 0:
        return 0.0
    return selected_count / total_context_blocks


def estimate_active_block_reduction(selected_ratio: float) -> float:
    if not 0.0 <= selected_ratio <= 1.0:
        raise ValueError("selected_ratio must be between 0 and 1")
    return 1.0 - selected_ratio


def summarize_scores(scores: Sequence[float]) -> dict[str, float] | None:
    if not scores:
        return None
    numeric_scores = [float(score) for score in scores]
    return {
        "minimum": min(numeric_scores),
        "maximum": max(numeric_scores),
        "mean": sum(numeric_scores) / len(numeric_scores),
    }


def build_phase12_shadow_event(
    *,
    request_id: str,
    layer_idx: int,
    context_token_count: int,
    block_size: int,
    total_context_blocks: int,
    ratio_policy: ShadowRatioPolicy,
    candidate_budget_blocks: int,
    selected_block_ids_by_score: Sequence[int],
    selector_policy: str,
    scores: Sequence[float] | None = None,
    sequence_id: str | None = None,
    step_idx: int | None = None,
) -> Phase12ShadowEvent:
    """Build a fail-closed, serializable shadow event."""

    if not request_id:
        raise ValueError("request_id is required")
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative")
    if context_token_count < 0:
        raise ValueError("context_token_count must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if total_context_blocks < 0:
        raise ValueError("total_context_blocks must be non-negative")
    if not 0 <= candidate_budget_blocks <= total_context_blocks:
        raise ValueError("candidate budget must be within total blocks")

    score_ids = list(selected_block_ids_by_score)
    if any(
        not isinstance(block_id, int)
        or isinstance(block_id, bool)
        or block_id < 0
        or block_id >= total_context_blocks
        for block_id in score_ids
    ):
        raise ValueError("selected block IDs must be valid context blocks")
    gather_ids = sort_selected_ids_for_gather(score_ids)
    ordering_valid = validate_ordering(score_ids, gather_ids)
    if not ordering_valid:
        raise ValueError("selected block IDs must be unique")
    if len(score_ids) > candidate_budget_blocks:
        raise ValueError("selected block count must not exceed budget")

    selected_ratio = compute_selected_ratio(
        len(score_ids),
        total_context_blocks,
    )
    selection = ShadowSelectionResult(
        selected_block_ids_by_score=tuple(score_ids),
        selected_block_ids_for_gather=tuple(gather_ids),
        selected_count=len(score_ids),
        selected_ratio=selected_ratio,
        estimated_active_block_reduction_ratio=(
            estimate_active_block_reduction(selected_ratio)
        ),
        ordering_valid=ordering_valid,
    )
    policy_data = ratio_policy.to_dict(layer_idx)
    policy_data.update({
        "min_budget": 1,
        "max_budget": None,
        "rounding": "ceil",
    })
    return Phase12ShadowEvent(
        request_id=request_id,
        sequence_id=sequence_id,
        layer_idx=layer_idx,
        step_idx=step_idx,
        context_token_count=context_token_count,
        block_size=block_size,
        total_context_blocks=total_context_blocks,
        ratio_policy_name=ratio_policy.name,
        ratio_policy=policy_data,
        candidate_budget_blocks=candidate_budget_blocks,
        selected_block_ids_by_score=list(
            selection.selected_block_ids_by_score
        ),
        selected_block_ids_for_gather=list(
            selection.selected_block_ids_for_gather
        ),
        selected_block_count=selection.selected_count,
        selected_ratio=selection.selected_ratio,
        estimated_active_block_reduction_ratio=(
            selection.estimated_active_block_reduction_ratio
        ),
        selector_policy=selector_policy,
        selector_scores_summary=summarize_scores(scores or ()),
        ordering_valid=selection.ordering_valid,
        caveats=[
            "synthetic shadow event only",
            "full KV remains allocated",
            "attention output is unchanged",
            "no measured runtime memory reduction",
        ],
    )
