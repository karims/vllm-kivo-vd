#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Passive, standalone observer scaffold for Phase 12 shadow events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from phase12_shadow_events import (
    Phase12ShadowEvent,
    build_phase12_shadow_event,
    derive_layer_budget,
    parse_ratio_policy,
)

DEFAULT_RATIO_POLICY = "balanced=0:0.60,5:0.45,8:0.45,11:0.60"


@dataclass(frozen=True)
class Phase12ShadowObserverConfig:
    enabled: bool = False
    output_jsonl: str = "outputs/kivo_vd/phase12_shadow_events.jsonl"
    ratio_policy: str = DEFAULT_RATIO_POLICY
    selector_policy: str = "query_key_block_score"
    block_size: int = 16
    min_budget: int = 1
    max_budget: int | None = None
    preview_only: bool = True
    shadow_only: bool = True
    active_routing: bool = False
    measured_runtime_reduction: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.min_budget < 0:
            raise ValueError("min_budget must be non-negative")
        if self.max_budget is not None and self.max_budget < self.min_budget:
            raise ValueError("max_budget must not be below min_budget")
        if not self.output_jsonl:
            raise ValueError("output_jsonl is required")
        if not self.shadow_only:
            raise ValueError("Phase 12 observer requires shadow_only=true")
        if self.active_routing:
            raise ValueError("Phase 12 observer requires active_routing=false")
        if self.measured_runtime_reduction:
            raise ValueError(
                "Phase 12 observer cannot claim measured runtime reduction"
            )
        parse_ratio_policy(self.ratio_policy)


@dataclass(frozen=True)
class Phase12ShadowObservation:
    request_id: str
    layer_idx: int
    context_token_count: int
    total_context_blocks: int
    block_ids: Sequence[int] | None = None
    scores: Sequence[float] | None = None
    step_idx: int | None = None
    sequence_id: str | None = None


class Phase12ShadowObserver:
    """Build and append shadow events without changing caller-owned state."""

    def __init__(self, config: Phase12ShadowObserverConfig) -> None:
        self.config = config
        self.ratio_policy = parse_ratio_policy(config.ratio_policy)
        self.events_seen = 0
        self.events_written = 0
        self.invalid_events = 0
        self.warnings = 0
        self.warning_messages: list[str] = []

    def observe(
        self,
        observation: Phase12ShadowObservation,
    ) -> Phase12ShadowEvent | None:
        self.events_seen += 1
        if not self.config.enabled:
            return None

        try:
            event = self._build_event(observation)
            self._append_event(event)
        except (OSError, TypeError, ValueError) as exc:
            self.invalid_events += 1
            self.warnings += 1
            self.warning_messages.append(str(exc))
            return None

        self.events_written += 1
        return event

    def get_counters(self) -> dict[str, int]:
        return {
            "events_seen": self.events_seen,
            "events_written": self.events_written,
            "invalid_events": self.invalid_events,
            "warnings": self.warnings,
        }

    def _build_event(
        self,
        observation: Phase12ShadowObservation,
    ) -> Phase12ShadowEvent:
        total_blocks = observation.total_context_blocks
        block_ids = (
            list(range(total_blocks))
            if observation.block_ids is None
            else list(observation.block_ids)
        )
        if len(block_ids) != total_blocks:
            raise ValueError(
                "block_ids length must equal total_context_blocks"
            )
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block_ids must be unique")

        budget = derive_layer_budget(
            total_context_blocks=total_blocks,
            layer_idx=observation.layer_idx,
            ratio_policy=self.ratio_policy,
            min_budget=self.config.min_budget,
            max_budget=self.config.max_budget,
        )
        preview_only = observation.scores is None or self.config.preview_only
        if observation.scores is None:
            ranked_ids = sorted(block_ids, reverse=True)
            score_summary_values: Sequence[float] | None = None
        else:
            scores = list(observation.scores)
            if len(scores) != len(block_ids):
                raise ValueError("scores length must equal block_ids length")
            ranked_pairs = sorted(
                zip(block_ids, scores, strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )
            ranked_ids = [block_id for block_id, _ in ranked_pairs]
            score_summary_values = scores

        return build_phase12_shadow_event(
            request_id=observation.request_id,
            sequence_id=observation.sequence_id,
            layer_idx=observation.layer_idx,
            step_idx=observation.step_idx,
            context_token_count=observation.context_token_count,
            block_size=self.config.block_size,
            total_context_blocks=total_blocks,
            ratio_policy=self.ratio_policy,
            candidate_budget_blocks=budget,
            selected_block_ids_by_score=ranked_ids[:budget],
            selector_policy=self.config.selector_policy,
            scores=score_summary_values,
            min_budget=self.config.min_budget,
            max_budget=self.config.max_budget,
            preview_only=preview_only,
            caveats=(
                "passive Phase 12 shadow observation only",
                "full KV remains allocated",
                "normal attention remains unchanged",
                "no measured runtime memory reduction",
            ),
        )

    def _append_event(self, event: Phase12ShadowEvent) -> None:
        output_path = Path(self.config.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(
                json.dumps(event.to_dict(), sort_keys=True) + "\n"
            )
