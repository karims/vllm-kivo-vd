#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, fail-closed bridge to the passive Phase 12 shadow observer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from phase12_shadow_observer import (
    DEFAULT_RATIO_POLICY,
    Phase12ShadowObservation,
    Phase12ShadowObserver,
    Phase12ShadowObserverConfig,
)

DEFAULT_OUTPUT = "outputs/kivo_vd/phase12_vllm_shadow_events.jsonl"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class Phase12VllmShadowHookConfig:
    enabled: bool = False
    output_jsonl: str = DEFAULT_OUTPUT
    ratio_policy: str = DEFAULT_RATIO_POLICY
    selector_policy: str = "query_key_block_score"
    block_size: int = 16
    min_budget: int = 1
    max_budget: int | None = None
    preview_only: bool = True


def _parse_bool(value: str, variable: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"{variable} must be one of: "
        f"{', '.join(sorted(TRUE_VALUES | FALSE_VALUES))}"
    )


def _parse_int(
    value: str,
    variable: str,
    *,
    minimum: int,
) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{variable} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{variable} must be at least {minimum}")
    return parsed


def build_config_from_env(
    env: Mapping[str, str] | None = None,
) -> Phase12VllmShadowHookConfig:
    """Build hook configuration without importing vLLM runtime modules."""

    source = os.environ if env is None else env
    enabled = _parse_bool(
        source.get("KIVO_PHASE12_SHADOW_ENABLED", "false"),
        "KIVO_PHASE12_SHADOW_ENABLED",
    )
    preview_only = _parse_bool(
        source.get("KIVO_PHASE12_PREVIEW_ONLY", "true"),
        "KIVO_PHASE12_PREVIEW_ONLY",
    )
    block_size = _parse_int(
        source.get("KIVO_PHASE12_BLOCK_SIZE", "16"),
        "KIVO_PHASE12_BLOCK_SIZE",
        minimum=1,
    )
    min_budget = _parse_int(
        source.get("KIVO_PHASE12_MIN_BUDGET", "1"),
        "KIVO_PHASE12_MIN_BUDGET",
        minimum=0,
    )
    raw_max_budget = source.get("KIVO_PHASE12_MAX_BUDGET", "").strip()
    max_budget = (
        _parse_int(
            raw_max_budget,
            "KIVO_PHASE12_MAX_BUDGET",
            minimum=0,
        )
        if raw_max_budget
        else None
    )
    if max_budget is not None and max_budget < min_budget:
        raise ValueError(
            "KIVO_PHASE12_MAX_BUDGET must not be below "
            "KIVO_PHASE12_MIN_BUDGET"
        )
    return Phase12VllmShadowHookConfig(
        enabled=enabled,
        output_jsonl=source.get(
            "KIVO_PHASE12_SHADOW_OUTPUT",
            DEFAULT_OUTPUT,
        ),
        ratio_policy=source.get(
            "KIVO_PHASE12_RATIO_POLICY",
            DEFAULT_RATIO_POLICY,
        ),
        selector_policy=source.get(
            "KIVO_PHASE12_SELECTOR_POLICY",
            "query_key_block_score",
        ),
        block_size=block_size,
        min_budget=min_budget,
        max_budget=max_budget,
        preview_only=preview_only,
    )


class Phase12VllmShadowHook:
    """Manual vLLM-facing API that can only emit passive shadow events."""

    def __init__(
        self,
        config: Phase12VllmShadowHookConfig,
        *,
        initialization_error: str | None = None,
    ) -> None:
        self.config = config
        self.initialization_error = initialization_error
        self.errors = 1 if initialization_error else 0
        self.observer: Phase12ShadowObserver | None = None
        if initialization_error is None:
            try:
                self.observer = Phase12ShadowObserver(
                    Phase12ShadowObserverConfig(
                        enabled=config.enabled,
                        output_jsonl=config.output_jsonl,
                        ratio_policy=config.ratio_policy,
                        selector_policy=config.selector_policy,
                        block_size=config.block_size,
                        min_budget=config.min_budget,
                        max_budget=config.max_budget,
                        preview_only=config.preview_only,
                        shadow_only=True,
                        active_routing=False,
                        measured_runtime_reduction=False,
                    )
                )
            except (TypeError, ValueError) as exc:
                self.initialization_error = str(exc)
                self.errors += 1

    def observe_decode_metadata(
        self,
        *,
        request_id: str,
        layer_idx: int,
        context_token_count: int,
        total_context_blocks: int,
        step_idx: int | None = None,
        sequence_id: str | None = None,
        block_ids: Sequence[int] | None = None,
        scores: Mapping[int, float] | Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Observe copied decode metadata and never raise into the caller."""

        if not self.config.enabled:
            return self._result(
                event_written=False,
                reason="disabled",
                error=self.initialization_error,
            )
        if self.observer is None:
            return self._result(
                event_written=False,
                reason="initialization_failed",
                error=self.initialization_error,
            )

        try:
            copied_block_ids = (
                None if block_ids is None else list(block_ids)
            )
            normalized_scores = self._normalize_scores(
                scores=scores,
                block_ids=copied_block_ids,
                total_context_blocks=total_context_blocks,
            )
            event = self.observer.observe(Phase12ShadowObservation(
                request_id=request_id,
                sequence_id=sequence_id,
                layer_idx=layer_idx,
                step_idx=step_idx,
                context_token_count=context_token_count,
                total_context_blocks=total_context_blocks,
                block_ids=copied_block_ids,
                scores=normalized_scores,
            ))
            if event is None:
                self.errors += 1
                error = (
                    self.observer.warning_messages[-1]
                    if self.observer.warning_messages
                    else "observer did not emit an event"
                )
                return self._result(
                    event_written=False,
                    reason="observer_rejected_metadata",
                    error=error,
                )
            event_dict = event.to_dict()
            return self._result(
                event_written=True,
                reason="event_written",
                event_summary={
                    "request_id": event.request_id,
                    "layer_idx": event.layer_idx,
                    "selected_block_count": event.selected_block_count,
                    "selected_ratio": event.selected_ratio,
                    "preview_only": event.preview_only,
                    "ordering_valid": event.ordering_valid,
                    "shadow_only": event.shadow_only,
                    "active_routing": event.active_routing,
                    "measured_runtime_reduction": (
                        event.measured_runtime_reduction
                    ),
                    "metadata_keys": sorted((metadata or {}).keys()),
                    "selected_block_ids_by_score": event_dict[
                        "selected_block_ids_by_score"
                    ],
                    "selected_block_ids_for_gather": event_dict[
                        "selected_block_ids_for_gather"
                    ],
                },
            )
        except Exception as exc:
            self.errors += 1
            return self._result(
                event_written=False,
                reason="hook_error",
                error=str(exc),
            )

    def _normalize_scores(
        self,
        *,
        scores: Mapping[int, float] | Sequence[float] | None,
        block_ids: list[int] | None,
        total_context_blocks: int,
    ) -> list[float] | None:
        if scores is None:
            return None
        effective_ids = (
            list(range(total_context_blocks))
            if block_ids is None
            else block_ids
        )
        if isinstance(scores, Mapping):
            if set(scores) != set(effective_ids):
                raise ValueError(
                    "score mapping keys must match the block ID set"
                )
            return [float(scores[block_id]) for block_id in effective_ids]
        return [float(score) for score in scores]

    def _result(
        self,
        *,
        event_written: bool,
        reason: str,
        error: str | None = None,
        event_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        warnings = (
            self.observer.get_counters()["warnings"]
            if self.observer is not None
            else self.errors
        )
        return {
            "enabled": self.config.enabled,
            "event_written": event_written,
            "reason": reason,
            "output_jsonl": self.config.output_jsonl,
            "warnings": warnings,
            "errors": self.errors,
            "error": error,
            "event_summary": event_summary,
            "shadow_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
        }


def maybe_get_shadow_hook_from_env(
    env: Mapping[str, str] | None = None,
) -> Phase12VllmShadowHook:
    """Return a callable hook even when env configuration is invalid."""

    try:
        config = build_config_from_env(env)
        return Phase12VllmShadowHook(config)
    except (TypeError, ValueError) as exc:
        return Phase12VllmShadowHook(
            Phase12VllmShadowHookConfig(enabled=False),
            initialization_error=str(exc),
        )
