#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""No-op-by-default runtime-facing helpers for Phase 12 shadow metadata."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from phase12_vllm_shadow_hook import (
    Phase12VllmShadowHook,
    maybe_get_shadow_hook_from_env,
)

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def is_phase12_shadow_enabled(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether Phase 12 shadow observation is explicitly enabled."""

    source = os.environ if env is None else env
    return source.get(
        "KIVO_PHASE12_SHADOW_ENABLED",
        "",
    ).strip().lower() in TRUE_VALUES


def _disabled_result(reason: str = "disabled") -> dict[str, Any]:
    return {
        "enabled": False,
        "event_written": False,
        "reason": reason,
        "output_jsonl": None,
        "warnings": 0,
        "errors": 0,
        "error": None,
        "event_summary": None,
        "shadow_only": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
    }


def _fail_closed_result(error: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "event_written": False,
        "reason": "touchpoint_error",
        "output_jsonl": None,
        "warnings": 1,
        "errors": 1,
        "error": error,
        "event_summary": None,
        "shadow_only": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
    }


def get_phase12_shadow_hook(
    env: Mapping[str, str] | None = None,
) -> Phase12VllmShadowHook | None:
    """Return an opt-in hook, or ``None`` for the default no-op path."""

    if not is_phase12_shadow_enabled(env):
        return None
    try:
        return maybe_get_shadow_hook_from_env(env)
    except Exception:
        return None


def observe_phase12_decode_shadow_metadata(
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
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Observe decode metadata if env-enabled; otherwise return no-op."""

    if not is_phase12_shadow_enabled(env):
        return _disabled_result()
    try:
        hook = get_phase12_shadow_hook(env)
        if hook is None:
            return _disabled_result("hook_unavailable")
        copied_metadata = dict(metadata or {})
        copied_metadata.setdefault("touchpoint", "decode_metadata")
        return hook.observe_decode_metadata(
            request_id=request_id,
            layer_idx=layer_idx,
            context_token_count=context_token_count,
            total_context_blocks=total_context_blocks,
            step_idx=step_idx,
            sequence_id=sequence_id,
            block_ids=None if block_ids is None else list(block_ids),
            scores=scores,
            metadata=copied_metadata,
        )
    except Exception as exc:
        return _fail_closed_result(str(exc))


def observe_phase12_block_table_shadow_metadata(
    *,
    request_id: str,
    layer_idx: int,
    context_token_count: int,
    total_context_blocks: int,
    block_ids: Sequence[int] | None = None,
    step_idx: int | None = None,
    sequence_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Observe copied block-table-like metadata without mutating the table."""

    return observe_phase12_decode_shadow_metadata(
        request_id=request_id,
        layer_idx=layer_idx,
        context_token_count=context_token_count,
        total_context_blocks=total_context_blocks,
        step_idx=step_idx,
        sequence_id=sequence_id,
        block_ids=None if block_ids is None else list(block_ids),
        scores=None,
        metadata={
            **dict(metadata or {}),
            "touchpoint": "block_table_metadata",
        },
        env=env,
    )
