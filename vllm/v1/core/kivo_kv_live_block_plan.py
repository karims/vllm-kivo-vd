# SPDX-License-Identifier: Apache-2.0

"""Plan-only live KV block demotion metadata for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from vllm.v1.core.kivo_kv_retention_policy import KivoKVRetentionDecision

_DEFAULT_ACTION = "plan_live_demotion_only"
_SUPPORTED_ACTIONS = {
    "plan_live_demotion_only",
    "apply_live_demotion_if_safe",
}
_DEFAULT_REQUIRE_BLOCK_TABLE_SYNC = True
_DEFAULT_PROTECT_RECENT_BLOCKS = 4
_DEFAULT_MIN_BLOCKS_BEFORE_ACTION = 8


@dataclass(frozen=True)
class KivoKVLiveBlockPlanConfig:
    enabled: bool
    action: str
    require_block_table_sync: bool
    protect_recent_blocks: int
    min_blocks_before_action: int


@dataclass(frozen=True)
class KivoKVLiveBlockPlan:
    enabled: bool
    action: str
    request_id: str | None
    all_block_ids: tuple[int, ...]
    keep_block_ids: tuple[int, ...]
    candidate_demote_block_ids: tuple[int, ...]
    protected_block_ids: tuple[int, ...]
    blocker_reasons: dict[str, int]
    can_mutate_ownership: bool
    can_mutate_block_table: bool
    safe_to_apply: bool


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def _parse_int_env(name: str, *, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def current_kivo_live_block_plan_config() -> KivoKVLiveBlockPlanConfig:
    return KivoKVLiveBlockPlanConfig(
        enabled=_parse_bool_env("KIVO_KV_LIVE_DEMOTION_ENABLE", default=False),
        action=os.getenv("KIVO_KV_LIVE_DEMOTION_ACTION", _DEFAULT_ACTION),
        require_block_table_sync=_parse_bool_env(
            "KIVO_KV_LIVE_DEMOTION_REQUIRE_BLOCK_TABLE_SYNC",
            default=_DEFAULT_REQUIRE_BLOCK_TABLE_SYNC,
        ),
        protect_recent_blocks=_parse_int_env(
            "KIVO_KV_LIVE_DEMOTION_PROTECT_RECENT_BLOCKS",
            default=_DEFAULT_PROTECT_RECENT_BLOCKS,
            minimum=0,
        ),
        min_blocks_before_action=_parse_int_env(
            "KIVO_KV_LIVE_DEMOTION_MIN_BLOCKS_BEFORE_ACTION",
            default=_DEFAULT_MIN_BLOCKS_BEFORE_ACTION,
            minimum=0,
        ),
    )


def build_kivo_live_block_plan(
    all_block_ids: Sequence[int],
    retention_decision: KivoKVRetentionDecision,
    *,
    request_id: str | None = None,
    shared_block_ids: Sequence[int] | None = None,
    block_table_sync_available: bool = False,
    ownership_mutation_available: bool = True,
    config: KivoKVLiveBlockPlanConfig | None = None,
) -> KivoKVLiveBlockPlan:
    """Build a fail-closed live demotion plan without mutating runtime state."""
    if config is None:
        config = current_kivo_live_block_plan_config()

    block_ids = tuple(int(block_id) for block_id in all_block_ids)
    shared_ids = {
        int(block_id) for block_id in (shared_block_ids or ()) if block_id is not None
    }

    if not config.enabled:
        return KivoKVLiveBlockPlan(
            enabled=False,
            action="plan_live_demotion_only",
            request_id=request_id,
            all_block_ids=block_ids,
            keep_block_ids=block_ids,
            candidate_demote_block_ids=(),
            protected_block_ids=(),
            blocker_reasons={"disabled": 1},
            can_mutate_ownership=False,
            can_mutate_block_table=False,
            safe_to_apply=False,
        )

    if config.action not in _SUPPORTED_ACTIONS:
        return KivoKVLiveBlockPlan(
            enabled=True,
            action=config.action,
            request_id=request_id,
            all_block_ids=block_ids,
            keep_block_ids=block_ids,
            candidate_demote_block_ids=(),
            protected_block_ids=block_ids,
            blocker_reasons={"invalid_action_fail_closed": 1},
            can_mutate_ownership=False,
            can_mutate_block_table=False,
            safe_to_apply=False,
        )

    candidate_demote: list[int] = []
    protected: list[int] = []
    blocker_reasons: dict[str, int] = {}

    if len(block_ids) < config.min_blocks_before_action:
        blocker_reasons["below_min_blocks_before_action"] = 1

    if config.require_block_table_sync and not block_table_sync_available:
        blocker_reasons["block_table_sync_unavailable"] = 1

    if not ownership_mutation_available:
        blocker_reasons["ownership_mutation_unavailable"] = 1

    retention_keep = set(retention_decision.keep_block_ids)
    retention_drop = set(retention_decision.candidate_drop_block_ids)

    for block_id in block_ids:
        if block_id in retention_keep:
            protected.append(block_id)
            continue
        if block_id in shared_ids:
            protected.append(block_id)
            blocker_reasons["shared_block_refcnt"] = (
                blocker_reasons.get("shared_block_refcnt", 0) + 1
            )
            continue
        if block_id not in retention_drop:
            protected.append(block_id)
            blocker_reasons["retention_policy_protected"] = (
                blocker_reasons.get("retention_policy_protected", 0) + 1
            )
            continue
        candidate_demote.append(block_id)

    keep_block_ids = tuple(
        block_id for block_id in block_ids if block_id not in set(candidate_demote)
    )
    protected_block_ids = tuple(
        block_id for block_id in block_ids if block_id in set(protected)
    )
    candidate_demote_block_ids = tuple(candidate_demote)

    if not candidate_demote_block_ids:
        blocker_reasons["no_live_demote_candidates"] = 1

    can_mutate_ownership = ownership_mutation_available and not bool(shared_ids)
    can_mutate_block_table = block_table_sync_available
    safe_to_apply = (
        config.action == "apply_live_demotion_if_safe"
        and not blocker_reasons
        and can_mutate_ownership
        and (can_mutate_block_table or not config.require_block_table_sync)
    )

    return KivoKVLiveBlockPlan(
        enabled=True,
        action=config.action,
        request_id=request_id,
        all_block_ids=block_ids,
        keep_block_ids=keep_block_ids,
        candidate_demote_block_ids=candidate_demote_block_ids,
        protected_block_ids=protected_block_ids,
        blocker_reasons=blocker_reasons,
        can_mutate_ownership=can_mutate_ownership,
        can_mutate_block_table=can_mutate_block_table,
        safe_to_apply=safe_to_apply,
    )
