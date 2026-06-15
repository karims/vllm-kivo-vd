# SPDX-License-Identifier: Apache-2.0

"""Fail-closed paired live ownership apply decisions for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

_DEFAULT_ACTION = "plan_paired_apply"
_SUPPORTED_ACTIONS = {
    "plan_paired_apply",
    "apply_block_table_then_mark_ownership",
}


@dataclass(frozen=True)
class KivoLiveOwnershipApplyConfig:
    enabled: bool
    action: str
    policy: str
    keep_recent_blocks: int
    max_full_blocks: int
    require_block_table_applied: bool
    require_slot_mapping_refresh: bool


@dataclass(frozen=True)
class KivoLiveOwnershipApplyDecision:
    enabled: bool
    action: str
    request_id: str | None
    visible_before_block_ids: tuple[int, ...]
    visible_after_block_ids: tuple[int, ...]
    candidate_demote_block_ids: tuple[int, ...]
    ownership_mutation_block_ids: tuple[int, ...]
    block_table_applied: bool
    slot_mapping_refresh_guaranteed: bool
    safe_to_mutate_ownership: bool
    blocker_reasons: dict[str, int]


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


def current_kivo_live_ownership_apply_config() -> KivoLiveOwnershipApplyConfig:
    enabled = _parse_bool_env("KIVO_KV_LIVE_APPLY_ENABLE", default=False)
    action = os.getenv("KIVO_KV_LIVE_APPLY_ACTION", _DEFAULT_ACTION)
    if not enabled:
        action = _DEFAULT_ACTION
    return KivoLiveOwnershipApplyConfig(
        enabled=enabled,
        action=action,
        policy=os.getenv("KIVO_KV_LIVE_APPLY_POLICY", "recent_only"),
        keep_recent_blocks=_parse_int_env(
            "KIVO_KV_LIVE_APPLY_KEEP_RECENT_BLOCKS", default=4, minimum=0
        ),
        max_full_blocks=_parse_int_env(
            "KIVO_KV_LIVE_APPLY_MAX_FULL_BLOCKS", default=64, minimum=1
        ),
        require_block_table_applied=_parse_bool_env(
            "KIVO_KV_LIVE_APPLY_REQUIRE_BLOCK_TABLE_APPLIED", default=True
        ),
        require_slot_mapping_refresh=_parse_bool_env(
            "KIVO_KV_LIVE_APPLY_REQUIRE_SLOT_MAPPING_REFRESH", default=True
        ),
    )


def build_kivo_live_ownership_apply_decision(
    *,
    request_id: str | None,
    visible_before_block_ids: Sequence[int],
    visible_after_block_ids: Sequence[int],
    candidate_demote_block_ids: Sequence[int],
    protected_block_ids: Sequence[int] = (),
    block_table_applied: bool,
    slot_mapping_refresh_guaranteed: bool,
    ownership_mapping_available: bool,
    config: KivoLiveOwnershipApplyConfig | None = None,
) -> KivoLiveOwnershipApplyDecision:
    """Build a paired live-ownership decision without mutating runtime state."""
    if config is None:
        config = current_kivo_live_ownership_apply_config()

    before = tuple(int(block_id) for block_id in visible_before_block_ids)
    after = tuple(int(block_id) for block_id in visible_after_block_ids)
    demote = tuple(int(block_id) for block_id in candidate_demote_block_ids)
    protected = tuple(int(block_id) for block_id in protected_block_ids)

    if not config.enabled:
        return KivoLiveOwnershipApplyDecision(
            enabled=False,
            action=_DEFAULT_ACTION,
            request_id=request_id,
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=(),
            ownership_mutation_block_ids=(),
            block_table_applied=block_table_applied,
            slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
            safe_to_mutate_ownership=False,
            blocker_reasons={"disabled": 1},
        )

    blocker_reasons: dict[str, int] = {}
    if config.action not in _SUPPORTED_ACTIONS:
        blocker_reasons["invalid_live_apply_action"] = 1

    if request_id is None:
        blocker_reasons["missing_request_id"] = 1
    if not ownership_mapping_available:
        blocker_reasons["ownership_mapping_unavailable"] = 1
    if not before:
        blocker_reasons["missing_visible_before_blocks"] = 1
    if not after:
        blocker_reasons["empty_visible_after_blocks"] = 1
    if config.require_block_table_applied and not block_table_applied:
        blocker_reasons["block_table_apply_required"] = 1
    if config.require_slot_mapping_refresh and not slot_mapping_refresh_guaranteed:
        blocker_reasons["slot_mapping_refresh_not_guaranteed"] = 1

    def _duplicate_count(values: tuple[int, ...]) -> int:
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return sum(count - 1 for count in counts.values() if count > 1)

    duplicate_before = _duplicate_count(before)
    if duplicate_before:
        blocker_reasons["duplicate_visible_before_ids"] = duplicate_before
    duplicate_after = _duplicate_count(after)
    if duplicate_after:
        blocker_reasons["duplicate_visible_after_ids"] = duplicate_after
    duplicate_demote = _duplicate_count(demote)
    if duplicate_demote:
        blocker_reasons["duplicate_candidate_demote_ids"] = duplicate_demote

    before_set = set(before)
    after_set = set(after)
    demote_set = set(demote)
    protected_set = set(protected)

    demote_not_in_before = tuple(block_id for block_id in demote if block_id not in before_set)
    if demote_not_in_before:
        blocker_reasons["candidate_demote_not_visible_before"] = len(demote_not_in_before)

    demote_still_visible = tuple(block_id for block_id in demote if block_id in after_set)
    if demote_still_visible:
        blocker_reasons["candidate_demote_still_visible_after"] = len(demote_still_visible)

    missing_protected = tuple(
        block_id for block_id in protected if block_id not in after_set
    )
    if missing_protected:
        blocker_reasons["protected_ids_missing_after"] = len(missing_protected)

    ownership_mutation_block_ids = tuple(
        block_id
        for block_id in demote
        if block_id in before_set and block_id not in after_set
    )

    overlap_protected = tuple(
        block_id for block_id in ownership_mutation_block_ids if block_id in protected_set
    )
    if overlap_protected:
        blocker_reasons["ownership_mutation_overlaps_protected"] = len(overlap_protected)

    if not ownership_mutation_block_ids:
        blocker_reasons["no_ownership_mutation_candidates"] = 1

    safe_to_mutate_ownership = False
    if config.action == "apply_block_table_then_mark_ownership":
        blocker_reasons["ownership_mutation_not_enabled_locally"] = 1

    return KivoLiveOwnershipApplyDecision(
        enabled=True,
        action=config.action,
        request_id=request_id,
        visible_before_block_ids=before,
        visible_after_block_ids=after,
        candidate_demote_block_ids=demote,
        ownership_mutation_block_ids=ownership_mutation_block_ids,
        block_table_applied=block_table_applied,
        slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
        safe_to_mutate_ownership=safe_to_mutate_ownership,
        blocker_reasons=blocker_reasons,
    )
