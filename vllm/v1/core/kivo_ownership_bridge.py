# SPDX-License-Identifier: Apache-2.0

"""Fail-closed core ownership bridge decisions for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

_DEFAULT_ACTION = "plan_only"
_SUPPORTED_ACTIONS = {
    "plan_only",
    "mark_demoted_if_safe",
}


@dataclass(frozen=True)
class KivoOwnershipBridgeConfig:
    enabled: bool
    action: str
    require_block_table_applied: bool
    require_slot_mapping_refresh: bool


@dataclass(frozen=True)
class KivoOwnershipBridgeDecision:
    enabled: bool
    action: str
    request_id: str | None
    demote_block_ids: tuple[int, ...]
    visible_after_block_ids: tuple[int, ...]
    ownership_before_block_ids: tuple[int, ...]
    ownership_after_block_ids: tuple[int, ...]
    block_table_applied: bool
    slot_mapping_refresh_guaranteed: bool
    safe_to_mark_demoted: bool
    safe_to_free: bool
    blocker_reasons: dict[str, int]


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def current_kivo_ownership_bridge_config() -> KivoOwnershipBridgeConfig:
    enabled = _parse_bool_env("KIVO_KV_OWNERSHIP_BRIDGE_ENABLE", default=False)
    action = os.getenv("KIVO_KV_OWNERSHIP_BRIDGE_ACTION", _DEFAULT_ACTION)
    if not enabled:
        action = _DEFAULT_ACTION
    return KivoOwnershipBridgeConfig(
        enabled=enabled,
        action=action,
        require_block_table_applied=_parse_bool_env(
            "KIVO_KV_OWNERSHIP_BRIDGE_REQUIRE_BLOCK_TABLE_APPLIED",
            default=True,
        ),
        require_slot_mapping_refresh=_parse_bool_env(
            "KIVO_KV_OWNERSHIP_BRIDGE_REQUIRE_SLOT_MAPPING_REFRESH",
            default=True,
        ),
    )


def build_kivo_ownership_bridge_decision(
    *,
    request_id: str | None,
    demote_block_ids: Sequence[int],
    visible_after_block_ids: Sequence[int],
    ownership_before_block_ids: Sequence[int],
    protected_block_ids: Sequence[int] = (),
    block_table_applied: bool,
    slot_mapping_refresh_guaranteed: bool,
    ownership_mapping_available: bool,
    config: KivoOwnershipBridgeConfig | None = None,
) -> KivoOwnershipBridgeDecision:
    """Build a narrow worker-to-core ownership bridge decision."""
    if config is None:
        config = current_kivo_ownership_bridge_config()

    demote = tuple(int(block_id) for block_id in demote_block_ids)
    visible_after = tuple(int(block_id) for block_id in visible_after_block_ids)
    ownership_before = tuple(int(block_id) for block_id in ownership_before_block_ids)
    protected = tuple(int(block_id) for block_id in protected_block_ids)

    if not config.enabled:
        return KivoOwnershipBridgeDecision(
            enabled=False,
            action=_DEFAULT_ACTION,
            request_id=request_id,
            demote_block_ids=(),
            visible_after_block_ids=visible_after,
            ownership_before_block_ids=ownership_before,
            ownership_after_block_ids=ownership_before,
            block_table_applied=block_table_applied,
            slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
            safe_to_mark_demoted=False,
            safe_to_free=False,
            blocker_reasons={"disabled": 1},
        )

    blocker_reasons: dict[str, int] = {}
    if config.action not in _SUPPORTED_ACTIONS:
        blocker_reasons["invalid_bridge_action"] = 1

    if request_id is None:
        blocker_reasons["missing_request_id"] = 1
    if not ownership_mapping_available:
        blocker_reasons["ownership_mapping_unavailable"] = 1
    if not ownership_before:
        blocker_reasons["missing_ownership_before_blocks"] = 1
    if not visible_after:
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

    duplicate_demote = _duplicate_count(demote)
    if duplicate_demote:
        blocker_reasons["duplicate_demote_ids"] = duplicate_demote
    duplicate_visible_after = _duplicate_count(visible_after)
    if duplicate_visible_after:
        blocker_reasons["duplicate_visible_after_ids"] = duplicate_visible_after
    duplicate_ownership_before = _duplicate_count(ownership_before)
    if duplicate_ownership_before:
        blocker_reasons["duplicate_ownership_before_ids"] = duplicate_ownership_before

    ownership_before_set = set(ownership_before)
    visible_after_set = set(visible_after)
    protected_set = set(protected)

    demote_missing_from_ownership = tuple(
        block_id for block_id in demote if block_id not in ownership_before_set
    )
    if demote_missing_from_ownership:
        blocker_reasons["demote_ids_missing_from_ownership_before"] = len(
            demote_missing_from_ownership
        )

    demote_still_visible = tuple(
        block_id for block_id in demote if block_id in visible_after_set
    )
    if demote_still_visible:
        blocker_reasons["demote_ids_still_visible_after"] = len(
            demote_still_visible
        )

    visible_after_not_owned = tuple(
        block_id for block_id in visible_after if block_id not in ownership_before_set
    )
    if visible_after_not_owned:
        blocker_reasons["visible_after_not_subset_of_ownership_before"] = len(
            visible_after_not_owned
        )

    demote_overlaps_protected = tuple(
        block_id for block_id in demote if block_id in protected_set
    )
    if demote_overlaps_protected:
        blocker_reasons["demote_overlaps_protected"] = len(demote_overlaps_protected)

    ownership_after = tuple(
        block_id for block_id in ownership_before if block_id not in set(demote)
    )
    if not ownership_after:
        blocker_reasons["empty_ownership_after"] = 1

    safe_to_mark_demoted = False
    if config.action == "mark_demoted_if_safe":
        blocker_reasons["ownership_mark_demoted_not_implemented"] = 1

    return KivoOwnershipBridgeDecision(
        enabled=True,
        action=config.action,
        request_id=request_id,
        demote_block_ids=demote,
        visible_after_block_ids=visible_after,
        ownership_before_block_ids=ownership_before,
        ownership_after_block_ids=ownership_after,
        block_table_applied=block_table_applied,
        slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
        safe_to_mark_demoted=safe_to_mark_demoted,
        safe_to_free=False,
        blocker_reasons=blocker_reasons,
    )
