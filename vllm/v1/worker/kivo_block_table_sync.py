# SPDX-License-Identifier: Apache-2.0

"""Plan-only worker block-table sync helpers for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

_DEFAULT_ACTION = "off"
_SUPPORTED_ACTIONS = {
    "off",
    "plan_filtered_view",
    "apply_filtered_view_if_safe",
}


@dataclass(frozen=True)
class KivoBlockTableSyncConfig:
    enabled: bool
    action: str
    require_exact_order: bool


@dataclass(frozen=True)
class KivoBlockTableSyncPlan:
    enabled: bool
    action: str
    request_id: str | None
    original_block_ids: tuple[int, ...]
    keep_block_ids: tuple[int, ...]
    filtered_block_ids: tuple[int, ...]
    removed_block_ids: tuple[int, ...]
    protected_block_ids: tuple[int, ...]
    blocker_reasons: dict[str, int]
    preserves_order: bool
    safe_to_apply: bool


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def current_kivo_block_table_sync_config() -> KivoBlockTableSyncConfig:
    enabled = _parse_bool_env("KIVO_KV_BLOCK_TABLE_SYNC_ENABLE", default=False)
    action = os.getenv("KIVO_KV_BLOCK_TABLE_SYNC_ACTION", _DEFAULT_ACTION)
    if not enabled:
        action = "off"
    return KivoBlockTableSyncConfig(
        enabled=enabled,
        action=action,
        require_exact_order=_parse_bool_env(
            "KIVO_KV_BLOCK_TABLE_SYNC_REQUIRE_EXACT_ORDER",
            default=True,
        ),
    )


def build_block_table_sync_plan(
    request_id: str | None,
    original_block_ids: Sequence[int],
    keep_block_ids: Sequence[int],
    protected_block_ids: Sequence[int] = (),
    *,
    action: str | None = None,
    config: KivoBlockTableSyncConfig | None = None,
) -> KivoBlockTableSyncPlan:
    """Build a fail-closed filtered row view without mutating block tables."""
    if config is None:
        config = current_kivo_block_table_sync_config()
    if action is not None and action != config.action:
        config = KivoBlockTableSyncConfig(
            enabled=config.enabled,
            action=action,
            require_exact_order=config.require_exact_order,
        )

    original = tuple(int(block_id) for block_id in original_block_ids)
    keep = tuple(int(block_id) for block_id in keep_block_ids)
    protected = tuple(int(block_id) for block_id in protected_block_ids)

    if not config.enabled or config.action == "off":
        return KivoBlockTableSyncPlan(
            enabled=False,
            action="off",
            request_id=request_id,
            original_block_ids=original,
            keep_block_ids=original,
            filtered_block_ids=original,
            removed_block_ids=(),
            protected_block_ids=(),
            blocker_reasons={"disabled": 1},
            preserves_order=True,
            safe_to_apply=False,
        )

    if config.action not in _SUPPORTED_ACTIONS:
        return KivoBlockTableSyncPlan(
            enabled=True,
            action=config.action,
            request_id=request_id,
            original_block_ids=original,
            keep_block_ids=original,
            filtered_block_ids=original,
            removed_block_ids=(),
            protected_block_ids=protected,
            blocker_reasons={"invalid_action_fail_closed": 1},
            preserves_order=True,
            safe_to_apply=False,
        )

    blocker_reasons: dict[str, int] = {}
    original_set = set(original)
    unknown_keep_ids = tuple(block_id for block_id in keep if block_id not in original_set)
    if unknown_keep_ids:
        blocker_reasons["keep_ids_not_in_original"] = len(unknown_keep_ids)

    keep_counts: dict[int, int] = {}
    for block_id in keep:
        keep_counts[block_id] = keep_counts.get(block_id, 0) + 1
    if any(count > 1 for count in keep_counts.values()):
        blocker_reasons["duplicate_keep_ids"] = sum(
            count - 1 for count in keep_counts.values() if count > 1
        )

    original_counts: dict[int, int] = {}
    for block_id in original:
        original_counts[block_id] = original_counts.get(block_id, 0) + 1
    if any(count > 1 for count in original_counts.values()):
        blocker_reasons["duplicate_original_ids"] = sum(
            count - 1 for count in original_counts.values() if count > 1
        )

    filtered: list[int] = [block_id for block_id in original if block_id in set(keep)]
    filtered_tuple = tuple(filtered)
    removed_tuple = tuple(block_id for block_id in original if block_id not in set(filtered))

    if not filtered_tuple:
        blocker_reasons["empty_filtered_view"] = 1

    missing_protected = tuple(
        block_id for block_id in protected if block_id not in set(filtered_tuple)
    )
    if missing_protected:
        blocker_reasons["protected_ids_missing_from_filtered"] = len(missing_protected)

    preserves_order = filtered == [block_id for block_id in original if block_id in set(filtered)]
    if config.require_exact_order and not preserves_order:
        blocker_reasons["order_not_preserved"] = 1

    safe_to_apply = config.action == "apply_filtered_view_if_safe" and not blocker_reasons
    if config.action == "apply_filtered_view_if_safe":
        blocker_reasons["apply_not_enabled_locally"] = 1
        safe_to_apply = False

    return KivoBlockTableSyncPlan(
        enabled=True,
        action=config.action,
        request_id=request_id,
        original_block_ids=original,
        keep_block_ids=keep,
        filtered_block_ids=filtered_tuple,
        removed_block_ids=removed_tuple,
        protected_block_ids=protected,
        blocker_reasons=blocker_reasons,
        preserves_order=preserves_order,
        safe_to_apply=safe_to_apply,
    )
