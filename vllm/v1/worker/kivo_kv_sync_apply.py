# SPDX-License-Identifier: Apache-2.0

"""Conservative KV sync-apply coordinator for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.kivo_block_table_sync import (
    KivoBlockTableSyncConfig,
    KivoBlockTableSyncPlan,
    build_block_table_sync_plan,
)

_DEFAULT_ACTION = "plan_only"
_SUPPORTED_ACTIONS = {
    "plan_only",
    "apply_block_table_only",
    "apply_block_table_and_mark_ownership",
}


@dataclass(frozen=True)
class KivoKVSyncApplyConfig:
    enabled: bool
    action: str
    require_slot_mapping_refresh: bool


@dataclass(frozen=True)
class KivoKVSyncApplyDecision:
    enabled: bool
    action: str
    request_id: str | None
    original_block_ids: tuple[int, ...]
    keep_block_ids: tuple[int, ...]
    demote_block_ids: tuple[int, ...]
    filtered_block_ids: tuple[int, ...]
    block_table_safe_to_apply: bool
    ownership_safe_to_apply: bool
    slot_mapping_refresh_required: bool
    slot_mapping_refresh_available: bool
    safe_to_apply: bool
    blocker_reasons: dict[str, int]


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def current_kivo_kv_sync_apply_config() -> KivoKVSyncApplyConfig:
    return KivoKVSyncApplyConfig(
        enabled=_parse_bool_env("KIVO_KV_SYNC_APPLY_ENABLE", default=False),
        action=os.getenv("KIVO_KV_SYNC_APPLY_ACTION", _DEFAULT_ACTION),
        require_slot_mapping_refresh=_parse_bool_env(
            "KIVO_KV_SYNC_APPLY_REQUIRE_SLOT_MAPPING_REFRESH",
            default=True,
        ),
    )


def build_kivo_kv_sync_apply_decision(
    request_id: str | None,
    original_block_ids: Sequence[int],
    keep_block_ids: Sequence[int],
    demote_block_ids: Sequence[int],
    protected_block_ids: Sequence[int] = (),
    *,
    slot_mapping_refresh_available: bool = False,
    config: KivoKVSyncApplyConfig | None = None,
) -> KivoKVSyncApplyDecision:
    if config is None:
        config = current_kivo_kv_sync_apply_config()

    original = tuple(int(block_id) for block_id in original_block_ids)
    keep = tuple(int(block_id) for block_id in keep_block_ids)
    demote = tuple(int(block_id) for block_id in demote_block_ids)
    protected = tuple(int(block_id) for block_id in protected_block_ids)

    if not config.enabled:
        return KivoKVSyncApplyDecision(
            enabled=False,
            action="plan_only",
            request_id=request_id,
            original_block_ids=original,
            keep_block_ids=original,
            demote_block_ids=(),
            filtered_block_ids=original,
            block_table_safe_to_apply=False,
            ownership_safe_to_apply=False,
            slot_mapping_refresh_required=config.require_slot_mapping_refresh,
            slot_mapping_refresh_available=slot_mapping_refresh_available,
            safe_to_apply=False,
            blocker_reasons={"disabled": 1},
        )

    blocker_reasons: dict[str, int] = {}
    if config.action not in _SUPPORTED_ACTIONS:
        blocker_reasons["invalid_action_fail_closed"] = 1

    keep_set = set(keep)
    protected_set = set(protected)
    demote_set = set(demote)

    overlap_keep_demote = keep_set & demote_set
    if overlap_keep_demote:
        blocker_reasons["demote_overlaps_keep"] = len(overlap_keep_demote)

    overlap_protected_demote = protected_set & demote_set
    if overlap_protected_demote:
        blocker_reasons["demote_overlaps_protected"] = len(overlap_protected_demote)

    block_table_plan: KivoBlockTableSyncPlan = build_block_table_sync_plan(
        request_id,
        original,
        keep,
        protected_block_ids=protected,
        config=KivoBlockTableSyncConfig(
            enabled=config.enabled,
            action=(
                "plan_filtered_view"
                if config.action in {"plan_only", "apply_block_table_only"}
                else "apply_filtered_view_if_safe"
            ),
            require_exact_order=True,
        ),
    )
    for reason, count in block_table_plan.blocker_reasons.items():
        blocker_reasons[reason] = blocker_reasons.get(reason, 0) + count

    slot_mapping_refresh_required = config.require_slot_mapping_refresh
    if slot_mapping_refresh_required and not slot_mapping_refresh_available:
        blocker_reasons["slot_mapping_refresh_unavailable"] = 1

    block_table_safe_to_apply = (
        config.action == "apply_block_table_only" and not blocker_reasons
    )
    ownership_safe_to_apply = False
    if config.action == "apply_block_table_and_mark_ownership":
        blocker_reasons["ownership_apply_not_enabled_locally"] = 1

    safe_to_apply = block_table_safe_to_apply and (
        not slot_mapping_refresh_required or slot_mapping_refresh_available
    )

    return KivoKVSyncApplyDecision(
        enabled=True,
        action=config.action,
        request_id=request_id,
        original_block_ids=original,
        keep_block_ids=keep,
        demote_block_ids=demote,
        filtered_block_ids=block_table_plan.filtered_block_ids,
        block_table_safe_to_apply=block_table_safe_to_apply,
        ownership_safe_to_apply=ownership_safe_to_apply,
        slot_mapping_refresh_required=slot_mapping_refresh_required,
        slot_mapping_refresh_available=slot_mapping_refresh_available,
        safe_to_apply=safe_to_apply,
        blocker_reasons=blocker_reasons,
    )


def apply_block_table_only_if_safe(
    block_table: BlockTable,
    row_idx: int,
    decision: KivoKVSyncApplyDecision,
) -> bool:
    """Apply a filtered row only in a direct local BlockTable context."""
    if not decision.enabled:
        return False
    if decision.action != "apply_block_table_only":
        return False
    if not decision.block_table_safe_to_apply:
        return False
    return block_table.replace_row_block_ids_if_safe(
        row_idx, decision.filtered_block_ids
    )
