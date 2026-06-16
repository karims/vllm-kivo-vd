# SPDX-License-Identifier: Apache-2.0

"""Fail-closed runtime-to-core demotion mark adapter for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from vllm.v1.core.kivo_ownership_bridge import KivoOwnershipBridgeConfig

_DEFAULT_ACTION = "off"


@dataclass(frozen=True)
class KivoRuntimeDemotionMarkConfig:
    enabled: bool
    action: str
    require_block_table_applied: bool
    require_slot_mapping_refresh: bool


@dataclass(frozen=True)
class KivoRuntimeDemotionMarkSummary:
    enabled: bool
    action: str
    attempted_request_count: int
    marked_request_count: int
    blocked_request_count: int
    marked_block_count: int
    blocker_reasons: dict[str, int]


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def current_kivo_runtime_demotion_mark_config() -> KivoRuntimeDemotionMarkConfig:
    enabled = _parse_bool_env("KIVO_KV_RUNTIME_DEMOTION_MARK_ENABLE", default=False)
    action = os.getenv("KIVO_KV_RUNTIME_DEMOTION_MARK_ACTION", _DEFAULT_ACTION)
    if not enabled:
        action = _DEFAULT_ACTION
    return KivoRuntimeDemotionMarkConfig(
        enabled=enabled,
        action=action,
        require_block_table_applied=_parse_bool_env(
            "KIVO_KV_RUNTIME_DEMOTION_MARK_REQUIRE_BLOCK_TABLE_APPLIED",
            default=True,
        ),
        require_slot_mapping_refresh=_parse_bool_env(
            "KIVO_KV_RUNTIME_DEMOTION_MARK_REQUIRE_SLOT_MAPPING_REFRESH",
            default=True,
        ),
    )


def maybe_mark_demoted_blocks_after_block_table_apply(
    *,
    request_id: str | None,
    demote_block_ids: Sequence[int],
    visible_after_block_ids: Sequence[int],
    protected_block_ids: Sequence[int] = (),
    block_table_applied: bool,
    slot_mapping_refresh_guaranteed: bool,
    kv_cache_manager: object | None,
    config: KivoRuntimeDemotionMarkConfig | None = None,
) -> KivoRuntimeDemotionMarkSummary:
    """Attempt runtime-to-core demotion marking through a narrow manager API."""
    if config is None:
        config = current_kivo_runtime_demotion_mark_config()

    if not config.enabled or config.action == "off":
        return KivoRuntimeDemotionMarkSummary(
            enabled=False,
            action="off",
            attempted_request_count=0,
            marked_request_count=0,
            blocked_request_count=0,
            marked_block_count=0,
            blocker_reasons={"disabled": 1},
        )

    blocker_reasons: dict[str, int] = {}
    if config.action != "mark_demoted_after_block_table_apply":
        blocker_reasons["invalid_runtime_demotion_action"] = 1
    if request_id is None:
        blocker_reasons["missing_request_id"] = 1
    if config.require_block_table_applied and not block_table_applied:
        blocker_reasons["block_table_apply_required"] = 1
    if config.require_slot_mapping_refresh and not slot_mapping_refresh_guaranteed:
        blocker_reasons["slot_mapping_refresh_not_guaranteed"] = 1

    demote = tuple(int(block_id) for block_id in demote_block_ids)
    visible_after = tuple(int(block_id) for block_id in visible_after_block_ids)
    visible_after_set = set(visible_after)
    if any(block_id in visible_after_set for block_id in demote):
        blocker_reasons["demote_ids_still_visible_after"] = sum(
            1 for block_id in demote if block_id in visible_after_set
        )

    if kv_cache_manager is None:
        blocker_reasons["kv_cache_manager_unavailable"] = 1
    elif not hasattr(kv_cache_manager, "mark_kivo_demoted_blocks_if_safe"):
        blocker_reasons["kv_cache_manager_missing_mark_helper"] = 1

    if blocker_reasons:
        return KivoRuntimeDemotionMarkSummary(
            enabled=True,
            action=config.action,
            attempted_request_count=1,
            marked_request_count=0,
            blocked_request_count=1,
            marked_block_count=0,
            blocker_reasons=blocker_reasons,
        )

    bridge_config = KivoOwnershipBridgeConfig(
        enabled=True,
        action="mark_demoted_if_safe",
        require_block_table_applied=config.require_block_table_applied,
        require_slot_mapping_refresh=config.require_slot_mapping_refresh,
    )
    decision = kv_cache_manager.mark_kivo_demoted_blocks_if_safe(
        request_id,
        demote,
        visible_after_block_ids=visible_after,
        protected_block_ids=tuple(int(block_id) for block_id in protected_block_ids),
        block_table_applied=block_table_applied,
        slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
        config=bridge_config,
    )
    if not getattr(decision, "safe_to_mark_demoted", False):
        return KivoRuntimeDemotionMarkSummary(
            enabled=True,
            action=config.action,
            attempted_request_count=1,
            marked_request_count=0,
            blocked_request_count=1,
            marked_block_count=0,
            blocker_reasons=dict(getattr(decision, "blocker_reasons", {})),
        )

    return KivoRuntimeDemotionMarkSummary(
        enabled=True,
        action=config.action,
        attempted_request_count=1,
        marked_request_count=1,
        blocked_request_count=0,
        marked_block_count=len(tuple(getattr(decision, "demote_block_ids", ()))),
        blocker_reasons={},
    )
