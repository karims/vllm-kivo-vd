# SPDX-License-Identifier: Apache-2.0

"""Gated runtime-facing block-table-only apply helpers for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from vllm.v1.core.kivo_live_ownership_apply import (
    KivoLiveOwnershipApplyConfig,
    build_kivo_live_ownership_apply_decision,
    current_kivo_live_ownership_apply_config,
)
from vllm.v1.core.kivo_kv_block_score_store import get_block_scores
from vllm.v1.core.kivo_kv_retention_policy import (
    KivoKVRetentionConfig,
    decide_kv_retention,
)
from vllm.v1.core.kivo_kv_live_block_plan import (
    KivoKVLiveBlockPlanConfig,
    build_kivo_live_block_plan,
)
from vllm.v1.core.kivo_ownership_bridge import (
    current_kivo_ownership_bridge_config,
)
from vllm.v1.worker.kivo_kv_sync_apply import (
    KivoKVSyncApplyConfig,
    apply_block_table_only_if_safe,
    build_kivo_kv_sync_apply_decision,
)
from vllm.v1.worker.kivo_runtime_demotion_mark import (
    KivoRuntimeDemotionMarkSummary,
    current_kivo_runtime_demotion_mark_config,
    maybe_mark_demoted_blocks_after_block_table_apply,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch


_DEFAULT_ACTION = "off"
_SUPPORTED_POLICIES = {"recent_only", "countsketch_online"}


@dataclass(frozen=True)
class KivoRuntimeBlockTableApplyConfig:
    enabled: bool
    action: str
    policy: str
    keep_recent_blocks: int
    max_full_blocks: int
    require_slot_mapping_refresh: bool


@dataclass(frozen=True)
class KivoRuntimeBlockTableApplySummary:
    enabled: bool
    action: str
    attempted_row_count: int
    applied_row_count: int
    blocked_row_count: int
    blocker_reasons: dict[str, int]
    max_removed_blocks: int
    total_removed_blocks: int
    paired_plan_attempted_row_count: int
    paired_plan_safe_row_count: int
    paired_plan_blocked_row_count: int
    paired_plan_blocker_reasons: dict[str, int]
    runtime_demotion_mark_attempted_request_count: int
    runtime_demotion_mark_marked_request_count: int
    runtime_demotion_mark_blocked_request_count: int
    runtime_demotion_mark_marked_block_count: int
    runtime_demotion_mark_blocker_reasons: dict[str, int]


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


def current_kivo_runtime_block_table_apply_config(
    *,
    action_default: str = _DEFAULT_ACTION,
) -> KivoRuntimeBlockTableApplyConfig:
    enabled = _parse_bool_env("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE", default=False)
    action = os.getenv("KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION", action_default)
    if not enabled:
        action = "off"
    return KivoRuntimeBlockTableApplyConfig(
        enabled=enabled,
        action=action,
        policy=os.getenv(
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY", "recent_only"
        ),
        keep_recent_blocks=_parse_int_env(
            "KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS", default=4, minimum=0
        ),
        max_full_blocks=_parse_int_env(
            "KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS", default=64, minimum=1
        ),
        require_slot_mapping_refresh=_parse_bool_env(
            "KIVO_KV_RUNTIME_BLOCK_TABLE_REQUIRE_SLOT_MAPPING_REFRESH",
            default=True,
        ),
    )


def build_runtime_block_table_apply_summary(
    input_batch: "InputBatch",
    *,
    req_ids: Sequence[str] | None = None,
    kv_cache_gid: int = 0,
    slot_mapping_refresh_available: bool = False,
    kv_cache_manager: object | None = None,
    config: KivoRuntimeBlockTableApplyConfig | None = None,
) -> KivoRuntimeBlockTableApplySummary:
    if config is None:
        config = current_kivo_runtime_block_table_apply_config()

    if not config.enabled or config.action == "off":
        return KivoRuntimeBlockTableApplySummary(
            enabled=False,
            action="off",
            attempted_row_count=0,
            applied_row_count=0,
            blocked_row_count=0,
            blocker_reasons={"disabled": 1},
            max_removed_blocks=0,
            total_removed_blocks=0,
            paired_plan_attempted_row_count=0,
            paired_plan_safe_row_count=0,
            paired_plan_blocked_row_count=0,
            paired_plan_blocker_reasons={"disabled": 1},
            runtime_demotion_mark_attempted_request_count=0,
            runtime_demotion_mark_marked_request_count=0,
            runtime_demotion_mark_blocked_request_count=0,
            runtime_demotion_mark_marked_block_count=0,
            runtime_demotion_mark_blocker_reasons={"disabled": 1},
        )

    if config.policy not in _SUPPORTED_POLICIES:
        return KivoRuntimeBlockTableApplySummary(
            enabled=True,
            action=config.action,
            attempted_row_count=0,
            applied_row_count=0,
            blocked_row_count=0,
            blocker_reasons={"invalid_runtime_policy": 1},
            max_removed_blocks=0,
            total_removed_blocks=0,
            paired_plan_attempted_row_count=0,
            paired_plan_safe_row_count=0,
            paired_plan_blocked_row_count=0,
            paired_plan_blocker_reasons={"invalid_runtime_policy": 1},
            runtime_demotion_mark_attempted_request_count=0,
            runtime_demotion_mark_marked_request_count=0,
            runtime_demotion_mark_blocked_request_count=0,
            runtime_demotion_mark_marked_block_count=0,
            runtime_demotion_mark_blocker_reasons={"invalid_runtime_policy": 1},
        )

    target_req_ids = list(req_ids) if req_ids is not None else list(input_batch.req_ids)
    attempted = 0
    applied = 0
    blocked = 0
    blocker_reasons: dict[str, int] = {}
    max_removed = 0
    total_removed = 0
    paired_attempted = 0
    paired_safe = 0
    paired_blocked = 0
    paired_blocker_reasons: dict[str, int] = {}
    runtime_mark_attempted = 0
    runtime_mark_marked = 0
    runtime_mark_blocked = 0
    runtime_mark_blocks = 0
    runtime_mark_blocker_reasons: dict[str, int] = {}
    live_apply_config = current_kivo_live_ownership_apply_config()
    ownership_bridge_config = current_kivo_ownership_bridge_config()
    runtime_demotion_mark_config = current_kivo_runtime_demotion_mark_config()

    for req_id in target_req_ids:
        attempted += 1
        req_index = input_batch.get_req_index(req_id)
        if req_index is None:
            blocked += 1
            blocker_reasons["missing_request_row_mapping"] = (
                blocker_reasons.get("missing_request_row_mapping", 0) + 1
            )
            continue
        original_row = input_batch.get_req_block_row_ids(req_id, kv_cache_gid)
        if original_row is None:
            blocked += 1
            blocker_reasons["missing_original_row"] = (
                blocker_reasons.get("missing_original_row", 0) + 1
            )
            continue

        retention_decision = decide_kv_retention(
            original_row,
            get_block_scores(original_row),
            request_id=req_id,
            config=KivoKVRetentionConfig(
                enabled=True,
                policy=config.policy,
                keep_recent_blocks=config.keep_recent_blocks,
                max_full_blocks=config.max_full_blocks,
                min_blocks_before_action=0,
                action="plan_only",
            ),
        )
        sync_decision = build_kivo_kv_sync_apply_decision(
            req_id,
            original_row,
            retention_decision.keep_block_ids,
            retention_decision.candidate_drop_block_ids,
            protected_block_ids=retention_decision.protected_block_ids,
            slot_mapping_refresh_available=slot_mapping_refresh_available,
            config=KivoKVSyncApplyConfig(
                enabled=True,
                action=(
                    "plan_only"
                    if config.action != "apply_block_table_only"
                    else "apply_block_table_only"
                ),
                require_slot_mapping_refresh=config.require_slot_mapping_refresh,
            ),
        )
        block_table_applied = False
        removed_count = len(sync_decision.original_block_ids) - len(
            sync_decision.filtered_block_ids
        )
        max_removed = max(max_removed, removed_count)
        total_removed += removed_count
        if config.action == "apply_block_table_only" and sync_decision.safe_to_apply:
            if apply_block_table_only_if_safe(
                input_batch.block_table[kv_cache_gid], req_index, sync_decision
            ):
                applied += 1
                block_table_applied = True
            else:
                blocked += 1
                blocker_reasons["block_table_replace_failed"] = (
                    blocker_reasons.get("block_table_replace_failed", 0) + 1
                )
        elif config.action != "plan_only":
            blocked += 1
            for reason, count in sync_decision.blocker_reasons.items():
                blocker_reasons[reason] = blocker_reasons.get(reason, 0) + count

        if not live_apply_config.enabled:
            continue

        paired_attempted += 1
        live_plan = build_kivo_live_block_plan(
            original_row,
            retention_decision,
            request_id=req_id,
            shared_block_ids=(),
            block_table_sync_available=slot_mapping_refresh_available,
            ownership_mutation_available=False,
            config=KivoKVLiveBlockPlanConfig(
                enabled=True,
                action="plan_live_demotion_only",
                require_block_table_sync=live_apply_config.require_block_table_applied,
                protect_recent_blocks=live_apply_config.keep_recent_blocks,
                min_blocks_before_action=0,
            ),
        )
        live_decision = build_kivo_live_ownership_apply_decision(
            request_id=req_id,
            visible_before_block_ids=original_row,
            visible_after_block_ids=sync_decision.filtered_block_ids,
            candidate_demote_block_ids=live_plan.candidate_demote_block_ids,
            protected_block_ids=live_plan.protected_block_ids,
            block_table_applied=block_table_applied,
            slot_mapping_refresh_guaranteed=slot_mapping_refresh_available,
            ownership_mapping_available=False,
            config=KivoLiveOwnershipApplyConfig(
                enabled=True,
                action=live_apply_config.action,
                policy=live_apply_config.policy,
                keep_recent_blocks=live_apply_config.keep_recent_blocks,
                max_full_blocks=live_apply_config.max_full_blocks,
                require_block_table_applied=live_apply_config.require_block_table_applied,
                require_slot_mapping_refresh=(
                    live_apply_config.require_slot_mapping_refresh
                ),
            ),
        )
        if live_decision.safe_to_mutate_ownership:
            paired_safe += 1
        else:
            paired_blocked += 1
        for reason, count in live_plan.blocker_reasons.items():
            paired_blocker_reasons[reason] = (
                paired_blocker_reasons.get(reason, 0) + count
            )
        for reason, count in live_decision.blocker_reasons.items():
            paired_blocker_reasons[reason] = (
                paired_blocker_reasons.get(reason, 0) + count
            )
        if ownership_bridge_config.enabled:
            paired_blocker_reasons["worker_path_lacks_core_kv_manager_reference"] = (
                paired_blocker_reasons.get(
                    "worker_path_lacks_core_kv_manager_reference", 0
                )
                + 1
            )
        if runtime_demotion_mark_config.enabled:
            mark_summary: KivoRuntimeDemotionMarkSummary = (
                maybe_mark_demoted_blocks_after_block_table_apply(
                    request_id=req_id,
                    demote_block_ids=live_plan.candidate_demote_block_ids,
                    visible_after_block_ids=sync_decision.filtered_block_ids,
                    protected_block_ids=live_plan.protected_block_ids,
                    block_table_applied=block_table_applied,
                    slot_mapping_refresh_guaranteed=slot_mapping_refresh_available,
                    kv_cache_manager=kv_cache_manager,
                    config=runtime_demotion_mark_config,
                )
            )
            runtime_mark_attempted += mark_summary.attempted_request_count
            runtime_mark_marked += mark_summary.marked_request_count
            runtime_mark_blocked += mark_summary.blocked_request_count
            runtime_mark_blocks += mark_summary.marked_block_count
            for reason, count in mark_summary.blocker_reasons.items():
                runtime_mark_blocker_reasons[reason] = (
                    runtime_mark_blocker_reasons.get(reason, 0) + count
                )

    return KivoRuntimeBlockTableApplySummary(
        enabled=True,
        action=config.action,
        attempted_row_count=attempted,
        applied_row_count=applied,
        blocked_row_count=blocked,
        blocker_reasons=blocker_reasons,
        max_removed_blocks=max_removed,
        total_removed_blocks=total_removed,
        paired_plan_attempted_row_count=paired_attempted,
        paired_plan_safe_row_count=paired_safe,
        paired_plan_blocked_row_count=paired_blocked,
        paired_plan_blocker_reasons=paired_blocker_reasons,
        runtime_demotion_mark_attempted_request_count=runtime_mark_attempted,
        runtime_demotion_mark_marked_request_count=runtime_mark_marked,
        runtime_demotion_mark_blocked_request_count=runtime_mark_blocked,
        runtime_demotion_mark_marked_block_count=runtime_mark_blocks,
        runtime_demotion_mark_blocker_reasons=runtime_mark_blocker_reasons,
    )


def maybe_apply_runtime_block_table_before_slot_mapping(
    input_batch: "InputBatch",
    *,
    req_ids: Sequence[str] | None = None,
    kv_cache_gid: int = 0,
    kv_cache_manager: object | None = None,
    config: KivoRuntimeBlockTableApplyConfig | None = None,
) -> KivoRuntimeBlockTableApplySummary:
    """Apply filtered worker rows only at the pre-slot-mapping hook point."""
    return build_runtime_block_table_apply_summary(
        input_batch,
        req_ids=req_ids,
        kv_cache_gid=kv_cache_gid,
        slot_mapping_refresh_available=True,
        kv_cache_manager=kv_cache_manager,
        config=config,
    )
