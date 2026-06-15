# SPDX-License-Identifier: Apache-2.0

"""Gated runtime-facing block-table-only apply helpers for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from vllm.v1.core.kivo_kv_block_score_store import get_block_scores
from vllm.v1.core.kivo_kv_retention_policy import (
    KivoKVRetentionConfig,
    decide_kv_retention,
)
from vllm.v1.worker.kivo_kv_sync_apply import (
    KivoKVSyncApplyConfig,
    apply_block_table_only_if_safe,
    build_kivo_kv_sync_apply_decision,
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
    )


def build_runtime_block_table_apply_summary(
    input_batch: "InputBatch",
    *,
    req_ids: Sequence[str] | None = None,
    kv_cache_gid: int = 0,
    slot_mapping_refresh_available: bool = False,
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
        )

    target_req_ids = list(req_ids) if req_ids is not None else list(input_batch.req_ids)
    summary = KivoRuntimeBlockTableApplySummary(
        enabled=True,
        action=config.action,
        attempted_row_count=0,
        applied_row_count=0,
        blocked_row_count=0,
        blocker_reasons={},
        max_removed_blocks=0,
        total_removed_blocks=0,
    )

    attempted = 0
    applied = 0
    blocked = 0
    blocker_reasons: dict[str, int] = {}
    max_removed = 0
    total_removed = 0

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
                require_slot_mapping_refresh=True,
            ),
        )
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
                continue
        blocked += 1
        for reason, count in sync_decision.blocker_reasons.items():
            blocker_reasons[reason] = blocker_reasons.get(reason, 0) + count

    return KivoRuntimeBlockTableApplySummary(
        enabled=True,
        action=config.action,
        attempted_row_count=attempted,
        applied_row_count=applied,
        blocked_row_count=blocked,
        blocker_reasons=blocker_reasons,
        max_removed_blocks=max_removed,
        total_removed_blocks=total_removed,
    )
