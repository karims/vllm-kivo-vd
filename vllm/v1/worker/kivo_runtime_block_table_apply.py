# SPDX-License-Identifier: Apache-2.0

"""Gated runtime-facing block-table-only apply helpers for Kivo-VD."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from vllm.v1.core.kivo_demotion_command import KivoDemotionCommand
from vllm.v1.core.kivo_demotion_counters import (
    add_kivo_demotion_blocker_reasons,
    export_kivo_demotion_counters_snapshot_if_enabled,
    increment_kivo_demotion_counter,
    set_kivo_demotion_counter_fields,
)
from vllm.v1.core.kivo_demotion_transport import (
    KivoDemotionTransportEnvelope,
    current_kivo_demotion_transport_config,
)
from vllm.v1.core.kivo_live_ownership_apply import (
    KivoLiveOwnershipApplyConfig,
    build_kivo_live_ownership_apply_decision,
    current_kivo_live_ownership_apply_config,
)
from vllm.v1.core.kivo_kv_block_score_store import get_block_scores
from vllm.v1.core.kivo_kv_retention_policy import (
    KivoKVRetentionDecision,
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
_COUNTER_SAMPLE_LIMIT = 8


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
    demotion_transport_exported_count: int
    demotion_transport_blocked_count: int
    demotion_transport_blocker_reasons: dict[str, int]
    demotion_transport_envelopes: tuple[KivoDemotionTransportEnvelope, ...]


@dataclass(frozen=True)
class KivoRuntimeDemotionCommandExport:
    attempted: bool
    command: KivoDemotionCommand | None
    blocker_reason: str | None
    blocker_reasons: dict[str, int]
    visible_before_block_ids: tuple[int, ...]
    visible_after_block_ids: tuple[int, ...]
    candidate_demote_block_ids: tuple[int, ...]


@dataclass(frozen=True)
class KivoRuntimeFilteredRowPlan:
    visible_before_block_ids: tuple[int, ...]
    visible_after_block_ids: tuple[int, ...]
    candidate_demote_block_ids: tuple[int, ...]
    protected_block_ids: tuple[int, ...]
    filtered_row_changed: bool
    noop_reason: str | None
    blocker_reasons: dict[str, int]


def build_kivo_demotion_command_for_runtime_row(
    *,
    request_id: str | None,
    visible_before_block_ids: Sequence[int],
    visible_after_block_ids: Sequence[int],
    candidate_demote_block_ids: Sequence[int],
    protected_block_ids: Sequence[int] = (),
    block_table_applied: bool,
    slot_mapping_refresh_guaranteed: bool,
) -> KivoRuntimeDemotionCommandExport:
    """Build a worker-side demotion command payload when local invariants hold."""
    blocker_reasons: dict[str, int] = {}
    increment_kivo_demotion_counter("demotion_command_export_attempted")
    before = tuple(int(block_id) for block_id in visible_before_block_ids)
    after = tuple(int(block_id) for block_id in visible_after_block_ids)
    demote = tuple(int(block_id) for block_id in candidate_demote_block_ids)
    protected = tuple(int(block_id) for block_id in protected_block_ids)

    if request_id is None:
        blocker_reasons["missing_request_id"] = 1
    if not block_table_applied:
        blocker_reasons["block_table_not_applied"] = 1
    if not slot_mapping_refresh_guaranteed:
        blocker_reasons["slot_mapping_refresh_not_guaranteed"] = 1
    if not demote:
        blocker_reasons["empty_candidate_demote_ids"] = 1
    if not after:
        blocker_reasons["empty_visible_after_blocks"] = 1

    before_set = set(before)
    after_set = set(after)
    protected_set = set(protected)
    if any(block_id not in before_set for block_id in demote):
        blocker_reasons["candidate_demote_not_visible_before"] = sum(
            1 for block_id in demote if block_id not in before_set
        )
    if any(block_id in after_set for block_id in demote):
        blocker_reasons["candidate_demote_still_visible_after"] = sum(
            1 for block_id in demote if block_id in after_set
        )
    if any(block_id not in before_set for block_id in after):
        blocker_reasons["visible_after_not_subset_of_visible_before"] = sum(
            1 for block_id in after if block_id not in before_set
        )
    if any(block_id in protected_set for block_id in demote):
        blocker_reasons["candidate_demote_overlaps_protected"] = sum(
            1 for block_id in demote if block_id in protected_set
        )

    if blocker_reasons or request_id is None:
        increment_kivo_demotion_counter("demotion_command_export_rejected")
        export_kivo_demotion_counters_snapshot_if_enabled(
            source="worker_demotion_command_export_rejected"
        )
        return KivoRuntimeDemotionCommandExport(
            attempted=True,
            command=None,
            blocker_reason=next(iter(blocker_reasons), None),
            blocker_reasons=blocker_reasons,
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )

    increment_kivo_demotion_counter("worker_envelopes_built")
    increment_kivo_demotion_counter("demotion_command_export_succeeded")
    export_kivo_demotion_counters_snapshot_if_enabled(
        source="worker_demotion_command_export_built"
    )
    return KivoRuntimeDemotionCommandExport(
        attempted=True,
        command=KivoDemotionCommand(
            request_id=request_id,
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
            protected_block_ids=protected,
            block_table_applied=block_table_applied,
            slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
        ),
        blocker_reason=None,
        blocker_reasons={},
        visible_before_block_ids=before,
        visible_after_block_ids=after,
        candidate_demote_block_ids=demote,
    )


def maybe_build_kivo_demotion_command_after_runtime_apply(
    *,
    request_id: str | None,
    visible_before_block_ids: Sequence[int] | None,
    visible_after_block_ids: Sequence[int] | None,
    candidate_demote_block_ids: Sequence[int] | None,
    protected_block_ids: Sequence[int] = (),
    apply_summary_present: bool,
    block_table_applied: bool,
    slot_mapping_refresh_guaranteed: bool,
    filtered_row_changed: bool,
    keep_recent_blocks: int,
    policy: str,
) -> KivoRuntimeDemotionCommandExport:
    increment_kivo_demotion_counter("demotion_command_export_path_entered")
    before = tuple(int(block_id) for block_id in (visible_before_block_ids or ()))
    after = tuple(int(block_id) for block_id in (visible_after_block_ids or ()))
    demote = tuple(int(block_id) for block_id in (candidate_demote_block_ids or ()))
    set_kivo_demotion_counter_fields(
        last_visible_before_count=len(before),
        last_visible_after_count=len(after),
        last_candidate_demote_count=len(demote),
        last_filtered_row_changed=filtered_row_changed,
        last_keep_recent_blocks=keep_recent_blocks,
        last_policy=policy,
    )

    if not apply_summary_present:
        increment_kivo_demotion_counter(
            "demotion_command_export_skipped_no_apply_summary"
        )
        return KivoRuntimeDemotionCommandExport(
            attempted=False,
            command=None,
            blocker_reason="no_apply_summary",
            blocker_reasons={"no_apply_summary": 1},
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )
    if not block_table_applied:
        increment_kivo_demotion_counter(
            "demotion_command_export_skipped_apply_not_successful"
        )
        return KivoRuntimeDemotionCommandExport(
            attempted=False,
            command=None,
            blocker_reason="apply_not_successful",
            blocker_reasons={"apply_not_successful": 1},
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )
    if request_id is None:
        increment_kivo_demotion_counter("demotion_command_export_skipped_no_request_id")
        return KivoRuntimeDemotionCommandExport(
            attempted=False,
            command=None,
            blocker_reason="missing_request_id",
            blocker_reasons={"missing_request_id": 1},
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )
    if not before:
        increment_kivo_demotion_counter(
            "demotion_command_export_skipped_no_visible_before"
        )
        return KivoRuntimeDemotionCommandExport(
            attempted=False,
            command=None,
            blocker_reason="missing_visible_before_blocks",
            blocker_reasons={"missing_visible_before_blocks": 1},
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )
    if not after:
        counter_name = "demotion_command_export_skipped_empty_after_filter"
        blocker_reason = "empty_after_filter"
        if not filtered_row_changed:
            counter_name = "demotion_command_export_skipped_no_visible_after"
            blocker_reason = "missing_visible_after_blocks"
        increment_kivo_demotion_counter(counter_name)
        return KivoRuntimeDemotionCommandExport(
            attempted=False,
            command=None,
            blocker_reason=blocker_reason,
            blocker_reasons={blocker_reason: 1},
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )
    if not demote:
        increment_kivo_demotion_counter(
            "demotion_command_export_skipped_no_candidate_demote_ids"
        )
        return KivoRuntimeDemotionCommandExport(
            attempted=False,
            command=None,
            blocker_reason="empty_candidate_demote_ids",
            blocker_reasons={"empty_candidate_demote_ids": 1},
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )

    result = build_kivo_demotion_command_for_runtime_row(
        request_id=request_id,
        visible_before_block_ids=before,
        visible_after_block_ids=after,
        candidate_demote_block_ids=demote,
        protected_block_ids=protected_block_ids,
        block_table_applied=block_table_applied,
        slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
    )
    if result.command is None:
        add_kivo_demotion_blocker_reasons(result.blocker_reasons)
    return result


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


def _sample_block_ids(block_ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(block_id) for block_id in block_ids[:_COUNTER_SAMPLE_LIMIT])


def _plan_runtime_filtered_row(
    *,
    original_row: Sequence[int],
    policy: str,
    keep_recent_blocks: int,
    max_full_blocks: int,
) -> KivoRuntimeFilteredRowPlan:
    increment_kivo_demotion_counter("filtered_row_plan_attempted")
    row = tuple(int(block_id) for block_id in original_row)
    nonzero_row = tuple(block_id for block_id in row if block_id != 0)
    zero_count = len(row) - len(nonzero_row)
    set_kivo_demotion_counter_fields(
        last_worker_row_raw_count=len(row),
        last_worker_row_nonzero_count=len(nonzero_row),
        last_worker_row_unique_count=len(set(nonzero_row)),
        last_worker_row_trailing_zero_count=zero_count,
    )

    if zero_count > 0:
        increment_kivo_demotion_counter("filtered_row_plan_rejected")
        increment_kivo_demotion_counter("padding_zero_ambiguous", zero_count)
        return KivoRuntimeFilteredRowPlan(
            visible_before_block_ids=row,
            visible_after_block_ids=row,
            candidate_demote_block_ids=(),
            protected_block_ids=(),
            filtered_row_changed=False,
            noop_reason="filtered_row_noop_padding_ambiguity",
            blocker_reasons={"padding_zero_ambiguous": zero_count},
        )

    if policy == "recent_only":
        keep_recent = min(max(0, keep_recent_blocks), len(row))
        protected_recent = tuple(row[-keep_recent:]) if keep_recent > 0 else ()
        visible_after = protected_recent
        candidate_drop = tuple(row[:-keep_recent]) if keep_recent > 0 else row
        if len(row) <= keep_recent:
            visible_after = row
            candidate_drop = ()
            noop_reason = "filtered_row_noop_no_blocks_above_budget"
            increment_kivo_demotion_counter("filtered_row_apply_noop")
        elif not visible_after:
            noop_reason = "filtered_row_noop_no_blocks_above_budget"
        else:
            noop_reason = None
        changed = tuple(row) != tuple(visible_after)
        if changed:
            increment_kivo_demotion_counter("filtered_row_changed_count")
            increment_kivo_demotion_counter(
                "filtered_row_candidate_drop_count", len(candidate_drop)
            )
        elif noop_reason is None:
            noop_reason = "filtered_row_noop_all_blocks_protected"
            increment_kivo_demotion_counter("filtered_row_apply_noop")
        increment_kivo_demotion_counter("filtered_row_plan_succeeded")
        set_kivo_demotion_counter_fields(
            last_filtered_keep_count=len(visible_after),
            last_filtered_drop_count=len(candidate_drop),
            last_filtered_drop_ids_sample=_sample_block_ids(candidate_drop),
            last_filtered_keep_ids_sample=_sample_block_ids(visible_after),
        )
        return KivoRuntimeFilteredRowPlan(
            visible_before_block_ids=row,
            visible_after_block_ids=tuple(visible_after),
            candidate_demote_block_ids=tuple(candidate_drop),
            protected_block_ids=tuple(protected_recent),
            filtered_row_changed=changed,
            noop_reason=noop_reason,
            blocker_reasons=(
                {noop_reason: 1}
                if noop_reason is not None and not changed
                else {}
            ),
        )

    retention_decision = decide_kv_retention(
        row,
        get_block_scores(row),
        config=KivoKVRetentionConfig(
            enabled=True,
            policy=policy,
            keep_recent_blocks=keep_recent_blocks,
            max_full_blocks=max_full_blocks,
            min_blocks_before_action=0,
            action="plan_only",
        ),
    )
    visible_after = tuple(retention_decision.keep_block_ids)
    candidate_drop = tuple(retention_decision.candidate_drop_block_ids)
    changed = tuple(row) != visible_after
    if changed:
        increment_kivo_demotion_counter("filtered_row_changed_count")
        increment_kivo_demotion_counter(
            "filtered_row_candidate_drop_count", len(candidate_drop)
        )
    else:
        increment_kivo_demotion_counter("filtered_row_apply_noop")
    increment_kivo_demotion_counter("filtered_row_plan_succeeded")
    set_kivo_demotion_counter_fields(
        last_filtered_keep_count=len(visible_after),
        last_filtered_drop_count=len(candidate_drop),
        last_filtered_drop_ids_sample=_sample_block_ids(candidate_drop),
        last_filtered_keep_ids_sample=_sample_block_ids(visible_after),
    )
    noop_reason = None if changed else "filtered_row_noop_no_blocks_above_budget"
    return KivoRuntimeFilteredRowPlan(
        visible_before_block_ids=row,
        visible_after_block_ids=visible_after,
        candidate_demote_block_ids=candidate_drop,
        protected_block_ids=tuple(retention_decision.protected_block_ids),
        filtered_row_changed=changed,
        noop_reason=noop_reason,
        blocker_reasons=(
            {noop_reason: 1}
            if noop_reason is not None
            else {}
        ),
    )


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
            demotion_transport_exported_count=0,
            demotion_transport_blocked_count=0,
            demotion_transport_blocker_reasons={"disabled": 1},
            demotion_transport_envelopes=(),
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
            demotion_transport_exported_count=0,
            demotion_transport_blocked_count=0,
            demotion_transport_blocker_reasons={"invalid_runtime_policy": 1},
            demotion_transport_envelopes=(),
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
    transport_exported = 0
    transport_blocked = 0
    transport_blocker_reasons: dict[str, int] = {}
    transport_envelopes: list[KivoDemotionTransportEnvelope] = []
    live_apply_config = current_kivo_live_ownership_apply_config()
    ownership_bridge_config = current_kivo_ownership_bridge_config()
    runtime_demotion_mark_config = current_kivo_runtime_demotion_mark_config()
    transport_config = current_kivo_demotion_transport_config()

    for req_id in target_req_ids:
        attempted += 1
        increment_kivo_demotion_counter("block_table_apply_attempted")
        req_index = input_batch.get_req_index(req_id)
        if req_index is None:
            blocked += 1
            increment_kivo_demotion_counter("block_table_apply_rejected")
            blocker_reasons["missing_request_row_mapping"] = (
                blocker_reasons.get("missing_request_row_mapping", 0) + 1
            )
            continue
        original_row = input_batch.get_req_block_row_ids(req_id, kv_cache_gid)
        if original_row is None:
            blocked += 1
            increment_kivo_demotion_counter("block_table_apply_rejected")
            blocker_reasons["missing_original_row"] = (
                blocker_reasons.get("missing_original_row", 0) + 1
            )
            continue

        filtered_row_plan = _plan_runtime_filtered_row(
            original_row=original_row,
            policy=config.policy,
            keep_recent_blocks=config.keep_recent_blocks,
            max_full_blocks=config.max_full_blocks,
        )
        sync_decision = build_kivo_kv_sync_apply_decision(
            req_id,
            original_row,
            filtered_row_plan.visible_after_block_ids,
            filtered_row_plan.candidate_demote_block_ids,
            protected_block_ids=filtered_row_plan.protected_block_ids,
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
                increment_kivo_demotion_counter("block_table_apply_succeeded")
            else:
                blocked += 1
                increment_kivo_demotion_counter("block_table_apply_rejected")
                blocker_reasons["block_table_replace_failed"] = (
                    blocker_reasons.get("block_table_replace_failed", 0) + 1
                )
        elif config.action != "plan_only":
            blocked += 1
            increment_kivo_demotion_counter("block_table_apply_rejected")
            for reason, count in sync_decision.blocker_reasons.items():
                blocker_reasons[reason] = blocker_reasons.get(reason, 0) + count

        live_plan = build_kivo_live_block_plan(
            original_row,
            KivoKVRetentionDecision(
                enabled=True,
                policy=config.policy,
                action="plan_only",
                request_id=req_id,
                all_block_ids=filtered_row_plan.visible_before_block_ids,
                keep_block_ids=filtered_row_plan.visible_after_block_ids,
                candidate_drop_block_ids=(
                    filtered_row_plan.candidate_demote_block_ids
                ),
                protected_block_ids=filtered_row_plan.protected_block_ids,
                reason_counts=dict(filtered_row_plan.blocker_reasons),
                score_available_count=0,
                score_missing_count=0,
                would_reduce_full_blocks_by=len(
                    filtered_row_plan.candidate_demote_block_ids
                ),
            ),
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
        if transport_config.enabled and transport_config.action != "off":
            command_export = maybe_build_kivo_demotion_command_after_runtime_apply(
                request_id=req_id,
                visible_before_block_ids=original_row,
                visible_after_block_ids=sync_decision.filtered_block_ids,
                candidate_demote_block_ids=live_plan.candidate_demote_block_ids,
                protected_block_ids=live_plan.protected_block_ids,
                apply_summary_present=True,
                block_table_applied=block_table_applied,
                slot_mapping_refresh_guaranteed=slot_mapping_refresh_available,
                filtered_row_changed=filtered_row_plan.filtered_row_changed,
                keep_recent_blocks=config.keep_recent_blocks,
                policy=config.policy,
            )
            if command_export.command is None:
                transport_blocked += 1
                for reason, count in command_export.blocker_reasons.items():
                    transport_blocker_reasons[reason] = (
                        transport_blocker_reasons.get(reason, 0) + count
                    )
            else:
                transport_exported += 1
                transport_envelopes.append(
                    KivoDemotionTransportEnvelope(
                        request_id=req_id,
                        command=command_export.command,
                        source=command_export.command.source,
                    )
                )
                export_kivo_demotion_counters_snapshot_if_enabled(
                    source="worker_transport_envelope_ready"
                )

        if not live_apply_config.enabled:
            continue

        paired_attempted += 1
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
        demotion_transport_exported_count=transport_exported,
        demotion_transport_blocked_count=transport_blocked,
        demotion_transport_blocker_reasons=transport_blocker_reasons,
        demotion_transport_envelopes=tuple(transport_envelopes),
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
