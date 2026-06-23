# SPDX-License-Identifier: Apache-2.0

"""Gated runtime-facing block-table-only apply helpers for Kivo-VD."""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from vllm.v1.core.kivo_demotion_command import KivoDemotionCommand
from vllm.v1.core.kivo_demotion_counters import (
    add_kivo_demotion_blocker_reasons,
    add_kivo_demotion_timing_counter,
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
from vllm.v1.worker.kivo_kv_sketch_runtime import (
    KivoKVSketchRuntime,
    extract_kv_block_tensor,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch


_DEFAULT_ACTION = "off"
_SUPPORTED_POLICIES = {
    "recent_only",
    "prefix_recent",
    "countsketch_online",
    "sketch_topk",
    "sketch_span_topk",
}
_COUNTER_SAMPLE_LIMIT = 8
_EXPORTED_DEMOTION_BLOCK_IDS_BY_REQUEST: dict[str, set[int]] = {}


def reset_kivo_demotion_command_dedupe_state_for_tests() -> None:
    _EXPORTED_DEMOTION_BLOCK_IDS_BY_REQUEST.clear()


@dataclass(frozen=True)
class KivoRuntimeBlockTableApplyConfig:
    enabled: bool
    action: str
    policy: str
    keep_recent_blocks: int
    max_full_blocks: int
    require_slot_mapping_refresh: bool
    sketch_topk_blocks: int = 0
    sketch_span_radius: int = 0
    keep_prefix_blocks: int = 0


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
    sketch_build_attempted: int
    sketch_build_succeeded: int
    sketch_build_failed: int
    sketched_blocks_total: int
    sketch_missing_prevented_demotion: int
    sketch_backend: str | None
    sketch_bytes_total: int


@dataclass(frozen=True)
class KivoRuntimeDemotionCommandExport:
    attempted: bool
    command: KivoDemotionCommand | None
    blocker_reason: str | None
    blocker_reasons: dict[str, int]
    visible_before_block_ids: tuple[int, ...]
    visible_after_block_ids: tuple[int, ...]
    candidate_demote_block_ids: tuple[int, ...]
    sketch_build_attempted: int = 0
    sketch_build_succeeded: int = 0
    sketch_build_failed: int = 0
    sketched_blocks_total: int = 0
    sketch_missing_prevented_demotion: int = 0
    sketch_backend: str | None = None
    sketch_bytes_total: int = 0


@dataclass(frozen=True)
class KivoRuntimeSketchGateResult:
    candidate_demote_block_ids: tuple[int, ...]
    blocker_reasons: dict[str, int]
    sketch_build_attempted: int
    sketch_build_succeeded: int
    sketch_build_failed: int
    sketched_blocks_total: int
    sketch_missing_prevented_demotion: int
    sketch_backend: str | None
    sketch_bytes_total: int


@dataclass(frozen=True)
class KivoRuntimeFilteredRowPlan:
    visible_before_block_ids: tuple[int, ...]
    visible_after_block_ids: tuple[int, ...]
    candidate_demote_block_ids: tuple[int, ...]
    protected_block_ids: tuple[int, ...]
    recent_keep_block_ids: tuple[int, ...]
    sketch_topk_keep_block_ids: tuple[int, ...]
    sketch_span_anchor_block_ids: tuple[int, ...]
    sketch_span_keep_block_ids: tuple[int, ...]
    old_block_score_pairs: tuple[tuple[int, float], ...]
    missing_score_block_ids: tuple[int, ...]
    retention_ratio_numerator: int
    retention_ratio_denominator: int
    contiguous_span_count: int
    max_gap_between_kept_blocks: int
    filtered_row_changed: bool
    noop_reason: str | None
    blocker_reasons: dict[str, int]


def _trace_retained_blocks_enabled() -> bool:
    return _parse_bool_env(
        "KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_RETAINED_BLOCKS", default=False
    )


def _trace_retained_blocks_max() -> int:
    return _parse_int_env(
        "KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_MAX_BLOCKS", default=64, minimum=1
    )


def _trace_retained_blocks_file() -> str | None:
    path = os.getenv("KIVO_KV_RUNTIME_BLOCK_TABLE_TRACE_FILE")
    if path is None:
        return None
    path = path.strip()
    return path or None


def _truncate_block_ids(block_ids: Sequence[int], *, limit: int) -> tuple[int, ...]:
    return tuple(int(block_id) for block_id in block_ids[:limit])


def _truncate_score_pairs(
    score_pairs: Sequence[tuple[int, float]], *, limit: int
) -> tuple[tuple[int, float], ...]:
    return tuple((int(block_id), float(score)) for block_id, score in score_pairs[:limit])


def _retention_shape_metrics(
    row: Sequence[int],
    kept_block_ids: Sequence[int],
) -> tuple[int, int]:
    if not kept_block_ids:
        return 0, 0
    keep_set = set(int(block_id) for block_id in kept_block_ids)
    kept_indices = [
        idx for idx, block_id in enumerate(row) if int(block_id) in keep_set
    ]
    if not kept_indices:
        return 0, 0
    contiguous_span_count = 1
    max_gap = 0
    for previous, current in zip(kept_indices, kept_indices[1:]):
        gap = max(0, current - previous - 1)
        if gap > 0:
            contiguous_span_count += 1
        if gap > max_gap:
            max_gap = gap
    return contiguous_span_count, max_gap


def _write_retention_trace(
    *,
    request_id: str | None,
    policy: str,
    plan: KivoRuntimeFilteredRowPlan,
) -> None:
    if not _trace_retained_blocks_enabled():
        return
    trace_file = _trace_retained_blocks_file()
    if not trace_file:
        return
    limit = _trace_retained_blocks_max()
    payload = {
        "request_id": request_id,
        "policy": policy,
        "visible_before_count": len(plan.visible_before_block_ids),
        "visible_before_block_ids": list(
            _truncate_block_ids(plan.visible_before_block_ids, limit=limit)
        ),
        "visible_after_count": len(plan.visible_after_block_ids),
        "visible_after_block_ids": list(
            _truncate_block_ids(plan.visible_after_block_ids, limit=limit)
        ),
        "recent_keep_count": len(plan.recent_keep_block_ids),
        "recent_keep_block_ids": list(
            _truncate_block_ids(plan.recent_keep_block_ids, limit=limit)
        ),
        "sketch_topk_keep_count": len(plan.sketch_topk_keep_block_ids),
        "sketch_topk_keep_block_ids": list(
            _truncate_block_ids(plan.sketch_topk_keep_block_ids, limit=limit)
        ),
        "sketch_span_anchor_count": len(plan.sketch_span_anchor_block_ids),
        "sketch_span_anchor_block_ids": list(
            _truncate_block_ids(plan.sketch_span_anchor_block_ids, limit=limit)
        ),
        "sketch_span_keep_count": len(plan.sketch_span_keep_block_ids),
        "sketch_span_keep_block_ids": list(
            _truncate_block_ids(plan.sketch_span_keep_block_ids, limit=limit)
        ),
        "candidate_demote_count": len(plan.candidate_demote_block_ids),
        "candidate_demote_block_ids": list(
            _truncate_block_ids(plan.candidate_demote_block_ids, limit=limit)
        ),
        "old_block_score_count": len(plan.old_block_score_pairs),
        "old_block_score_pairs": [
            {"block_id": block_id, "score": score}
            for block_id, score in _truncate_score_pairs(
                plan.old_block_score_pairs, limit=limit
            )
        ],
        "missing_score_count": len(plan.missing_score_block_ids),
        "missing_score_block_ids": list(
            _truncate_block_ids(plan.missing_score_block_ids, limit=limit)
        ),
        "final_block_order_count": len(plan.visible_after_block_ids),
        "final_block_order": list(
            _truncate_block_ids(plan.visible_after_block_ids, limit=limit)
        ),
        "retention_ratio": (
            float(plan.retention_ratio_numerator) / float(plan.retention_ratio_denominator)
            if plan.retention_ratio_denominator > 0
            else 0.0
        ),
        "contiguous_span_count": plan.contiguous_span_count,
        "max_gap_between_kept_blocks": plan.max_gap_between_kept_blocks,
        "filtered_row_changed": plan.filtered_row_changed,
        "noop_reason": plan.noop_reason,
    }
    target = Path(trace_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def build_kivo_demotion_command_for_runtime_row(
    *,
    request_id: str | None,
    visible_before_block_ids: Sequence[int],
    visible_after_block_ids: Sequence[int],
    candidate_demote_block_ids: Sequence[int],
    protected_block_ids: Sequence[int] = (),
    block_table_applied: bool,
    slot_mapping_refresh_guaranteed: bool,
    sketch_gated: bool = False,
    sketch_backend: str | None = None,
) -> KivoRuntimeDemotionCommandExport:
    """Build a worker-side demotion command payload when local invariants hold."""
    started_at = time.perf_counter()
    blocker_reasons: dict[str, int] = {}
    try:
        increment_kivo_demotion_counter("demotion_command_export_attempted")
        before = tuple(int(block_id) for block_id in visible_before_block_ids)
        after = tuple(int(block_id) for block_id in visible_after_block_ids)
        raw_demote = tuple(int(block_id) for block_id in candidate_demote_block_ids)
        protected = tuple(int(block_id) for block_id in protected_block_ids)

        increment_kivo_demotion_counter(
            "demotion_command_dedupe_input_blocks",
            len(raw_demote),
        )
        already_exported = (
            _EXPORTED_DEMOTION_BLOCK_IDS_BY_REQUEST.get(request_id, set())
            if request_id is not None
            else set()
        )
        demote = tuple(
            block_id for block_id in raw_demote if block_id not in already_exported
        )
        dedupe_dropped = len(raw_demote) - len(demote)
        increment_kivo_demotion_counter(
            "demotion_command_dedupe_dropped_blocks",
            dedupe_dropped,
        )
        increment_kivo_demotion_counter(
            "demotion_command_dedupe_output_blocks",
            len(demote),
        )

        if request_id is None:
            blocker_reasons["missing_request_id"] = 1
        if not block_table_applied:
            blocker_reasons["block_table_not_applied"] = 1
        if not slot_mapping_refresh_guaranteed:
            blocker_reasons["slot_mapping_refresh_not_guaranteed"] = 1
        if raw_demote and not demote:
            increment_kivo_demotion_counter(
                "demotion_command_dedupe_empty_after_drop"
            )
            blocker_reasons["dedupe_empty_after_drop"] = 1
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

        _EXPORTED_DEMOTION_BLOCK_IDS_BY_REQUEST.setdefault(request_id, set()).update(
            demote
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
                sketch_gated=sketch_gated,
                sketch_backend=sketch_backend,
                sketch_block_count=len(demote),
            ),
            blocker_reason=None,
            blocker_reasons={},
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
        )
    finally:
        add_kivo_demotion_timing_counter(
            "demotion_export_wall_time_seconds",
            time.perf_counter() - started_at,
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
    kv_sketch_runtime: KivoKVSketchRuntime | None = None,
    kv_cache_tensor: Any | None = None,
) -> KivoRuntimeDemotionCommandExport:
    increment_kivo_demotion_counter("demotion_command_export_path_entered")
    before = tuple(int(block_id) for block_id in (visible_before_block_ids or ()))
    after = tuple(int(block_id) for block_id in (visible_after_block_ids or ()))
    demote = tuple(int(block_id) for block_id in (candidate_demote_block_ids or ()))
    sketch_gate = _gate_candidate_demote_blocks_by_sketch(
        candidate_demote_block_ids=demote,
        kv_sketch_runtime=kv_sketch_runtime,
        kv_cache_tensor=kv_cache_tensor,
    )
    demote = sketch_gate.candidate_demote_block_ids
    set_kivo_demotion_counter_fields(
        last_visible_before_count=len(before),
        last_visible_after_count=len(after),
        last_candidate_demote_count=len(demote),
        last_filtered_row_changed=filtered_row_changed,
        last_keep_recent_blocks=keep_recent_blocks,
        last_policy=policy,
    )
    if _parse_bool_env("KIVO_KV_DEMOTION_DECODE_ONLY", default=False):
        increment_kivo_demotion_counter("decode_only_requested")
        # This worker hook does not currently receive a reliable decode-vs-prefill
        # phase signal. Record the request, then leave behavior unchanged.

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
            sketch_backend=sketch_gate.sketch_backend,
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
            sketch_backend=sketch_gate.sketch_backend,
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
            sketch_backend=sketch_gate.sketch_backend,
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
            sketch_backend=sketch_gate.sketch_backend,
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
            sketch_backend=sketch_gate.sketch_backend,
        )
    if not demote:
        increment_kivo_demotion_counter(
            "demotion_command_export_skipped_no_candidate_demote_ids"
        )
        blocker_reasons = {"empty_candidate_demote_ids": 1}
        if sketch_gate.blocker_reasons:
            blocker_reasons.update(sketch_gate.blocker_reasons)
        return KivoRuntimeDemotionCommandExport(
            attempted=False,
            command=None,
            blocker_reason="empty_candidate_demote_ids",
            blocker_reasons=blocker_reasons,
            visible_before_block_ids=before,
            visible_after_block_ids=after,
            candidate_demote_block_ids=demote,
            sketch_build_attempted=sketch_gate.sketch_build_attempted,
            sketch_build_succeeded=sketch_gate.sketch_build_succeeded,
            sketch_build_failed=sketch_gate.sketch_build_failed,
            sketched_blocks_total=sketch_gate.sketched_blocks_total,
            sketch_missing_prevented_demotion=(
                sketch_gate.sketch_missing_prevented_demotion
            ),
            sketch_backend=sketch_gate.sketch_backend,
            sketch_bytes_total=sketch_gate.sketch_bytes_total,
        )

    result = build_kivo_demotion_command_for_runtime_row(
        request_id=request_id,
        visible_before_block_ids=before,
        visible_after_block_ids=after,
        candidate_demote_block_ids=demote,
        protected_block_ids=protected_block_ids,
        block_table_applied=block_table_applied,
        slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
        sketch_gated=(
            kv_sketch_runtime is not None
            and kv_sketch_runtime.config.enabled
            and bool(demote)
        ),
        sketch_backend=sketch_gate.sketch_backend,
    )
    if result.command is None:
        combined = dict(result.blocker_reasons)
        for reason, count in sketch_gate.blocker_reasons.items():
            combined[reason] = combined.get(reason, 0) + count
        add_kivo_demotion_blocker_reasons(combined)
        return KivoRuntimeDemotionCommandExport(
            attempted=result.attempted,
            command=None,
            blocker_reason=result.blocker_reason,
            blocker_reasons=combined,
            visible_before_block_ids=result.visible_before_block_ids,
            visible_after_block_ids=result.visible_after_block_ids,
            candidate_demote_block_ids=result.candidate_demote_block_ids,
            sketch_build_attempted=sketch_gate.sketch_build_attempted,
            sketch_build_succeeded=sketch_gate.sketch_build_succeeded,
            sketch_build_failed=sketch_gate.sketch_build_failed,
            sketched_blocks_total=sketch_gate.sketched_blocks_total,
            sketch_missing_prevented_demotion=(
                sketch_gate.sketch_missing_prevented_demotion
            ),
            sketch_backend=sketch_gate.sketch_backend,
            sketch_bytes_total=sketch_gate.sketch_bytes_total,
        )
    return KivoRuntimeDemotionCommandExport(
        attempted=result.attempted,
        command=result.command,
        blocker_reason=result.blocker_reason,
        blocker_reasons=result.blocker_reasons,
        visible_before_block_ids=result.visible_before_block_ids,
        visible_after_block_ids=result.visible_after_block_ids,
        candidate_demote_block_ids=result.candidate_demote_block_ids,
        sketch_build_attempted=sketch_gate.sketch_build_attempted,
        sketch_build_succeeded=sketch_gate.sketch_build_succeeded,
        sketch_build_failed=sketch_gate.sketch_build_failed,
        sketched_blocks_total=sketch_gate.sketched_blocks_total,
        sketch_missing_prevented_demotion=(
            sketch_gate.sketch_missing_prevented_demotion
        ),
        sketch_backend=sketch_gate.sketch_backend,
        sketch_bytes_total=sketch_gate.sketch_bytes_total,
    )


def _gate_candidate_demote_blocks_by_sketch(
    *,
    candidate_demote_block_ids: Sequence[int],
    kv_sketch_runtime: KivoKVSketchRuntime | None,
    kv_cache_tensor: Any | None,
) -> KivoRuntimeSketchGateResult:
    candidate_ids = tuple(int(block_id) for block_id in candidate_demote_block_ids)
    if kv_sketch_runtime is None or not kv_sketch_runtime.config.enabled:
        return KivoRuntimeSketchGateResult(
            candidate_demote_block_ids=candidate_ids,
            blocker_reasons={},
            sketch_build_attempted=0,
            sketch_build_succeeded=0,
            sketch_build_failed=0,
            sketched_blocks_total=0,
            sketch_missing_prevented_demotion=0,
            sketch_backend=None,
            sketch_bytes_total=0,
        )

    set_kivo_demotion_counter_fields(
        sketch_backend=kv_sketch_runtime.backend.backend_name
    )
    blocker_reasons: dict[str, int] = {}
    kept_block_ids: list[int] = []
    attempted = 0
    succeeded = 0
    failed = 0
    sketch_bytes_total = 0

    if kv_cache_tensor is None:
        set_kivo_demotion_counter_fields(
            sketch_backend=kv_sketch_runtime.backend.backend_name
        )
        blocker_reasons["sketch_kv_cache_unavailable"] = len(candidate_ids) or 1
        increment_kivo_demotion_counter(
            "sketch_missing_prevented_demotion", len(candidate_ids)
        )
        return KivoRuntimeSketchGateResult(
            candidate_demote_block_ids=(),
            blocker_reasons=blocker_reasons,
            sketch_build_attempted=0,
            sketch_build_succeeded=0,
            sketch_build_failed=len(candidate_ids),
            sketched_blocks_total=0,
            sketch_missing_prevented_demotion=len(candidate_ids),
            sketch_backend=kv_sketch_runtime.backend.backend_name,
            sketch_bytes_total=0,
        )

    for block_id in candidate_ids:
        attempted += 1
        increment_kivo_demotion_counter("sketch_build_attempted")
        block_tensor, extract_error = extract_kv_block_tensor(kv_cache_tensor, block_id)
        if block_tensor is None:
            failed += 1
            increment_kivo_demotion_counter("sketch_build_failed")
            increment_kivo_demotion_counter("sketch_missing_prevented_demotion")
            reason = extract_error or "sketch_block_extraction_failed"
            blocker_reasons[reason] = blocker_reasons.get(reason, 0) + 1
            continue

        result = kv_sketch_runtime.build_and_store_block_sketch(
            block_id=block_id,
            block_tensor=block_tensor,
            kv_kind="kv",
        )
        if result.success:
            succeeded += 1
            sketch_bytes_total += int(result.sketch_bytes)
            kept_block_ids.append(block_id)
            increment_kivo_demotion_counter("sketch_build_succeeded")
            increment_kivo_demotion_counter("sketched_blocks_total")
            increment_kivo_demotion_counter(
                "sketch_bytes_total", int(result.sketch_bytes)
            )
        else:
            failed += 1
            increment_kivo_demotion_counter("sketch_build_failed")
            increment_kivo_demotion_counter("sketch_missing_prevented_demotion")
            reason = result.blocker_reason or "sketch_build_failed"
            blocker_reasons[reason] = blocker_reasons.get(reason, 0) + 1

    return KivoRuntimeSketchGateResult(
        candidate_demote_block_ids=tuple(kept_block_ids),
        blocker_reasons=blocker_reasons,
        sketch_build_attempted=attempted,
        sketch_build_succeeded=succeeded,
        sketch_build_failed=failed,
        sketched_blocks_total=succeeded,
        sketch_missing_prevented_demotion=len(candidate_ids) - len(kept_block_ids),
        sketch_backend=kv_sketch_runtime.backend.backend_name,
        sketch_bytes_total=sketch_bytes_total,
    )


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
    sketch_topk_blocks: int = 0,
    sketch_span_radius: int = 0,
    keep_prefix_blocks: int = 0,
    kv_sketch_runtime: KivoKVSketchRuntime | None = None,
    kv_cache_tensor: Any | None = None,
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
            recent_keep_block_ids=(),
            sketch_topk_keep_block_ids=(),
            sketch_span_anchor_block_ids=(),
            sketch_span_keep_block_ids=(),
            old_block_score_pairs=(),
            missing_score_block_ids=(),
            retention_ratio_numerator=len(row),
            retention_ratio_denominator=len(row),
            contiguous_span_count=1 if row else 0,
            max_gap_between_kept_blocks=0,
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
            recent_keep_block_ids=tuple(protected_recent),
            sketch_topk_keep_block_ids=(),
            sketch_span_anchor_block_ids=(),
            sketch_span_keep_block_ids=(),
            old_block_score_pairs=(),
            missing_score_block_ids=(),
            retention_ratio_numerator=len(visible_after),
            retention_ratio_denominator=len(row),
            contiguous_span_count=1 if visible_after else 0,
            max_gap_between_kept_blocks=0,
            filtered_row_changed=changed,
            noop_reason=noop_reason,
            blocker_reasons=(
                {noop_reason: 1}
                if noop_reason is not None and not changed
                else {}
            ),
        )

    if policy == "prefix_recent":
        keep_prefix = min(max(0, keep_prefix_blocks), len(row))
        keep_recent = min(max(0, keep_recent_blocks), len(row))
        prefix_ids = tuple(row[:keep_prefix]) if keep_prefix > 0 else ()
        recent_ids = tuple(row[-keep_recent:]) if keep_recent > 0 else ()
        total_budget = max(0, max_full_blocks)

        ordered_keep: list[int] = []
        for block_id in prefix_ids + recent_ids:
            if block_id in ordered_keep:
                continue
            ordered_keep.append(block_id)
        ordered_keep = ordered_keep[:total_budget]

        keep_set = set(ordered_keep)
        visible_after = tuple(block_id for block_id in row if block_id in keep_set)
        candidate_drop = tuple(
            block_id for block_id in row if block_id not in keep_set
        )
        prefix_kept = tuple(
            block_id for block_id in prefix_ids if block_id in keep_set
        )
        recent_kept = tuple(
            block_id for block_id in recent_ids if block_id in keep_set
        )
        changed = tuple(row) != visible_after
        contiguous_span_count, max_gap = _retention_shape_metrics(row, visible_after)
        increment_kivo_demotion_counter(
            "prefix_recent_prefix_blocks_kept", len(prefix_kept)
        )
        increment_kivo_demotion_counter(
            "prefix_recent_recent_blocks_kept", len(recent_kept)
        )
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
            last_prefix_recent_keep_ids_sample=_sample_block_ids(visible_after),
            last_retention_ratio_numerator=len(visible_after),
            last_retention_ratio_denominator=len(row),
            last_contiguous_span_count=contiguous_span_count,
            last_max_gap_between_kept_blocks=max_gap,
        )
        noop_reason = None if changed else "filtered_row_noop_no_blocks_above_budget"
        return KivoRuntimeFilteredRowPlan(
            visible_before_block_ids=row,
            visible_after_block_ids=visible_after,
            candidate_demote_block_ids=candidate_drop,
            protected_block_ids=tuple(visible_after),
            recent_keep_block_ids=recent_kept,
            sketch_topk_keep_block_ids=(),
            sketch_span_anchor_block_ids=(),
            sketch_span_keep_block_ids=(),
            old_block_score_pairs=(),
            missing_score_block_ids=(),
            retention_ratio_numerator=len(visible_after),
            retention_ratio_denominator=len(row),
            contiguous_span_count=contiguous_span_count,
            max_gap_between_kept_blocks=max_gap,
            filtered_row_changed=changed,
            noop_reason=noop_reason,
            blocker_reasons=({noop_reason: 1} if noop_reason is not None else {}),
        )

    if policy == "sketch_topk":
        keep_recent = min(max(0, keep_recent_blocks), len(row))
        protected_recent = tuple(row[-keep_recent:]) if keep_recent > 0 else ()
        older = tuple(row[:-keep_recent]) if keep_recent > 0 else row
        topk_budget = max(0, sketch_topk_blocks)
        total_budget = max(len(protected_recent), max_full_blocks)
        topk_budget = min(topk_budget, max(0, total_budget - len(protected_recent)))
        score_map = {}
        if kv_sketch_runtime is not None and kv_sketch_runtime.config.enabled:
            score_map = kv_sketch_runtime.ensure_scores_for_blocks(
                older,
                kv_cache_tensor=kv_cache_tensor,
                kv_kind="kv",
            )
        older_index = {block_id: idx for idx, block_id in enumerate(older)}
        scored_older = [
            (block_id, float(score_map[block_id]))
            for block_id in older
            if block_id in score_map
        ]
        scored_older.sort(key=lambda item: (-item[1], -older_index[item[0]]))
        selected_old = {
            block_id for block_id, _ in scored_older[:topk_budget]
        }
        keep_set = set(protected_recent) | selected_old
        visible_after = tuple(block_id for block_id in row if block_id in keep_set)
        candidate_drop = tuple(block_id for block_id in row if block_id not in keep_set)
        missing_score_count = len(older) - len(scored_older)
        changed = tuple(row) != visible_after
        increment_kivo_demotion_counter(
            "sketch_topk_old_blocks_considered", len(older)
        )
        increment_kivo_demotion_counter(
            "sketch_topk_extra_blocks_kept", len(selected_old)
        )
        increment_kivo_demotion_counter(
            "sketch_topk_missing_scores", missing_score_count
        )
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
            last_sketch_topk_keep_ids_sample=_sample_block_ids(
                tuple(block_id for block_id in row if block_id in selected_old)
            ),
        )
        noop_reason = None if changed else "filtered_row_noop_no_blocks_above_budget"
        return KivoRuntimeFilteredRowPlan(
            visible_before_block_ids=row,
            visible_after_block_ids=visible_after,
            candidate_demote_block_ids=candidate_drop,
            protected_block_ids=protected_recent,
            recent_keep_block_ids=protected_recent,
            sketch_topk_keep_block_ids=tuple(
                block_id for block_id in row if block_id in selected_old
            ),
            sketch_span_anchor_block_ids=(),
            sketch_span_keep_block_ids=(),
            old_block_score_pairs=tuple(
                (int(block_id), float(score)) for block_id, score in scored_older
            ),
            missing_score_block_ids=tuple(
                block_id for block_id in older if block_id not in score_map
            ),
            retention_ratio_numerator=len(visible_after),
            retention_ratio_denominator=len(row),
            contiguous_span_count=_retention_shape_metrics(row, visible_after)[0],
            max_gap_between_kept_blocks=_retention_shape_metrics(row, visible_after)[1],
            filtered_row_changed=changed,
            noop_reason=noop_reason,
            blocker_reasons=(
                {noop_reason: 1}
                if noop_reason is not None
                else {}
            ),
        )

    if policy == "sketch_span_topk":
        keep_recent = min(max(0, keep_recent_blocks), len(row))
        protected_recent = tuple(row[-keep_recent:]) if keep_recent > 0 else ()
        older = tuple(row[:-keep_recent]) if keep_recent > 0 else row
        topk_budget = max(0, sketch_topk_blocks)
        span_radius = max(0, sketch_span_radius)
        total_budget = max(len(protected_recent), max_full_blocks)
        available_old_budget = max(0, total_budget - len(protected_recent))
        topk_budget = min(topk_budget, available_old_budget)

        score_map = {}
        if kv_sketch_runtime is not None and kv_sketch_runtime.config.enabled:
            score_map = kv_sketch_runtime.ensure_scores_for_blocks(
                older,
                kv_cache_tensor=kv_cache_tensor,
                kv_kind="kv",
            )
        older_index = {block_id: idx for idx, block_id in enumerate(older)}
        scored_older = [
            (block_id, float(score_map[block_id]))
            for block_id in older
            if block_id in score_map
        ]
        scored_older.sort(key=lambda item: (-item[1], -older_index[item[0]]))
        anchor_ids = [block_id for block_id, _ in scored_older[:topk_budget]]
        anchor_set = set(anchor_ids)

        neighbor_candidates: list[tuple[int, int, int]] = []
        seen_neighbor_ids: set[int] = set()
        for anchor_rank, anchor_id in enumerate(anchor_ids):
            anchor_idx = older_index[anchor_id]
            for neighbor_idx in range(
                max(0, anchor_idx - span_radius),
                min(len(older), anchor_idx + span_radius + 1),
            ):
                neighbor_id = older[neighbor_idx]
                if neighbor_id in anchor_set or neighbor_id in seen_neighbor_ids:
                    continue
                seen_neighbor_ids.add(neighbor_id)
                distance = abs(neighbor_idx - anchor_idx)
                neighbor_candidates.append((distance, anchor_rank, neighbor_idx))
        neighbor_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        neighbor_ids = [older[idx] for _, _, idx in neighbor_candidates]

        selected_old_ordered: list[int] = []
        for block_id in anchor_ids + neighbor_ids:
            if block_id in selected_old_ordered:
                continue
            if len(selected_old_ordered) >= available_old_budget:
                break
            selected_old_ordered.append(block_id)

        selected_old = set(selected_old_ordered)
        selected_anchor_ids = tuple(
            block_id for block_id in anchor_ids if block_id in selected_old
        )
        selected_neighbor_ids = tuple(
            block_id for block_id in selected_old_ordered if block_id not in anchor_set
        )
        keep_set = set(protected_recent) | selected_old
        visible_after = tuple(block_id for block_id in row if block_id in keep_set)
        candidate_drop = tuple(block_id for block_id in row if block_id not in keep_set)
        missing_score_ids = tuple(
            block_id for block_id in older if block_id not in score_map
        )
        missing_score_count = len(missing_score_ids)
        changed = tuple(row) != visible_after
        contiguous_span_count, max_gap = _retention_shape_metrics(row, visible_after)
        increment_kivo_demotion_counter(
            "sketch_span_old_blocks_considered", len(older)
        )
        increment_kivo_demotion_counter(
            "sketch_span_anchor_blocks_kept", len(selected_anchor_ids)
        )
        increment_kivo_demotion_counter(
            "sketch_span_neighbor_blocks_kept", len(selected_neighbor_ids)
        )
        increment_kivo_demotion_counter(
            "sketch_span_missing_scores", missing_score_count
        )
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
            last_sketch_span_anchor_ids_sample=_sample_block_ids(selected_anchor_ids),
            last_sketch_span_keep_ids_sample=_sample_block_ids(
                tuple(block_id for block_id in row if block_id in selected_old)
            ),
            last_retention_ratio_numerator=len(visible_after),
            last_retention_ratio_denominator=len(row),
            last_contiguous_span_count=contiguous_span_count,
            last_max_gap_between_kept_blocks=max_gap,
        )
        noop_reason = None if changed else "filtered_row_noop_no_blocks_above_budget"
        return KivoRuntimeFilteredRowPlan(
            visible_before_block_ids=row,
            visible_after_block_ids=visible_after,
            candidate_demote_block_ids=candidate_drop,
            protected_block_ids=protected_recent,
            recent_keep_block_ids=protected_recent,
            sketch_topk_keep_block_ids=(),
            sketch_span_anchor_block_ids=selected_anchor_ids,
            sketch_span_keep_block_ids=tuple(
                block_id for block_id in row if block_id in selected_old
            ),
            old_block_score_pairs=tuple(
                (int(block_id), float(score)) for block_id, score in scored_older
            ),
            missing_score_block_ids=missing_score_ids,
            retention_ratio_numerator=len(visible_after),
            retention_ratio_denominator=len(row),
            contiguous_span_count=contiguous_span_count,
            max_gap_between_kept_blocks=max_gap,
            filtered_row_changed=changed,
            noop_reason=noop_reason,
            blocker_reasons=(
                {noop_reason: 1}
                if noop_reason is not None
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
        recent_keep_block_ids=(),
        sketch_topk_keep_block_ids=(),
        sketch_span_anchor_block_ids=(),
        sketch_span_keep_block_ids=(),
        old_block_score_pairs=(),
        missing_score_block_ids=(),
        retention_ratio_numerator=len(visible_after),
        retention_ratio_denominator=len(row),
        contiguous_span_count=_retention_shape_metrics(row, visible_after)[0],
        max_gap_between_kept_blocks=_retention_shape_metrics(row, visible_after)[1],
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
        sketch_topk_blocks=_parse_int_env(
            "KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_TOPK", default=0, minimum=0
        ),
        sketch_span_radius=_parse_int_env(
            "KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_SPAN_RADIUS",
            default=0,
            minimum=0,
        ),
        keep_prefix_blocks=_parse_int_env(
            "KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_PREFIX_BLOCKS",
            default=0,
            minimum=0,
        ),
    )


def build_runtime_block_table_apply_summary(
    input_batch: "InputBatch",
    *,
    req_ids: Sequence[str] | None = None,
    kv_cache_gid: int = 0,
    slot_mapping_refresh_available: bool = False,
    kv_cache_manager: object | None = None,
    kv_sketch_runtime: KivoKVSketchRuntime | None = None,
    kv_cache_tensor: Any | None = None,
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
            sketch_build_attempted=0,
            sketch_build_succeeded=0,
            sketch_build_failed=0,
            sketched_blocks_total=0,
            sketch_missing_prevented_demotion=0,
            sketch_backend=None,
            sketch_bytes_total=0,
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
            sketch_build_attempted=0,
            sketch_build_succeeded=0,
            sketch_build_failed=0,
            sketched_blocks_total=0,
            sketch_missing_prevented_demotion=0,
            sketch_backend=(
                kv_sketch_runtime.backend.backend_name
                if kv_sketch_runtime is not None
                else None
            ),
            sketch_bytes_total=0,
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
    sketch_build_attempted = 0
    sketch_build_succeeded = 0
    sketch_build_failed = 0
    sketched_blocks_total = 0
    sketch_missing_prevented_demotion = 0
    sketch_bytes_total = 0
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
            sketch_topk_blocks=config.sketch_topk_blocks,
            sketch_span_radius=config.sketch_span_radius,
            keep_prefix_blocks=config.keep_prefix_blocks,
            kv_sketch_runtime=kv_sketch_runtime,
            kv_cache_tensor=kv_cache_tensor,
        )
        _write_retention_trace(
            request_id=req_id,
            policy=config.policy,
            plan=filtered_row_plan,
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
                kv_sketch_runtime=kv_sketch_runtime,
                kv_cache_tensor=kv_cache_tensor,
            )
            sketch_build_attempted += command_export.sketch_build_attempted
            sketch_build_succeeded += command_export.sketch_build_succeeded
            sketch_build_failed += command_export.sketch_build_failed
            sketched_blocks_total += command_export.sketched_blocks_total
            sketch_missing_prevented_demotion += (
                command_export.sketch_missing_prevented_demotion
            )
            sketch_bytes_total += command_export.sketch_bytes_total
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
        sketch_build_attempted=sketch_build_attempted,
        sketch_build_succeeded=sketch_build_succeeded,
        sketch_build_failed=sketch_build_failed,
        sketched_blocks_total=sketched_blocks_total,
        sketch_missing_prevented_demotion=sketch_missing_prevented_demotion,
        sketch_backend=(
            kv_sketch_runtime.backend.backend_name
            if kv_sketch_runtime is not None
            and kv_sketch_runtime.config.enabled
            else None
        ),
        sketch_bytes_total=sketch_bytes_total,
    )


def maybe_apply_runtime_block_table_before_slot_mapping(
    input_batch: "InputBatch",
    *,
    req_ids: Sequence[str] | None = None,
    kv_cache_gid: int = 0,
    kv_cache_manager: object | None = None,
    kv_sketch_runtime: KivoKVSketchRuntime | None = None,
    kv_cache_tensor: Any | None = None,
    config: KivoRuntimeBlockTableApplyConfig | None = None,
) -> KivoRuntimeBlockTableApplySummary:
    """Apply filtered worker rows only at the pre-slot-mapping hook point."""
    return build_runtime_block_table_apply_summary(
        input_batch,
        req_ids=req_ids,
        kv_cache_gid=kv_cache_gid,
        slot_mapping_refresh_available=True,
        kv_cache_manager=kv_cache_manager,
        kv_sketch_runtime=kv_sketch_runtime,
        kv_cache_tensor=kv_cache_tensor,
        config=config,
    )
