# SPDX-License-Identifier: Apache-2.0

"""Lightweight gated counters for Kivo demotion transport observation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass, field


@dataclass
class KivoDemotionCounters:
    block_table_apply_attempted: int = 0
    block_table_apply_succeeded: int = 0
    block_table_apply_rejected: int = 0
    filtered_row_plan_attempted: int = 0
    filtered_row_plan_succeeded: int = 0
    filtered_row_plan_rejected: int = 0
    filtered_row_apply_noop: int = 0
    filtered_row_changed_count: int = 0
    filtered_row_candidate_drop_count: int = 0
    demotion_command_export_path_entered: int = 0
    demotion_command_export_skipped_no_apply_summary: int = 0
    demotion_command_export_skipped_apply_not_successful: int = 0
    demotion_command_export_skipped_no_request_id: int = 0
    demotion_command_export_skipped_no_visible_before: int = 0
    demotion_command_export_skipped_no_visible_after: int = 0
    demotion_command_export_skipped_no_candidate_demote_ids: int = 0
    demotion_command_export_skipped_empty_after_filter: int = 0
    demotion_command_export_attempted: int = 0
    demotion_command_export_rejected: int = 0
    demotion_command_export_succeeded: int = 0
    demotion_command_dedupe_input_blocks: int = 0
    demotion_command_dedupe_dropped_blocks: int = 0
    demotion_command_dedupe_output_blocks: int = 0
    demotion_command_dedupe_empty_after_drop: int = 0
    sketch_build_attempted: int = 0
    sketch_build_succeeded: int = 0
    sketch_build_failed: int = 0
    sketched_blocks_total: int = 0
    sketch_missing_prevented_demotion: int = 0
    sketch_missing_prevented_free: int = 0
    freed_after_sketch_blocks_total: int = 0
    sketch_bytes_total: int = 0
    decode_only_requested: int = 0
    decode_only_supported: int = 0
    demotion_skipped_not_decode_phase: int = 0
    worker_envelopes_built: int = 0
    worker_envelopes_attached: int = 0
    scheduler_envelopes_received: int = 0
    core_transport_batches_received: int = 0
    core_commands_attempted: int = 0
    core_commands_accepted: int = 0
    core_commands_rejected: int = 0
    manager_mark_demoted_attempted: int = 0
    manager_mark_demoted_succeeded: int = 0
    manager_mark_demoted_rejected: int = 0
    ownership_remove_attempted: int = 0
    ownership_remove_succeeded: int = 0
    ownership_remove_rejected: int = 0
    ownership_remove_invariant_checked: int = 0
    ownership_remove_invariant_failed: int = 0
    ownership_prefilter_input_blocks: int = 0
    ownership_prefilter_dropped_already_removed: int = 0
    ownership_prefilter_output_blocks: int = 0
    ownership_removed_subset_of_marked: int = 0
    ownership_removed_subset_of_owned: int = 0
    ownership_removed_absent_after: int = 0
    ownership_remaining_nonempty: int = 0
    ownership_removed_reintroduced: int = 0
    ownership_demoted_bookkeeping_cleared: int = 0
    demoted_blocks_marked: int = 0
    ownership_removed_blocks: int = 0
    ownership_removed_blocks_total: int = 0
    req_to_blocks_removed: int = 0
    free_to_pool_attempted: int = 0
    free_to_pool_succeeded: int = 0
    free_to_pool_rejected: int = 0
    free_to_pool_blocks: int = 0
    free_to_pool_double_free_prevented: int = 0
    free_to_pool_calls: int = 0
    free_prefilter_input_blocks: int = 0
    free_prefilter_dropped_already_freed: int = 0
    free_prefilter_output_blocks: int = 0
    demotion_export_wall_time_seconds: float = 0.0
    demotion_export_max_call_wall_time_seconds: float = 0.0
    ownership_remove_wall_time_seconds: float = 0.0
    ownership_remove_max_call_wall_time_seconds: float = 0.0
    free_to_pool_wall_time_seconds: float = 0.0
    free_to_pool_max_call_wall_time_seconds: float = 0.0
    audit_bookkeeping_wall_time_seconds: float = 0.0
    audit_bookkeeping_max_call_wall_time_seconds: float = 0.0
    block_pool_free_capacity_before: int = 0
    block_pool_free_capacity_after: int = 0
    block_pool_free_capacity_delta: int = 0
    block_pool_num_free_blocks_before: int = 0
    block_pool_num_free_blocks_after: int = 0
    block_pool_num_free_blocks_delta: int = 0
    block_pool_free_accounting_observed: int = 0
    block_pool_free_accounting_increased: int = 0
    block_pool_free_accounting_rejected: int = 0
    block_pool_free_accounting_blocker_reasons: dict[str, int] = field(
        default_factory=dict
    )
    padding_zero_ambiguous: int = 0
    last_worker_row_raw_count: int = 0
    last_worker_row_nonzero_count: int = 0
    last_worker_row_unique_count: int = 0
    last_worker_row_trailing_zero_count: int = 0
    last_filtered_keep_count: int = 0
    last_filtered_drop_count: int = 0
    last_filtered_drop_ids_sample: tuple[int, ...] = ()
    last_filtered_keep_ids_sample: tuple[int, ...] = ()
    last_removed_block_ids_sample: tuple[int, ...] = ()
    last_remaining_block_ids_sample: tuple[int, ...] = ()
    last_freed_block_ids_sample: tuple[int, ...] = ()
    last_free_rejected_block_ids_sample: tuple[int, ...] = ()
    last_owned_before_remove_count: int = 0
    last_owned_after_remove_count: int = 0
    last_marked_demoted_before_remove_count: int = 0
    last_removed_after_absent: bool | None = None
    last_visible_before_count: int = 0
    last_visible_after_count: int = 0
    last_candidate_demote_count: int = 0
    ownership_remaining_blocks_last: int = 0
    last_filtered_row_changed: bool | None = None
    last_keep_recent_blocks: int = 0
    last_policy: str | None = None
    sketch_backend: str | None = None
    blocker_reasons: dict[str, int] = field(default_factory=dict)


_COUNTERS = KivoDemotionCounters()


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def kivo_demotion_counters_enabled() -> bool:
    return _parse_bool_env("KIVO_KV_DEMOTION_COUNTERS_ENABLE", default=False)


def get_kivo_demotion_counters_export_file() -> str | None:
    path = os.getenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE")
    if path is None:
        return None
    path = path.strip()
    return path or None


def reset_kivo_demotion_counters() -> None:
    global _COUNTERS
    _COUNTERS = KivoDemotionCounters()


def get_kivo_demotion_counters_snapshot() -> dict[str, object]:
    return asdict(_COUNTERS)


def increment_kivo_demotion_counter(name: str, amount: int = 1) -> None:
    if not kivo_demotion_counters_enabled() or amount <= 0:
        return
    current = getattr(_COUNTERS, name, None)
    if not isinstance(current, int):
        return
    setattr(_COUNTERS, name, current + amount)


def add_kivo_demotion_timing_counter(name: str, elapsed_seconds: float) -> None:
    if not kivo_demotion_counters_enabled() or elapsed_seconds <= 0:
        return
    current = getattr(_COUNTERS, name, None)
    if not isinstance(current, float):
        return
    setattr(_COUNTERS, name, current + elapsed_seconds)
    max_name = name.replace("_wall_time_seconds", "_max_call_wall_time_seconds")
    max_current = getattr(_COUNTERS, max_name, None)
    if isinstance(max_current, float) and elapsed_seconds > max_current:
        setattr(_COUNTERS, max_name, elapsed_seconds)


def add_kivo_demotion_blocker_reasons(blocker_reasons: dict[str, int]) -> None:
    if not kivo_demotion_counters_enabled():
        return
    for reason, count in blocker_reasons.items():
        if count <= 0:
            continue
        _COUNTERS.blocker_reasons[reason] = (
            _COUNTERS.blocker_reasons.get(reason, 0) + count
        )


def add_kivo_block_pool_accounting_blocker_reasons(
    blocker_reasons: dict[str, int],
) -> None:
    if not kivo_demotion_counters_enabled():
        return
    for reason, count in blocker_reasons.items():
        if count <= 0:
            continue
        _COUNTERS.block_pool_free_accounting_blocker_reasons[reason] = (
            _COUNTERS.block_pool_free_accounting_blocker_reasons.get(reason, 0)
            + count
        )


def set_kivo_demotion_counter_fields(**kwargs: object) -> None:
    if not kivo_demotion_counters_enabled():
        return
    for name, value in kwargs.items():
        if not hasattr(_COUNTERS, name):
            continue
        setattr(_COUNTERS, name, value)


def export_kivo_demotion_counters_snapshot_if_enabled(
    *,
    source: str = "unknown",
) -> None:
    if not kivo_demotion_counters_enabled():
        return
    export_file = get_kivo_demotion_counters_export_file()
    if export_file is None:
        return
    target = Path(export_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "source": source,
        "counters": get_kivo_demotion_counters_snapshot(),
    }
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(target)
