# SPDX-License-Identifier: Apache-2.0

"""Process-local aggregate counters for low-overhead Kivo source modes.

The source-level attention observers can emit verbose JSONL records or, in
low-overhead mode, skip JSONL serialization and only update in-memory counters.
This module keeps the counter path small, process-local, and fail-closed.
"""

from __future__ import annotations

import copy
import os
import threading
from typing import Any

S3_2B_SCHEMA = "kivo_source_s3_2b_active_recent_window_attention_metadata_v1"
S3_3C_PLAN_SCHEMA = "kivo_source_s3_3c_active_sketch_plan_v1"
S3_3C_METADATA_SCHEMA = "kivo_source_s3_3c_active_sketch_metadata_alias_v1"

_COUNTER_LOCK = threading.Lock()
_COUNTERS_BY_SCHEMA: dict[str, dict[str, Any]] = {}


def current_record_mode() -> str:
    """Return the current source recording mode.

    Supported values:
    - ``events``: verbose JSONL emission
    - ``counters``: in-memory aggregate counters only
    - ``off``: no recording
    """

    return os.getenv("KIVO_SOURCE_RECORD_MODE", "events").strip().lower()


def is_counters_mode() -> bool:
    return current_record_mode() == "counters"


def _new_summary() -> dict[str, Any]:
    return {
        "event_count": 0,
        "sketch_computed_count": 0,
        "sketch_blocked_count": 0,
        "mutation_attempted_count": 0,
        "mutation_applied_count": 0,
        "active_routing_count": 0,
        "runtime_behavior_changed_count": 0,
        "blocker_count": 0,
        "blocker_reason_counts": {},
        "max_candidate_block_count": 0,
        "max_selected_block_count": 0,
        "max_excluded_block_count": 0,
        "max_unselected_block_count": 0,
        "max_aliased_block_count": 0,
        "max_visible_block_count": 0,
        "max_original_visible_block_count": 0,
        "max_theoretical_attention_visible_block_reduction_ratio": 0.0,
        "sketch_plan_used_count": 0,
        "metadata_alias_count": 0,
        "recent_window_event_count": 0,
        "min_seq_len": None,
        "max_seq_len": None,
        "last_seq_len": None,
    }


def _summary_for_schema(schema_version: str) -> dict[str, Any]:
    summary = _COUNTERS_BY_SCHEMA.get(schema_version)
    if summary is None:
        summary = _new_summary()
        _COUNTERS_BY_SCHEMA[schema_version] = summary
    return summary


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bump_max(summary: dict[str, Any], field: str, value: Any) -> None:
    numeric = _int_value(value)
    if numeric is None:
        return
    summary[field] = max(int(summary.get(field, 0) or 0), numeric)


def _bump_max_float(summary: dict[str, Any], field: str, value: Any) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    try:
        summary[field] = max(float(summary.get(field, 0.0) or 0.0), numeric)
    except (TypeError, ValueError):
        summary[field] = numeric


def _bump_min_max_last(summary: dict[str, Any], value: Any) -> None:
    numeric = _int_value(value)
    if numeric is None:
        return
    current_min = summary.get("min_seq_len")
    summary["min_seq_len"] = (
        numeric if current_min is None else min(int(current_min), numeric)
    )
    summary["max_seq_len"] = (
        numeric
        if summary.get("max_seq_len") is None
        else max(int(summary["max_seq_len"]), numeric)
    )
    summary["last_seq_len"] = numeric


def _bump_blocker(summary: dict[str, Any], blocker_reason: Any) -> None:
    if not blocker_reason:
        return
    summary["blocker_count"] += 1
    reason_counts = summary.setdefault("blocker_reason_counts", {})
    key = str(blocker_reason)
    reason_counts[key] = int(reason_counts.get(key, 0) or 0) + 1


def _record_flags(summary: dict[str, Any], record: dict[str, Any]) -> None:
    if record.get("mutation_attempted") is True:
        summary["mutation_attempted_count"] += 1
    if record.get("mutation_applied") is True:
        summary["mutation_applied_count"] += 1
    if record.get("active_routing") is True:
        summary["active_routing_count"] += 1
    if record.get("runtime_behavior_changed") is True:
        summary["runtime_behavior_changed_count"] += 1


def record_counter_event(record: dict[str, Any]) -> None:
    """Record a compact counter event in counters-only mode.

    The helper is intentionally a no-op unless ``KIVO_SOURCE_RECORD_MODE`` is
    set to ``counters``.
    """

    if not is_counters_mode():
        return

    schema_version = str(record.get("schema_version") or "unknown")
    with _COUNTER_LOCK:
        summary = _summary_for_schema(schema_version)
        summary["event_count"] += 1

        blocker_reason = record.get("mutation_blocker_reason") or record.get(
            "sketch_plan_blocker_reason"
        )
        _bump_blocker(summary, blocker_reason)

        if schema_version == S3_2B_SCHEMA:
            summary["recent_window_event_count"] += 1
            _record_flags(summary, record)
            if record.get("visible_block_count") is not None:
                _bump_max(summary, "max_visible_block_count", record["visible_block_count"])
            if record.get("visible_block_count_estimate") is not None:
                _bump_max(
                    summary,
                    "max_visible_block_count",
                    record["visible_block_count_estimate"],
                )
            if record.get("original_visible_block_count") is not None:
                _bump_max(
                    summary,
                    "max_original_visible_block_count",
                    record["original_visible_block_count"],
                )
            if record.get("selected_block_count") is not None:
                _bump_max(summary, "max_selected_block_count", record["selected_block_count"])
            if record.get("excluded_block_count") is not None:
                _bump_max(summary, "max_excluded_block_count", record["excluded_block_count"])
                _bump_max(summary, "max_unselected_block_count", record["excluded_block_count"])
            if record.get("theoretical_attention_visible_block_reduction_ratio") is not None:
                _bump_max_float(
                    summary,
                    "max_theoretical_attention_visible_block_reduction_ratio",
                    record["theoretical_attention_visible_block_reduction_ratio"],
                )
            if record.get("max_seq_len") is not None:
                _bump_min_max_last(summary, record["max_seq_len"])
        elif schema_version == S3_3C_PLAN_SCHEMA:
            if record.get("sketch_computed") is True:
                summary["sketch_computed_count"] += 1
            else:
                summary["sketch_blocked_count"] += 1
            if record.get("candidate_block_count") is not None:
                _bump_max(summary, "max_candidate_block_count", record["candidate_block_count"])
            if record.get("selected_block_count") is not None:
                _bump_max(summary, "max_selected_block_count", record["selected_block_count"])
            if record.get("excluded_block_count") is not None:
                _bump_max(summary, "max_excluded_block_count", record["excluded_block_count"])
                _bump_max(summary, "max_unselected_block_count", record["excluded_block_count"])
            if record.get("max_seq_len") is not None:
                _bump_min_max_last(summary, record["max_seq_len"])
        elif schema_version == S3_3C_METADATA_SCHEMA:
            summary["metadata_alias_count"] += 1
            if record.get("sketch_plan_used") is True:
                summary["sketch_plan_used_count"] += 1
            _record_flags(summary, record)
            if record.get("visible_block_count_estimate") is not None:
                _bump_max(
                    summary,
                    "max_visible_block_count",
                    record["visible_block_count_estimate"],
                )
            if record.get("selected_block_count") is not None:
                _bump_max(summary, "max_selected_block_count", record["selected_block_count"])
            if record.get("excluded_block_count") is not None:
                _bump_max(summary, "max_excluded_block_count", record["excluded_block_count"])
                _bump_max(summary, "max_unselected_block_count", record["excluded_block_count"])
            if record.get("aliased_block_count") is not None:
                _bump_max(summary, "max_aliased_block_count", record["aliased_block_count"])
            if record.get("visible_block_count") is not None:
                _bump_max(summary, "max_visible_block_count", record["visible_block_count"])
            if record.get("original_visible_block_count") is not None:
                _bump_max(
                    summary,
                    "max_original_visible_block_count",
                    record["original_visible_block_count"],
                )
            if record.get("theoretical_attention_visible_block_reduction_ratio") is not None:
                _bump_max_float(
                    summary,
                    "max_theoretical_attention_visible_block_reduction_ratio",
                    record["theoretical_attention_visible_block_reduction_ratio"],
                )
            if record.get("max_seq_len") is not None:
                _bump_min_max_last(summary, record["max_seq_len"])
        else:
            _record_flags(summary, record)


def snapshot_counters() -> dict[str, Any]:
    with _COUNTER_LOCK:
        return copy.deepcopy(_COUNTERS_BY_SCHEMA)


def get_and_reset_counters() -> dict[str, Any]:
    with _COUNTER_LOCK:
        snapshot = copy.deepcopy(_COUNTERS_BY_SCHEMA)
        _COUNTERS_BY_SCHEMA.clear()
        return snapshot


def flatten_counters(counters_by_schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten schema-keyed counters into one combined summary."""

    combined = _new_summary()
    for summary in counters_by_schema.values():
        combined["event_count"] += int(summary.get("event_count", 0) or 0)
        combined["sketch_computed_count"] += int(
            summary.get("sketch_computed_count", 0) or 0
        )
        combined["sketch_blocked_count"] += int(
            summary.get("sketch_blocked_count", 0) or 0
        )
        combined["mutation_attempted_count"] += int(
            summary.get("mutation_attempted_count", 0) or 0
        )
        combined["mutation_applied_count"] += int(
            summary.get("mutation_applied_count", 0) or 0
        )
        combined["active_routing_count"] += int(
            summary.get("active_routing_count", 0) or 0
        )
        combined["runtime_behavior_changed_count"] += int(
            summary.get("runtime_behavior_changed_count", 0) or 0
        )
        combined["blocker_count"] += int(summary.get("blocker_count", 0) or 0)
        combined["sketch_plan_used_count"] += int(
            summary.get("sketch_plan_used_count", 0) or 0
        )
        combined["metadata_alias_count"] += int(
            summary.get("metadata_alias_count", 0) or 0
        )
        combined["recent_window_event_count"] += int(
            summary.get("recent_window_event_count", 0) or 0
        )
        for field in [
            "max_candidate_block_count",
            "max_selected_block_count",
            "max_excluded_block_count",
            "max_unselected_block_count",
            "max_aliased_block_count",
            "max_visible_block_count",
            "max_original_visible_block_count",
        ]:
            combined[field] = max(
                int(combined.get(field, 0) or 0),
                int(summary.get(field, 0) or 0),
            )
        try:
            combined["max_theoretical_attention_visible_block_reduction_ratio"] = max(
                float(
                    combined["max_theoretical_attention_visible_block_reduction_ratio"]
                ),
                float(
                    summary.get(
                        "max_theoretical_attention_visible_block_reduction_ratio",
                        0.0,
                    )
                    or 0.0
                ),
            )
        except (TypeError, ValueError):
            pass
        for key, value in (summary.get("blocker_reason_counts") or {}).items():
            reason_counts = combined.setdefault("blocker_reason_counts", {})
            reason_counts[str(key)] = int(reason_counts.get(str(key), 0)) + int(value)
        if summary.get("min_seq_len") is not None:
            current_min = combined.get("min_seq_len")
            combined["min_seq_len"] = (
                int(summary["min_seq_len"])
                if current_min is None
                else min(int(current_min), int(summary["min_seq_len"]))
            )
        if summary.get("max_seq_len") is not None:
            combined["max_seq_len"] = (
                int(summary["max_seq_len"])
                if combined.get("max_seq_len") is None
                else max(int(combined["max_seq_len"]), int(summary["max_seq_len"]))
            )
        if summary.get("last_seq_len") is not None:
            combined["last_seq_len"] = int(summary["last_seq_len"])
    return combined
