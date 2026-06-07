#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Analyze exported Kivo-VD runtime dry-run observer events."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Kivo-VD dry-run event JSONL exports."
    )
    parser.add_argument(
        "--input", default="outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl"
    )
    parser.add_argument(
        "--output", default="outputs/kivo_vd/vllm_kivo_dry_run_summary.json"
    )
    return parser.parse_args()


def _read_events(path: Path) -> tuple[list[dict[str, Any]], list[str], int]:
    if not path.exists():
        return [], [f"event file is missing: {path}"], 0

    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    malformed_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_rows += 1
                warnings.append(f"malformed JSONL row {line_no}: {exc}")
                continue
            if not isinstance(row, dict):
                malformed_rows += 1
                warnings.append(f"malformed JSONL row {line_no}: not an object")
                continue
            events.append(row)
    return events, warnings, malformed_rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _unique_sorted(values: set[Any]) -> list[Any]:
    return sorted(values, key=lambda value: str(value))


def analyze_events(input_path: str | Path) -> dict[str, Any]:
    events, warnings, malformed_rows = _read_events(Path(input_path))
    event_counts = Counter(str(event.get("event_type", "unknown")) for event in events)
    routing_events = [
        event
        for event in events
        if event.get("event_type") == "dry_run_routing_decision"
    ]

    selected_counts = [
        float(event.get("selected_block_count", 0)) for event in routing_events
    ]
    recent_counts = [
        float(event.get("recent_block_count", 0)) for event in routing_events
    ]
    skipped_counts = [
        float(event.get("skipped_block_count", 0)) for event in routing_events
    ]

    allocation_or_free_types = {
        "before_allocate_slots",
        "after_allocate_slots",
        "free_request",
    }
    if not routing_events:
        warnings.append("no dry_run_routing_decision events found")
    if (
        events
        and not routing_events
        and set(event_counts).issubset(allocation_or_free_types)
    ):
        warnings.append("only allocation/free events found")
    if routing_events and all(count == 0 for count in selected_counts):
        warnings.append("selected block count is always zero")

    request_ids = {
        event["request_id"] for event in events if event.get("request_id") is not None
    }
    sources = {event["source"] for event in events if event.get("source") is not None}
    candidate_budgets = {
        event["candidate_budget_blocks"]
        for event in routing_events
        if event.get("candidate_budget_blocks") is not None
    }
    recent_windows = {
        event["recent_window_blocks"]
        for event in routing_events
        if event.get("recent_window_blocks") is not None
    }
    full_id_events = [
        event
        for event in routing_events
        if event.get("full_block_ids_exported") is True
        or event.get("selected_block_ids_full") is not None
    ]

    return {
        "input": str(input_path),
        "total_events": len(events),
        "malformed_rows": malformed_rows,
        "event_counts": dict(sorted(event_counts.items())),
        "num_dry_run_routing_decision_events": len(routing_events),
        "avg_selected_block_count": _mean(selected_counts),
        "avg_recent_block_count": _mean(recent_counts),
        "avg_skipped_block_count": _mean(skipped_counts),
        "candidate_budget_blocks": _unique_sorted(candidate_budgets),
        "recent_window_blocks": _unique_sorted(recent_windows),
        "full_block_ids_exported_count": len(full_id_events),
        "preview_only_routing_event_count": (
            len(routing_events) - len(full_id_events)
        ),
        "all_routing_events_have_full_block_ids": (
            bool(routing_events) and len(full_id_events) == len(routing_events)
        ),
        "selected_block_preview": [
            event.get("selected_block_preview")
            for event in routing_events[:5]
            if event.get("selected_block_preview") is not None
        ],
        "recent_block_preview": [
            event.get("recent_block_preview")
            for event in routing_events[:5]
            if event.get("recent_block_preview") is not None
        ],
        "skipped_block_preview": [
            event.get("skipped_block_preview")
            for event in routing_events[:5]
            if event.get("skipped_block_preview") is not None
        ],
        "request_ids_seen": _unique_sorted(request_ids),
        "sources_seen": _unique_sorted(sources),
        "warnings": warnings,
    }


def main() -> int:
    args = _parse_args()
    summary = analyze_events(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
