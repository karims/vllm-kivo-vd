#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate Kivo-VD Phase 12 shadow event invariants."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "event_type",
    "version",
    "request_id",
    "layer_idx",
    "context_token_count",
    "block_size",
    "total_context_blocks",
    "candidate_budget_blocks",
    "selected_block_ids_by_score",
    "selected_block_ids_for_gather",
    "selected_block_count",
    "selected_ratio",
    "shadow_only",
    "active_routing",
    "measured_runtime_reduction",
)
RATIO_TOLERANCE = 1e-6


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Kivo-VD Phase 12 shadow event JSON or JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase12_shadow_event_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase12_shadow_event_validation.md",
    )
    return parser.parse_args(argv)


def load_events(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"shadow event input is missing: {input_path}")
    if input_path.suffix == ".jsonl":
        events = []
        for line_number, line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed JSONL row {line_number}: {input_path}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} must be an object"
                )
            events.append(value)
        return events

    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("events"), list):
        value = value["events"]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError("JSON input must be an event, event list, or events object")


def _integer_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0
            for item in value
        )
    )


def _record(
    *,
    event_index: int,
    check: str,
    message: str,
    severity: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    counts[f"{check}_{severity}"] += 1
    item = {
        "event_index": event_index,
        "check": check,
        "message": message,
    }
    if severity == "failed":
        errors.append(item)
    else:
        warnings.append(item)


def validate_event(
    event: dict[str, Any],
    event_index: int,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    missing = [field for field in REQUIRED_FIELDS if field not in event]
    if missing:
        _record(
            event_index=event_index,
            check="required_fields",
            message=f"missing required fields: {', '.join(missing)}",
            severity="failed",
            errors=errors,
            warnings=warnings,
            counts=counts,
        )
    else:
        counts["required_fields_passed"] += 1

    def require_false(field: str) -> None:
        if event.get(field) is False:
            counts[f"{field}_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check=field,
                message=f"{field} must be false",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )

    if event.get("shadow_only") is True:
        counts["shadow_only_passed"] += 1
    else:
        _record(
            event_index=event_index,
            check="shadow_only",
            message="shadow_only must be true",
            severity="failed",
            errors=errors,
            warnings=warnings,
            counts=counts,
        )
    require_false("active_routing")
    require_false("measured_runtime_reduction")

    score_ids = event.get("selected_block_ids_by_score")
    gather_ids = event.get("selected_block_ids_for_gather")
    valid_lists = _integer_list(score_ids) and _integer_list(gather_ids)
    if valid_lists:
        counts["selected_id_types_passed"] += 1
        if len(score_ids) == len(set(score_ids)) and len(gather_ids) == len(
            set(gather_ids)
        ):
            counts["duplicate_ids_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check="duplicate_ids",
                message="selected block ID lists must not contain duplicates",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
        if gather_ids == sorted(gather_ids):
            counts["gather_order_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check="gather_order",
                message="selected_block_ids_for_gather must be ascending",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
        if set(score_ids) == set(gather_ids):
            counts["selected_id_set_match_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check="selected_id_set_match",
                message="score-order and gather-order ID sets must match",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
    else:
        _record(
            event_index=event_index,
            check="selected_id_types",
            message="selected block IDs must be non-negative integer lists",
            severity="failed",
            errors=errors,
            warnings=warnings,
            counts=counts,
        )

    total_blocks = event.get("total_context_blocks")
    budget = event.get("candidate_budget_blocks")
    selected_count = event.get("selected_block_count")
    valid_counts = all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for value in (total_blocks, budget, selected_count)
    )
    if valid_counts:
        counts["count_types_passed"] += 1
        if budget <= total_blocks:
            counts["budget_bound_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check="budget_bound",
                message="candidate budget must not exceed total blocks",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
        if selected_count <= total_blocks:
            counts["selected_count_bound_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check="selected_count_bound",
                message="selected count must not exceed total blocks",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
        if valid_lists and selected_count == len(gather_ids):
            counts["selected_count_match_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check="selected_count_match",
                message="selected count must equal the gather ID list length",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
        if valid_lists and all(item < total_blocks for item in gather_ids):
            counts["selected_id_bounds_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check="selected_id_bounds",
                message="selected block IDs must be below total blocks",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
        ratio = event.get("selected_ratio")
        expected_ratio = (
            selected_count / total_blocks if total_blocks > 0 else 0.0
        )
        if isinstance(ratio, int | float) and not isinstance(ratio, bool):
            if abs(float(ratio) - expected_ratio) <= RATIO_TOLERANCE:
                counts["selected_ratio_passed"] += 1
            else:
                _record(
                    event_index=event_index,
                    check="selected_ratio",
                    message=(
                        f"selected ratio {ratio} differs from "
                        f"expected {expected_ratio}"
                    ),
                    severity="warned",
                    errors=errors,
                    warnings=warnings,
                    counts=counts,
                )
        else:
            _record(
                event_index=event_index,
                check="selected_ratio",
                message="selected_ratio must be numeric",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )
    else:
        _record(
            event_index=event_index,
            check="count_types",
            message="block counts and budget must be non-negative integers",
            severity="failed",
            errors=errors,
            warnings=warnings,
            counts=counts,
        )

    for field in ("ordering_valid", "causal_valid"):
        if field not in event or event[field] is True:
            counts[f"{field}_passed"] += 1
        else:
            _record(
                event_index=event_index,
                check=field,
                message=f"{field} must be true when present",
                severity="failed",
                errors=errors,
                warnings=warnings,
                counts=counts,
            )

    return {
        "event_index": event_index,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "per_check_counts": dict(counts),
    }


def validate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    results = [
        validate_event(event, event_index)
        for event_index, event in enumerate(events)
    ]
    errors = [item for result in results for item in result["errors"]]
    warnings = [item for result in results for item in result["warnings"]]
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(result["per_check_counts"])
    valid_events = sum(result["valid"] for result in results)
    return {
        "validation_passed": valid_events == len(events),
        "total_events": len(events),
        "valid_events": valid_events,
        "invalid_events": len(events) - valid_events,
        "warnings": warnings,
        "errors": errors,
        "per_check_counts": dict(sorted(counts.items())),
        "caveats": {
            "shadow_only_validation": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "no_vllm_runtime_behavior_change": True,
        },
    }


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 12 Shadow Event Validation",
        "",
        "## Validation Status",
        "",
        f"- Passed: `{_format(report['validation_passed'])}`",
        f"- Total events: `{report['total_events']}`",
        f"- Valid events: `{report['valid_events']}`",
        f"- Invalid events: `{report['invalid_events']}`",
        "",
        "## Check Counts",
        "",
        "| check | count |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{check}` | `{count}` |"
        for check, count in report["per_check_counts"].items()
    )
    for title, key in (("Errors", "errors"), ("Warnings", "warnings")):
        lines.extend(["", f"## {title}", ""])
        lines.extend(
            (
                f"- Event `{item['event_index']}` / `{item['check']}`: "
                f"{item['message']}"
            )
            for item in report[key]
        )
        if not report[key]:
            lines.append("- none")
    lines.extend([
        "",
        "## Caveats",
        "",
        "- This validator checks saved shadow events only.",
        "- Shadow events must not enable active routing.",
        "- No vLLM runtime behavior change is implied.",
        "- No measured runtime memory reduction is claimed.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        report = validate_events(load_events(args.input))
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(json.dumps({
            "validation_passed": report["validation_passed"],
            "total_events": report["total_events"],
            "valid_events": report["valid_events"],
            "invalid_events": report["invalid_events"],
            "output_json": args.output_json,
            "output_md": args.output_md,
        }, separators=(",", ":")))
        return 0 if report["validation_passed"] else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
