#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check long-context generation evidence before Phase 11.6."""

import argparse
import json
from pathlib import Path
from typing import Any

AGGRESSIVE_SHORT_CONTEXT_MAP = "0:12,5:8,8:8,11:12"
REQUIRED_CAVEATS = (
    "outside_vllm",
    "no_vllm_integration",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Phase 11.5 long-context adaptive-generation evidence "
            "before a context-scaled Phase 11.6 experiment."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase11_5_long_context_readiness.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase11_5_long_context_readiness.md",
    )
    parser.add_argument("--min-exact-match-rate", type=float, default=1.0)
    parser.add_argument("--min-token-match-rate", type=float, default=1.0)
    parser.add_argument(
        "--max-normalized-edit-distance",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-average-kl", type=float, default=0.01)
    parser.add_argument(
        "--max-selected-ratio-for-tradeoff",
        type=float,
        default=0.60,
    )
    return parser.parse_args(argv)


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _normalize_map(value: Any) -> str:
    if isinstance(value, str):
        entries = []
        for item in value.split(","):
            if not item.strip() or ":" not in item:
                return value
            layer, budget = item.split(":", 1)
            entries.append((int(layer), int(budget)))
        return ",".join(
            f"{layer}:{budget}" for layer, budget in sorted(entries)
        )
    if isinstance(value, dict):
        try:
            entries = sorted(
                (int(layer), int(budget))
                for layer, budget in value.items()
            )
        except (TypeError, ValueError):
            return ""
        return ",".join(f"{layer}:{budget}" for layer, budget in entries)
    return ""


def _load_input(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Phase 11.5 result is missing: {input_path}")
    if input_path.suffix == ".jsonl":
        rows = []
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
                    f"JSONL row {line_number} must be an object: {input_path}"
                )
            rows.append(value)
        return rows, {}

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {input_path}")
    if isinstance(payload.get("policy_length_map_tokens"), list):
        rows = payload["policy_length_map_tokens"]
    elif isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif "policy" in payload:
        rows = [payload]
    else:
        raise ValueError(
            f"summary lacks policy_length_map_tokens rows: {input_path}"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"result rows must be objects: {input_path}")
    caveats = payload.get("caveats", {})
    return rows, caveats if isinstance(caveats, dict) else {}


def _actual_length(row: dict[str, Any]) -> float | None:
    for field in (
        "average_actual_prompt_token_length",
        "actual_prompt_token_length",
    ):
        value = _number(row.get(field))
        if value is not None:
            return value
    lengths = row.get("actual_prompt_token_lengths")
    if isinstance(lengths, list):
        numbers = [
            number
            for item in lengths
            if (number := _number(item)) is not None
        ]
        if numbers:
            return sum(numbers) / len(numbers)
    return None


def _input_rows(
    input_paths: list[str | Path],
) -> list[dict[str, Any]]:
    normalized = []
    for path in input_paths:
        rows, caveats = _load_input(path)
        caveats_available = bool(caveats)
        caveats_ok = (
            all(caveats.get(key) is True for key in REQUIRED_CAVEATS)
            and (
                caveats.get("measured_runtime_reduction") is False
                or caveats.get("no_measured_runtime_reduction") is True
            )
            if caveats_available
            else True
        )
        for row in rows:
            if row.get("status") not in (None, "succeeded"):
                continue
            selected_ratio = _number(
                row.get(
                    "average_selected_block_ratio_across_patched_layers"
                )
            )
            estimated_reduction = _number(
                row.get("estimated_active_block_reduction_ratio")
            )
            if estimated_reduction is None and selected_ratio is not None:
                estimated_reduction = 1.0 - selected_ratio
            normalized.append({
                "input_path": str(path),
                "policy": row.get("policy"),
                "layer_budget_map": _normalize_map(
                    row.get("layer_budget_map")
                ),
                "target_token_length": _number(
                    row.get("target_token_length")
                ),
                "actual_prompt_token_length": _actual_length(row),
                "max_new_tokens": row.get("max_new_tokens"),
                "exact_sequence_match_rate": _number(
                    row.get("exact_sequence_match_rate")
                ),
                "average_token_match_rate": _number(
                    row.get("average_token_match_rate")
                ),
                "average_normalized_edit_distance": _number(
                    row.get("average_normalized_edit_distance")
                ),
                "average_per_step_kl_divergence": _number(
                    row.get("average_per_step_kl_divergence")
                ),
                "average_per_step_top1_match_rate": _number(
                    row.get("average_per_step_top1_match_rate")
                ),
                "average_selected_block_ratio_across_patched_layers": (
                    selected_ratio
                ),
                "estimated_active_block_reduction_ratio": (
                    estimated_reduction
                ),
                "caveats_available": caveats_available,
                "required_caveats_present": caveats_ok,
            })
    return normalized


def _is_long_context(row: dict[str, Any]) -> bool:
    lengths = (
        row["target_token_length"],
        row["actual_prompt_token_length"],
    )
    return any(length is not None and length >= 768 for length in lengths)


def _passes(
    row: dict[str, Any],
    *,
    min_exact_match_rate: float,
    min_token_match_rate: float,
    max_normalized_edit_distance: float,
    max_average_kl: float,
) -> bool:
    required = (
        row["exact_sequence_match_rate"],
        row["average_token_match_rate"],
        row["average_normalized_edit_distance"],
        row["average_per_step_kl_divergence"],
    )
    if any(value is None for value in required):
        return False
    return all([
        row["policy"] == "query_key_block_score",
        _is_long_context(row),
        row["exact_sequence_match_rate"] >= min_exact_match_rate,
        row["average_token_match_rate"] >= min_token_match_rate,
        row["average_normalized_edit_distance"]
        <= max_normalized_edit_distance,
        row["average_per_step_kl_divergence"] <= max_average_kl,
        row["required_caveats_present"],
    ])


def _matching_oracle(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    keys = (
        "layer_budget_map",
        "target_token_length",
        "max_new_tokens",
    )
    return next(
        (
            candidate
            for candidate in rows
            if candidate["policy"] == "oracle_topk"
            and all(candidate[key] == row[key] for key in keys)
        ),
        None,
    )


def _warnings(
    rows: list[dict[str, Any]],
    passing_rows: list[dict[str, Any]],
) -> list[str]:
    warnings = []
    aggressive = [
        row
        for row in rows
        if row["layer_budget_map"] == AGGRESSIVE_SHORT_CONTEXT_MAP
        and _is_long_context(row)
    ]
    if any(row not in passing_rows for row in aggressive):
        warnings.append(
            "short-context map 0:12,5:8,8:8,11:12 failed on "
            "long-context prompts"
        )
    if any(
        row["average_selected_block_ratio_across_patched_layers"] is not None
        and row["average_selected_block_ratio_across_patched_layers"] <= 0.22
        and row not in passing_rows
        for row in rows
    ):
        warnings.append(
            "very aggressive selected ratios around 0.18-0.21 were unsafe"
        )
    for row in rows:
        if row["policy"] != "query_key_block_score" or row in passing_rows:
            continue
        oracle = _matching_oracle(row, rows)
        if oracle is None:
            continue
        oracle_clean = (
            oracle["exact_sequence_match_rate"] == 1.0
            and oracle["average_token_match_rate"] == 1.0
            and oracle["average_normalized_edit_distance"] == 0.0
        )
        if oracle_clean:
            warnings.append(
                "query-key failure while oracle passes indicates selector "
                "margin risk"
            )
        else:
            warnings.append(
                "oracle failure at aggressive budgets indicates a "
                "budget/risk issue"
            )
    warnings.extend([
        "no vLLM integration has been implemented",
        "no measured runtime memory reduction has been demonstrated",
    ])
    return list(dict.fromkeys(warnings))


def build_readiness_report(
    input_paths: list[str | Path],
    *,
    min_exact_match_rate: float = 1.0,
    min_token_match_rate: float = 1.0,
    max_normalized_edit_distance: float = 0.0,
    max_average_kl: float = 0.01,
    max_selected_ratio_for_tradeoff: float = 0.60,
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one Phase 11.5 result is required")
    rows = _input_rows(input_paths)
    query_rows = [
        row
        for row in rows
        if row["policy"] == "query_key_block_score" and _is_long_context(row)
    ]
    passing = [
        row
        for row in query_rows
        if _passes(
            row,
            min_exact_match_rate=min_exact_match_rate,
            min_token_match_rate=min_token_match_rate,
            max_normalized_edit_distance=max_normalized_edit_distance,
            max_average_kl=max_average_kl,
        )
    ]
    safest = min(
        passing,
        key=lambda row: row["average_per_step_kl_divergence"],
        default=None,
    )
    tradeoff_candidates = [
        row
        for row in passing
        if row["average_selected_block_ratio_across_patched_layers"]
        is not None
        and row["average_selected_block_ratio_across_patched_layers"]
        <= max_selected_ratio_for_tradeoff
    ]
    best_tradeoff = max(
        tradeoff_candidates,
        key=lambda row: (
            row["estimated_active_block_reduction_ratio"]
            if row["estimated_active_block_reduction_ratio"] is not None
            else float("-inf")
        ),
        default=None,
    )
    checks = {
        "long_context_query_key_run_exists": bool(query_rows),
        "passing_long_context_query_key_run_exists": bool(passing),
        "exact_match_threshold_met": any(
            row["exact_sequence_match_rate"] is not None
            and row["exact_sequence_match_rate"] >= min_exact_match_rate
            for row in query_rows
        ),
        "token_match_threshold_met": any(
            row["average_token_match_rate"] is not None
            and row["average_token_match_rate"] >= min_token_match_rate
            for row in query_rows
        ),
        "edit_distance_threshold_met": any(
            row["average_normalized_edit_distance"] is not None
            and row["average_normalized_edit_distance"]
            <= max_normalized_edit_distance
            for row in query_rows
        ),
        "average_kl_threshold_met": any(
            row["average_per_step_kl_divergence"] is not None
            and row["average_per_step_kl_divergence"] <= max_average_kl
            for row in query_rows
        ),
    }
    return {
        "phase11_6_ready": bool(passing),
        "phase12_ready": False,
        "checks": checks,
        "thresholds": {
            "min_exact_match_rate": min_exact_match_rate,
            "min_token_match_rate": min_token_match_rate,
            "max_normalized_edit_distance": max_normalized_edit_distance,
            "max_average_kl": max_average_kl,
            "max_selected_ratio_for_tradeoff": (
                max_selected_ratio_for_tradeoff
            ),
        },
        "safest_passing_config": safest,
        "best_tradeoff_config": best_tradeoff,
        "warnings": _warnings(rows, passing),
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "greedy_generation_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "latency_improvement": False,
            "generation_quality_preservation_claim": False,
        },
        "input_summaries": rows,
        "recommended_next_step": (
            "Phase 11.6 should test ratio/context-scaled maps or a small "
            "model with 2K-8K+ context. GPT-2 is near its context limit."
            if passing
            else "Collect a clean long-context query-key configuration "
            "before Phase 11.6."
        ),
    }


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(
    lines: list[str],
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append(
            "| " + " | ".join(f"`{_format(value)}`" for value in row) + " |"
        )


def _config_section(
    lines: list[str],
    title: str,
    config: dict[str, Any] | None,
) -> None:
    lines.extend(["", f"## {title}", ""])
    if config is None:
        lines.append("No qualifying configuration was found.")
        return
    fields = (
        "layer_budget_map",
        "target_token_length",
        "actual_prompt_token_length",
        "max_new_tokens",
        "exact_sequence_match_rate",
        "average_token_match_rate",
        "average_normalized_edit_distance",
        "average_per_step_kl_divergence",
        "average_selected_block_ratio_across_patched_layers",
        "estimated_active_block_reduction_ratio",
    )
    _table(
        lines,
        ["field", "value"],
        [[field, config.get(field)] for field in fields],
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 11.5 Long-Context Readiness",
        "",
        "## Decision",
        "",
        f"- Phase 11.6 ready: `{_format(report['phase11_6_ready'])}`",
        "- Phase 12 ready: `false` by design.",
    ]
    _config_section(
        lines,
        "Safest Passing Configuration",
        report["safest_passing_config"],
    )
    _config_section(
        lines,
        "Best Quality/Savings Tradeoff",
        report["best_tradeoff_config"],
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend(
        [f"- {warning}" for warning in report["warnings"]]
        or ["- none"]
    )
    lines.extend([
        "",
        "## Caveats",
        "",
        "- Evidence comes from standalone HuggingFace/PyTorch outside vLLM.",
        "- No vLLM integration or active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency improvement is claimed.",
        "- Generation quality preservation is not claimed.",
        "",
        "## Recommended Next Step",
        "",
        report["recommended_next_step"],
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        report = build_readiness_report(
            args.inputs,
            min_exact_match_rate=args.min_exact_match_rate,
            min_token_match_rate=args.min_token_match_rate,
            max_normalized_edit_distance=args.max_normalized_edit_distance,
            max_average_kl=args.max_average_kl,
            max_selected_ratio_for_tradeoff=(
                args.max_selected_ratio_for_tradeoff
            ),
        )
        _write(
            args.output_json,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write(args.output_md, render_markdown(report))
        print(json.dumps({
            "phase11_6_ready": report["phase11_6_ready"],
            "phase12_ready": False,
            "safest_passing_config": report["safest_passing_config"],
            "best_tradeoff_config": report["best_tradeoff_config"],
            "output_json": args.output_json,
            "output_md": args.output_md,
        }, separators=(",", ":")))
        return 0 if report["phase11_6_ready"] else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
