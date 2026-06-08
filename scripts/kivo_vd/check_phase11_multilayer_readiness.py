#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check multi-layer generation evidence before a larger offline sweep."""

import argparse
import json
from pathlib import Path
from typing import Any

RECOMMENDED_ADAPTIVE_MAP = {"0": 12, "5": 8, "8": 8, "11": 12}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Phase 11.3 multi-layer generation readiness for a larger "
            "offline Phase 11.4 sweep."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase11_4_multilayer_readiness.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase11_4_multilayer_readiness.md",
    )
    parser.add_argument("--min-exact-match-rate", type=float, default=1.0)
    parser.add_argument("--min-token-match-rate", type=float, default=1.0)
    parser.add_argument(
        "--max-normalized-edit-distance",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-average-kl", type=float, default=0.01)
    return parser.parse_args(argv)


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"multi-layer result is missing: {input_path}")
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"multi-layer result must be an object: {input_path}")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _normalize_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for layer, budget in value.items():
        try:
            layer_text = str(int(layer))
            budget_value = int(budget)
        except (TypeError, ValueError):
            return {}
        result[layer_text] = budget_value
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _input_summary(
    path: str | Path,
    payload: dict[str, Any],
    *,
    min_exact_match_rate: float,
    min_token_match_rate: float,
    max_normalized_edit_distance: float,
    max_average_kl: float,
) -> dict[str, Any]:
    config = payload.get("config", {})
    aggregate = payload.get("aggregate", {})
    caveats = payload.get("caveats", {})
    if not all(isinstance(item, dict) for item in (config, aggregate, caveats)):
        raise ValueError(f"result has malformed config/aggregate: {path}")
    layer_map = _normalize_map(payload.get("layer_budget_map"))
    policy = config.get("selection_policy")
    exact = _number(aggregate.get("exact_sequence_match_rate"))
    token_match = _number(aggregate.get("average_token_match_rate"))
    edit_distance = _number(
        aggregate.get("average_normalized_edit_distance")
    )
    average_kl = _number(aggregate.get("average_per_step_kl_divergence"))
    if not isinstance(policy, str) or any(
        value is None
        for value in (exact, token_match, edit_distance, average_kl)
    ):
        raise ValueError(f"result lacks required metrics: {path}")
    caveats_ok = all([
        caveats.get("outside_vllm") is True,
        caveats.get("no_vllm_integration") is True,
        caveats.get("no_measured_runtime_reduction") is True,
    ])
    clean = all([
        exact >= min_exact_match_rate,
        token_match >= min_token_match_rate,
        edit_distance <= max_normalized_edit_distance,
        average_kl <= max_average_kl,
        caveats_ok,
    ])
    return {
        "input_path": str(path),
        "policy": policy,
        "layer_budget_map": layer_map,
        "max_new_tokens": config.get("max_new_tokens"),
        "exact_sequence_match_rate": exact,
        "average_token_match_rate": token_match,
        "average_normalized_edit_distance": edit_distance,
        "average_per_step_kl_divergence": average_kl,
        "average_per_step_top1_match_rate": _number(
            aggregate.get("average_per_step_top1_match_rate")
        ),
        "average_selected_block_ratio_across_patched_layers": _number(
            aggregate.get(
                "average_selected_block_ratio_across_patched_layers"
            )
        ),
        "required_caveats_present": caveats_ok,
        "clean": clean,
    }


def build_readiness_report(
    input_paths: list[str | Path],
    *,
    min_exact_match_rate: float = 1.0,
    min_token_match_rate: float = 1.0,
    max_normalized_edit_distance: float = 0.0,
    max_average_kl: float = 0.01,
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one Phase 11.3 result is required")
    rows = [
        _input_summary(
            path,
            _load_json(path),
            min_exact_match_rate=min_exact_match_rate,
            min_token_match_rate=min_token_match_rate,
            max_normalized_edit_distance=max_normalized_edit_distance,
            max_average_kl=max_average_kl,
        )
        for path in input_paths
    ]
    adaptive_query_rows = [
        row
        for row in rows
        if row["policy"] == "query_key_block_score"
        and row["layer_budget_map"] == RECOMMENDED_ADAPTIVE_MAP
    ]
    clean_adaptive_query_rows = [
        row for row in adaptive_query_rows if row["clean"]
    ]
    naive_map = {"5": 8, "8": 8, "11": 8}
    naive_query_failures = [
        row
        for row in rows
        if row["policy"] == "query_key_block_score"
        and row["layer_budget_map"] == naive_map
        and not row["clean"]
    ]
    naive_oracle_passes = [
        row
        for row in rows
        if row["policy"] == "oracle_topk"
        and row["layer_budget_map"] == naive_map
        and row["clean"]
    ]
    warnings = []
    if naive_query_failures:
        warnings.append(
            "query_key_block_score failed for naive map 5:8,8:8,11:8"
        )
    if naive_query_failures and naive_oracle_passes:
        warnings.append(
            "oracle passed where query-key failed, indicating a "
            "selector/accumulation issue"
        )
    warnings.extend([
        "layer 11 requires budget 12 under the tested multi-layer map",
        "layer 0 requires budget at least 12 based on Phase 11.2",
    ])
    checks = {
        "adaptive_query_key_run_exists": bool(adaptive_query_rows),
        "adaptive_query_key_run_clean": bool(clean_adaptive_query_rows),
        "exact_match_threshold_met": any(
            row["exact_sequence_match_rate"] >= min_exact_match_rate
            for row in adaptive_query_rows
        ),
        "token_match_threshold_met": any(
            row["average_token_match_rate"] >= min_token_match_rate
            for row in adaptive_query_rows
        ),
        "edit_distance_threshold_met": any(
            row["average_normalized_edit_distance"]
            <= max_normalized_edit_distance
            for row in adaptive_query_rows
        ),
        "average_kl_threshold_met": any(
            row["average_per_step_kl_divergence"] <= max_average_kl
            for row in adaptive_query_rows
        ),
        "required_caveats_present": any(
            row["required_caveats_present"] for row in adaptive_query_rows
        ),
    }
    phase11_4_ready = all(checks.values())
    return {
        "phase11_4_ready": phase11_4_ready,
        "recommended_adaptive_layer_budget_map": RECOMMENDED_ADAPTIVE_MAP,
        "checks": checks,
        "thresholds": {
            "min_exact_match_rate": min_exact_match_rate,
            "min_token_match_rate": min_token_match_rate,
            "max_normalized_edit_distance": max_normalized_edit_distance,
            "max_average_kl": max_average_kl,
        },
        "warnings": warnings,
        "input_summaries": rows,
        "recommended_next_step": (
            "Phase 11.4 should run a larger offline generation sweep outside "
            "vLLM using map 0:12,5:8,8:8,11:12, more prompts, and "
            "max_new_tokens 32/64 for query-key and oracle."
            if phase11_4_ready
            else "Collect a clean adaptive query-key result before Phase 11.4."
        ),
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "multilayer_evidence_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "latency_improvement": False,
            "generation_quality_preservation_claim": False,
        },
    }


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _append_table(
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


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 11.4 Multi-Layer Readiness",
        "",
        f"- Phase 11.4 ready: `{_format(report['phase11_4_ready'])}`",
        "",
        "## Recommended Adaptive Map",
        "",
    ]
    _append_table(
        lines,
        ["layer", "budget"],
        [
            [layer, budget]
            for layer, budget in report[
                "recommended_adaptive_layer_budget_map"
            ].items()
        ],
    )
    lines.extend(["", "## Checks", ""])
    _append_table(
        lines,
        ["check", "passed"],
        [[key, value] for key, value in report["checks"].items()],
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
        "- Evidence comes from standalone multi-layer tests outside vLLM.",
        "- No vLLM integration is implemented or authorized.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency improvement is claimed.",
        "- Generation quality preservation is not claimed.",
        "",
        "## Recommended Next Step",
        "",
        report["recommended_next_step"],
    ])
    return "\n".join(lines) + "\n"


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
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
        )
        _write_json(args.output_json, report)
        _write_text(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "phase11_4_ready": report["phase11_4_ready"],
                    "recommended_adaptive_layer_budget_map": report[
                        "recommended_adaptive_layer_budget_map"
                    ],
                    "warnings": report["warnings"],
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["phase11_4_ready"] else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
