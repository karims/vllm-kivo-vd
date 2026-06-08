#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Summarize Phase 11.2 generation evidence for adaptive-budget testing."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_EXACT_MATCH_RATE = 0.95
MIN_TOKEN_MATCH_RATE = 0.95
MAX_AVERAGE_KL = 0.01


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Phase 11.2 single-layer generation evidence before "
            "adaptive multi-layer tests."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase11_generation_readiness.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase11_generation_readiness.md",
    )
    return parser.parse_args(argv)


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"generation result is missing: {input_path}")
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"generation result must be an object: {input_path}")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _result_row(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    aggregate = payload.get("aggregate", {})
    if not isinstance(config, dict) or not isinstance(aggregate, dict):
        raise ValueError(f"result lacks config or aggregate object: {path}")
    layer = config.get("layer_index")
    budget = config.get("candidate_budget_blocks")
    policy = config.get("selection_policy")
    if not isinstance(layer, int) or not isinstance(budget, int):
        raise ValueError(f"result lacks integer layer/budget: {path}")
    if not isinstance(policy, str):
        raise ValueError(f"result lacks selection policy: {path}")
    exact = _number(aggregate.get("exact_sequence_match_rate"))
    token_match = _number(aggregate.get("average_token_match_rate"))
    average_kl = _number(aggregate.get("average_per_step_kl_divergence"))
    if exact is None or token_match is None or average_kl is None:
        raise ValueError(f"result lacks required generation metrics: {path}")
    clean = (
        exact >= MIN_EXACT_MATCH_RATE
        and token_match >= MIN_TOKEN_MATCH_RATE
        and average_kl <= MAX_AVERAGE_KL
    )
    return {
        "input_path": str(path),
        "layer_index": layer,
        "candidate_budget_blocks": budget,
        "policy": policy,
        "max_new_tokens": config.get("max_new_tokens"),
        "exact_sequence_match_rate": exact,
        "average_token_match_rate": token_match,
        "average_prefix_match_length": _number(
            aggregate.get("average_prefix_match_length")
        ),
        "average_normalized_edit_distance": _number(
            aggregate.get("average_normalized_edit_distance")
        ),
        "average_per_step_kl_divergence": average_kl,
        "average_per_step_top1_match_rate": _number(
            aggregate.get("average_per_step_top1_match_rate")
        ),
        "average_selected_block_ratio": _number(
            aggregate.get("average_selected_block_ratio")
        ),
        "clean": clean,
    }


def _adaptive_budget_map(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["layer_index"]].append(row)
    recommendations = {}
    for layer, layer_rows in sorted(grouped.items()):
        budgets = sorted({row["candidate_budget_blocks"] for row in layer_rows})
        clean_budgets = []
        for budget in budgets:
            budget_rows = [
                row
                for row in layer_rows
                if row["candidate_budget_blocks"] == budget
            ]
            policies = {row["policy"] for row in budget_rows}
            required = {"query_key_block_score", "oracle_topk"}
            if required.issubset(policies) and all(
                row["clean"]
                for row in budget_rows
                if row["policy"] in required
            ):
                clean_budgets.append(budget)
        recommendations[str(layer)] = {
            "minimum_clean_observed_budget": (
                min(clean_budgets) if clean_budgets else None
            ),
            "safer_recommended_budget": (
                max(16, min(clean_budgets)) if clean_budgets else None
            ),
            "observed_budgets": budgets,
            "divergent_budgets": sorted({
                row["candidate_budget_blocks"]
                for row in layer_rows
                if not row["clean"]
            }),
        }
    return recommendations


def build_readiness_report(
    input_paths: list[str | Path],
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one generation result is required")
    rows = [_result_row(path, _load_json(path)) for path in input_paths]
    warnings = []
    divergent = [row for row in rows if not row["clean"]]
    for row in divergent:
        warnings.append(
            f"layer {row['layer_index']} budget "
            f"{row['candidate_budget_blocks']} policy {row['policy']} "
            "has generation divergence"
        )
    budget_map = _adaptive_budget_map(rows)
    missing_clean_layers = [
        layer
        for layer, recommendation in budget_map.items()
        if recommendation["minimum_clean_observed_budget"] is None
    ]
    if missing_clean_layers:
        warnings.append(
            "no clean paired query-key/oracle budget for layers "
            + ",".join(missing_clean_layers)
        )
    phase11_3_ready = bool(budget_map) and not missing_clean_layers
    return {
        "phase11_3_ready": phase11_3_ready,
        "num_inputs": len(input_paths),
        "num_clean_results": sum(row["clean"] for row in rows),
        "num_divergent_results": len(divergent),
        "any_layer_budget_divergence": bool(divergent),
        "thresholds": {
            "minimum_exact_sequence_match_rate": MIN_EXACT_MATCH_RATE,
            "minimum_token_match_rate": MIN_TOKEN_MATCH_RATE,
            "maximum_average_per_step_kl": MAX_AVERAGE_KL,
        },
        "adaptive_budget_map": budget_map,
        "result_rows": rows,
        "warnings": warnings,
        "allowed_scope": (
            "Phase 11.3 may test adaptive multi-layer generation patches "
            "outside vLLM only."
        ),
        "recommended_next_step": (
            "Use layer 0 budget 12 or 16 and layers 5/8/11 budget 8 or 16 "
            "for conservative multi-layer tests outside vLLM."
            if phase11_3_ready
            else "Collect clean paired selector/oracle results before "
            "multi-layer testing."
        ),
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "single_layer_evidence_only": True,
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
        "# Kivo-VD Phase 11 Generation Readiness",
        "",
        f"- Phase 11.3 ready: `{_format(report['phase11_3_ready'])}`",
        f"- Allowed scope: {report['allowed_scope']}",
        "",
        "## Adaptive Budget Map",
        "",
    ]
    _append_table(
        lines,
        ["layer", "minimum clean", "safer budget", "divergent budgets"],
        [
            [
                layer,
                values["minimum_clean_observed_budget"],
                values["safer_recommended_budget"],
                values["divergent_budgets"],
            ]
            for layer, values in report["adaptive_budget_map"].items()
        ],
    )
    lines.extend(["", "## Result Rows", ""])
    _append_table(
        lines,
        ["layer", "budget", "policy", "exact", "token match", "KL", "clean"],
        [
            [
                row["layer_index"],
                row["candidate_budget_blocks"],
                row["policy"],
                row["exact_sequence_match_rate"],
                row["average_token_match_rate"],
                row["average_per_step_kl_divergence"],
                row["clean"],
            ]
            for row in report["result_rows"]
        ],
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
        "- Evidence comes from standalone single-layer tests outside vLLM.",
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
        report = build_readiness_report(args.inputs)
        _write_json(args.output_json, report)
        _write_text(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "phase11_3_ready": report["phase11_3_ready"],
                    "any_layer_budget_divergence": report[
                        "any_layer_budget_divergence"
                    ],
                    "adaptive_budget_map": report["adaptive_budget_map"],
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["phase11_3_ready"] else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
