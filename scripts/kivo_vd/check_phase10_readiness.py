#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check Phase 10 selector evidence before offline Phase 11 quality tests."""

import argparse
import json
from pathlib import Path
from typing import Any

MIN_DEPLOYABLE_AVERAGE_COSINE = 0.95
MIN_DEPLOYABLE_MIN_COSINE = 0.90
MAX_DEPLOYABLE_RELATIVE_L2 = 0.50


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Kivo-VD Phase 10 readiness for Phase 11."
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase10_readiness.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase10_readiness.md",
    )
    parser.add_argument("--min-practical-budget", type=int, default=8)
    parser.add_argument("--recommended-budgets", default="8,16,32,64")
    return parser.parse_args(argv)


def parse_int_csv(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("recommended budgets must not be empty")
    if any(item <= 0 for item in result):
        raise ValueError("recommended budgets must be positive")
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"summary file is missing: {input_path}")
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary file must contain a JSON object")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError("summary payload lacks a JSON summary object")
    return summary


def _budgets_from_payload(
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> list[int]:
    config = payload.get("config", {})
    if isinstance(config, dict) and isinstance(config.get("budgets"), list):
        return [int(item) for item in config["budgets"]]
    budgets = []
    for row in summary.get("per_budget", []):
        if isinstance(row, dict) and "candidate_budget_blocks" in row:
            budgets.append(int(row["candidate_budget_blocks"]))
    return sorted(set(budgets))


def _find_policy(
    rows: list[Any],
    policy: str,
) -> dict[str, Any] | None:
    for row in rows:
        if isinstance(row, dict) and row.get("policy") == policy:
            return row
    return None


def _best_deployable(
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    direct = summary.get("best_deployable_selector")
    if isinstance(direct, dict):
        return direct
    candidates = [
        row
        for row in summary.get("per_policy_sketch_dim", [])
        if isinstance(row, dict) and row.get("policy") != "oracle_topk"
    ]
    if not candidates:
        candidates = [
            row
            for row in summary.get("per_policy", [])
            if isinstance(row, dict) and row.get("policy") != "oracle_topk"
        ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -float(row.get("average_cosine_similarity", 0.0)),
            float(row.get("max_relative_l2_error", float("inf"))),
        ),
    )[0]


def _metric_passes(
    best: dict[str, Any] | None,
) -> dict[str, bool]:
    if best is None:
        return {
            "deployable_average_cosine_ok": False,
            "deployable_min_cosine_ok": False,
            "deployable_max_relative_l2_ok": False,
        }
    average_cosine = _number(best.get("average_cosine_similarity"))
    min_cosine = _number(best.get("min_cosine_similarity"))
    max_l2 = _number(best.get("max_relative_l2_error"))
    return {
        "deployable_average_cosine_ok": (
            average_cosine is not None
            and average_cosine >= MIN_DEPLOYABLE_AVERAGE_COSINE
        ),
        "deployable_min_cosine_ok": (
            min_cosine is not None
            and min_cosine >= MIN_DEPLOYABLE_MIN_COSINE
        ),
        "deployable_max_relative_l2_ok": (
            max_l2 is not None and max_l2 <= MAX_DEPLOYABLE_RELATIVE_L2
        ),
    }


def build_readiness_report(
    *,
    summary_path: str | Path,
    min_practical_budget: int = 8,
    recommended_budgets: list[int] | None = None,
) -> dict[str, Any]:
    if min_practical_budget <= 0:
        raise ValueError("min practical budget must be positive")
    if recommended_budgets is None:
        recommended_budgets = [8, 16, 32, 64]

    payload = _load_json(summary_path)
    summary = _summary_from_payload(payload)
    warnings: list[str] = []

    success = bool(payload.get("success", summary.get("num_failed") == 0))
    num_failed = int(summary.get("num_failed", 0) or 0)
    sweep_succeeded = success and num_failed == 0
    if not sweep_succeeded:
        warnings.append("policy sweep did not fully succeed")

    per_policy = summary.get("per_policy", [])
    oracle = _find_policy(per_policy, "oracle_topk")
    if oracle is None:
        warnings.append("oracle_topk summary row is missing")

    best = _best_deployable(summary)
    if best is None:
        warnings.append("best deployable selector is missing")
    elif best.get("policy") == "oracle_topk":
        warnings.append("best deployable selector must not be oracle_topk")

    metric_checks = _metric_passes(best)
    for check_name, passed in metric_checks.items():
        if not passed:
            warnings.append(f"{check_name} is false")

    budgets = _budgets_from_payload(payload, summary)
    if budgets and all(budget < min_practical_budget for budget in budgets):
        warnings.append(
            "sweep only includes budgets below the practical minimum "
            f"{min_practical_budget}; rerun with {recommended_budgets}"
        )

    checks = {
        "sweep_succeeded": sweep_succeeded,
        "oracle_topk_exists": oracle is not None,
        "best_deployable_selector_exists": best is not None,
        "best_deployable_selector_not_oracle": (
            best is not None and best.get("policy") != "oracle_topk"
        ),
        **metric_checks,
    }
    phase11_ready = all(checks.values())
    return {
        "phase11_ready": phase11_ready,
        "summary_path": str(summary_path),
        "checks": checks,
        "warnings": warnings,
        "practical_budget_guidance": {
            "min_practical_budget": min_practical_budget,
            "recommended_budgets": recommended_budgets,
            "observed_budgets": budgets,
            "budget_2": "experimental_only",
            "budget_4": "aggressive_stress_test_only",
            "budget_8": "minimum_practical_candidate",
            "budget_16": "safer_baseline",
            "budget_32_64": "long_context_enterprise_oriented",
        },
        "best_deployable_selector": best,
        "oracle_topk_summary": oracle,
        "allowed_scope": (
            "Phase 11 may evaluate logits/generation quality outside vLLM "
            "only. Runtime/vLLM attention integration remains out of scope."
        ),
        "recommended_next_step": (
            "Run offline logits/generation-quality evaluation with "
            "query_key_block_score and practical budgets such as 8,16,32,64."
            if phase11_ready
            else "Rerun or improve Phase 10 selector evidence before Phase 11."
        ),
        "caveats": {
            "outside_vllm": True,
            "no_logits_or_generation_quality_yet": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "vllm_integration": False,
            "latency_improvement": False,
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
    best = report.get("best_deployable_selector") or {}
    oracle = report.get("oracle_topk_summary") or {}
    lines = [
        "# Kivo-VD Phase 10.4 Readiness Gate",
        "",
        f"- Phase 11 ready: `{_format(report['phase11_ready'])}`",
        f"- Allowed scope: {report['allowed_scope']}",
        "",
        "## Checks",
        "",
    ]
    _append_table(
        lines,
        ["check", "passed"],
        [[key, value] for key, value in report["checks"].items()],
    )
    lines.extend(["", "## Best Deployable Selector", ""])
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in best.items()],
    )
    lines.extend(["", "## Oracle Top-K Summary", ""])
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in oracle.items()],
    )
    guidance = report["practical_budget_guidance"]
    lines.extend(["", "## Practical Budget Guidance", ""])
    _append_table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in guidance.items()],
    )
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Caveats",
        "",
        "- Evaluation remains outside vLLM.",
        "- No logits or generation quality has been measured yet.",
        "- No active routing is implemented.",
        "- No measured runtime memory reduction is claimed.",
        "- No vLLM integration is authorized by this gate.",
        "- No latency improvement is claimed.",
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
            summary_path=args.summary,
            min_practical_budget=args.min_practical_budget,
            recommended_budgets=parse_int_csv(args.recommended_budgets),
        )
        _write_json(args.output_json, report)
        _write_text(args.output_md, render_markdown(report))
        print(
            json.dumps(
                {
                    "phase11_ready": report["phase11_ready"],
                    "best_deployable_selector": (
                        report["best_deployable_selector"] or {}
                    ).get("policy"),
                    "warnings": report["warnings"],
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["phase11_ready"] else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
