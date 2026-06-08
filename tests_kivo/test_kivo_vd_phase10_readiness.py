# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "check_phase10_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase10_readiness", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selector(
    policy: str,
    *,
    average_cosine: float,
    min_cosine: float,
    max_relative_l2: float,
) -> dict:
    return {
        "policy": policy,
        "sketch_dim": None,
        "count": 15,
        "average_cosine_similarity": average_cosine,
        "min_cosine_similarity": min_cosine,
        "average_relative_l2_error": 0.15,
        "max_relative_l2_error": max_relative_l2,
        "average_attention_mass_captured": 0.90,
    }


def _write_summary(
    tmp_path: Path,
    *,
    best: dict | None,
    budgets: list[int] | None = None,
    include_oracle: bool = True,
) -> Path:
    if budgets is None:
        budgets = [8, 16]
    oracle = _selector(
        "oracle_topk",
        average_cosine=0.99,
        min_cosine=0.95,
        max_relative_l2=0.32,
    )
    per_policy = [oracle] if include_oracle else []
    if best is not None:
        per_policy.append(best)
    payload = {
        "success": True,
        "config": {"budgets": budgets},
        "summary": {
            "num_runs": len(per_policy),
            "num_succeeded": len(per_policy),
            "num_failed": 0,
            "per_policy": per_policy,
            "per_policy_sketch_dim": per_policy,
            "per_budget": [
                {"candidate_budget_blocks": budget} for budget in budgets
            ],
            "best_deployable_selector": best,
        },
    }
    path = tmp_path / "policy_sweep_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_readiness_true_for_strong_query_key_selector(tmp_path: Path) -> None:
    module = _load_module()
    best = _selector(
        "query_key_block_score",
        average_cosine=0.986,
        min_cosine=0.944,
        max_relative_l2=0.336,
    )
    summary_path = _write_summary(tmp_path, best=best)

    report = module.build_readiness_report(summary_path=summary_path)

    assert report["phase11_ready"] is True
    assert report["best_deployable_selector"]["policy"] == (
        "query_key_block_score"
    )
    assert "outside vLLM" in report["allowed_scope"]


def test_readiness_false_if_only_oracle_exists(tmp_path: Path) -> None:
    module = _load_module()
    summary_path = _write_summary(tmp_path, best=None)

    report = module.build_readiness_report(summary_path=summary_path)

    assert report["phase11_ready"] is False
    assert report["checks"]["best_deployable_selector_exists"] is False


def test_readiness_false_if_min_cosine_is_too_low(tmp_path: Path) -> None:
    module = _load_module()
    best = _selector(
        "query_key_block_score",
        average_cosine=0.98,
        min_cosine=0.89,
        max_relative_l2=0.40,
    )
    summary_path = _write_summary(tmp_path, best=best)

    report = module.build_readiness_report(summary_path=summary_path)

    assert report["phase11_ready"] is False
    assert report["checks"]["deployable_min_cosine_ok"] is False


def test_warns_if_all_budgets_are_below_practical_minimum(
    tmp_path: Path,
) -> None:
    module = _load_module()
    best = _selector(
        "query_key_block_score",
        average_cosine=0.98,
        min_cosine=0.92,
        max_relative_l2=0.40,
    )
    summary_path = _write_summary(tmp_path, best=best, budgets=[2, 4])

    report = module.build_readiness_report(summary_path=summary_path)

    assert any("only includes budgets below" in item for item in report[
        "warnings"
    ])
    assert report["practical_budget_guidance"]["recommended_budgets"] == [
        8,
        16,
        32,
        64,
    ]


def test_markdown_contains_required_caveats(tmp_path: Path) -> None:
    module = _load_module()
    best = _selector(
        "query_key_block_score",
        average_cosine=0.98,
        min_cosine=0.92,
        max_relative_l2=0.40,
    )
    summary_path = _write_summary(tmp_path, best=best)
    report = module.build_readiness_report(summary_path=summary_path)

    markdown = module.render_markdown(report)

    assert "outside vLLM" in markdown
    assert "No logits or generation quality has been measured yet" in markdown
    assert "No active routing is implemented" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No vLLM integration is authorized" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "kivo_vd" / "check_phase10_readiness.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--summary",
        "--output-json",
        "--output-md",
        "--min-practical-budget",
        "--recommended-budgets",
    ):
        assert flag in process.stdout
