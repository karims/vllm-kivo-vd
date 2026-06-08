# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_ratio_scaled_long_context_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_ratio_scaled_long_context_sweep",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    ratio_name: str = "balanced",
    policy: str = "query_key_block_score",
    exact: float = 1.0,
    token_match: float = 1.0,
    edit_distance: float = 0.0,
    kl: float = 0.001,
    selected_ratio: float = 0.5,
) -> dict:
    return {
        "status": "succeeded",
        "ratio_policy_name": ratio_name,
        "ratio_policy_spec": "0:0.6,5:0.45,8:0.45,11:0.6",
        "derived_layer_budget_map": "0:35,5:27,8:27,11:35",
        "target_token_length": 960,
        "average_actual_prompt_tokens": 917.0,
        "estimated_context_blocks": 58,
        "policy": policy,
        "max_new_tokens": 16,
        "num_prompts": 2,
        "exact_sequence_match_rate": exact,
        "average_token_match_rate": token_match,
        "average_prefix_match_length": 16.0,
        "average_normalized_edit_distance": edit_distance,
        "average_per_step_kl_divergence": kl,
        "average_per_step_top1_match_rate": token_match,
        "average_selected_block_ratio_across_patched_layers": selected_ratio,
        "estimated_active_block_reduction_ratio": 1.0 - selected_ratio,
        "failure_flags": [],
        "warnings": [],
    }


def test_parses_ratio_policies() -> None:
    module = _load_module()

    parsed = module.parse_ratio_policies(
        "balanced=0:0.60,5:0.45;safer=0:0.70,5:0.55"
    )

    assert parsed == {
        "balanced": {0: 0.60, 5: 0.45},
        "safer": {0: 0.70, 5: 0.55},
    }


def test_derive_budget_map_with_clamping() -> None:
    module = _load_module()

    result = module.derive_layer_budget_map(
        ratios={0: 0.60, 5: 0.10, 8: 2.0},
        num_blocks=58,
        min_budget=8,
        max_budget=40,
        rounding="ceil",
    )

    assert result == {0: 35, 5: 8, 8: 40}


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [
        ("floor", 26),
        ("ceil", 27),
        ("round", 26),
    ],
)
def test_budget_rounding_modes(rounding: str, expected: int) -> None:
    module = _load_module()

    result = module.derive_layer_budget_map(
        ratios={5: 0.45},
        num_blocks=58,
        min_budget=1,
        max_budget=None,
        rounding=rounding,
    )

    assert result[5] == expected


def test_dry_run_creates_planned_rows_without_model_download(
    tmp_path: Path,
) -> None:
    module = _load_module()
    args = module._parse_args([
        "--dry-run",
        "--target-token-lengths",
        "768",
        "--ratio-policies",
        "balanced=0:0.60,5:0.45,8:0.45,11:0.60",
        "--policies",
        "query_key_block_score,oracle_topk",
        "--max-new-tokens-values",
        "16",
        "--output-dir",
        str(tmp_path),
    ])

    result = module.run_sweep(args)

    assert result["summary"]["counts"] == {
        "total": 2,
        "succeeded": 0,
        "failed": 0,
        "planned": 2,
    }
    rows = [
        json.loads(line)
        for line in Path(result["rows_path"]).read_text().splitlines()
    ]
    assert rows[0]["derived_layer_budget_map"] == "0:29,5:22,8:22,11:29"
    assert result["prompts_json"] is None


def test_summary_aggregation_from_fake_rows() -> None:
    module = _load_module()
    rows = [
        _row(ratio_name="balanced"),
        _row(ratio_name="balanced", policy="oracle_topk", kl=0.0001),
    ]

    summary = module.build_summary(
        rows,
        config={"model": "gpt2"},
        derived_maps=[],
    )

    assert summary["counts"]["succeeded"] == 2
    assert summary["readiness"]["phase11_7_ready"] is True
    assert summary["readiness"]["phase12_ready"] is False
    assert summary["per_ratio_policy"][0]["count"] == 2


def test_oracle_gap_calculation() -> None:
    module = _load_module()
    rows = [
        _row(kl=0.02, exact=0.8, token_match=0.9, edit_distance=0.1),
        _row(policy="oracle_topk", kl=0.005),
    ]

    gap = module.calculate_oracle_gaps(rows)[0]

    assert gap["query_minus_oracle_kl"] == pytest.approx(0.015)
    assert gap["oracle_minus_query_exact_match"] == pytest.approx(0.2)
    assert gap["oracle_minus_query_token_match"] == pytest.approx(0.1)
    assert gap["query_minus_oracle_edit_distance"] == pytest.approx(0.1)


def test_best_tradeoff_prefers_reduction_and_excludes_oracle() -> None:
    module = _load_module()
    rows = [
        _row(ratio_name="safer", selected_ratio=0.7, kl=0.0001),
        _row(ratio_name="balanced", selected_ratio=0.48, kl=0.002),
        _row(ratio_name="oracle", policy="oracle_topk", selected_ratio=0.2),
    ]

    best = module.best_deployable_tradeoff(rows)

    assert best["ratio_policy_name"] == "balanced"


def test_safest_passing_prefers_lower_kl() -> None:
    module = _load_module()
    rows = [
        _row(ratio_name="safer", selected_ratio=0.7, kl=0.0001),
        _row(ratio_name="balanced", selected_ratio=0.48, kl=0.002),
    ]

    safest = module.safest_passing_config(rows)

    assert safest["ratio_policy_name"] == "safer"


def test_failure_flags() -> None:
    module = _load_module()
    row = _row(
        exact=0.5,
        token_match=0.98,
        edit_distance=0.1,
        kl=0.02,
        selected_ratio=0.9,
    )
    row["estimated_active_block_reduction_ratio"] = 0.1
    row["average_actual_prompt_tokens"] = 700.0

    assert set(module.failure_flags(row)) == {
        "exact_sequence_match_below_1",
        "token_match_below_0.99",
        "normalized_edit_distance_above_0",
        "average_kl_above_0.01",
        "per_step_top1_below_1",
        "selected_ratio_above_0.85",
        "estimated_reduction_below_0.20",
        "actual_prompt_length_too_short",
    }


def test_markdown_caveats() -> None:
    module = _load_module()
    summary = module.build_summary(
        [_row(), _row(policy="oracle_topk")],
        config={"model": "gpt2"},
        derived_maps=[],
    )

    markdown = module.render_markdown(summary)

    assert "outside vLLM" in markdown
    assert "No vLLM integration or active routing" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No latency claim" in markdown
    assert "generation-quality probe" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_ratio_scaled_long_context_sweep.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--target-token-lengths",
        "--ratio-policies",
        "--min-budget",
        "--max-budget",
        "--budget-rounding",
        "--policies",
        "--max-new-tokens-values",
        "--dry-run",
        "--continue-on-error",
    ):
        assert flag in process.stdout
