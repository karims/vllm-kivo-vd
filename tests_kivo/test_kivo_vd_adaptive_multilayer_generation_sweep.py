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
        / "run_adaptive_multilayer_generation_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_adaptive_multilayer_generation_sweep",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    policy: str,
    layer_map: str = "0:12,5:8,8:8,11:12",
    max_new_tokens: int = 32,
    exact: float = 1.0,
    token_match: float = 1.0,
    edit_distance: float = 0.0,
    kl: float = 0.001,
    top1: float = 1.0,
) -> dict:
    return {
        "status": "succeeded",
        "policy": policy,
        "layer_budget_map": layer_map,
        "max_new_tokens": max_new_tokens,
        "prompt_set": "default",
        "num_prompts": 5,
        "exact_sequence_match_rate": exact,
        "average_token_match_rate": token_match,
        "average_prefix_match_length": 32.0,
        "average_normalized_edit_distance": edit_distance,
        "average_per_step_kl_divergence": kl,
        "average_per_step_top1_match_rate": top1,
        "average_selected_block_ratio_across_patched_layers": 0.4,
        "failure_flags": [],
        "warning": None,
    }


def test_parses_semicolon_layer_budget_maps() -> None:
    module = _load_module()

    result = module.parse_layer_budget_maps(
        "0:12,5:8,8:8,11:12;0:16,5:8,8:8,11:16"
    )

    assert result == [
        {0: 12, 5: 8, 8: 8, 11: 12},
        {0: 16, 5: 8, 8: 8, 11: 16},
    ]


def test_parses_max_new_tokens_values() -> None:
    module = _load_module()

    assert module.parse_int_csv(
        "32,64",
        label="--max-new-tokens-values",
    ) == [32, 64]

    with pytest.raises(ValueError, match="positive"):
        module.parse_int_csv("32,0", label="tokens")


def test_dry_run_creates_planned_rows_without_model_download(
    tmp_path: Path,
) -> None:
    module = _load_module()
    args = module._parse_args([
        "--dry-run",
        "--max-new-tokens-values",
        "32",
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
    assert {row["status"] for row in rows} == {"planned"}


def test_summary_aggregation_and_readiness() -> None:
    module = _load_module()
    rows = [
        _row(policy="query_key_block_score"),
        _row(policy="oracle_topk", kl=0.0001),
    ]

    summary = module.build_summary(rows, config={"model": "gpt2"})

    assert summary["counts"]["succeeded"] == 2
    assert len(summary["per_policy"]) == 2
    assert summary["readiness"]["phase11_5_ready"] is True
    assert summary["readiness"]["phase12_ready"] is False


def test_oracle_gap_calculation() -> None:
    module = _load_module()
    rows = [
        _row(
            policy="query_key_block_score",
            exact=0.8,
            token_match=0.9,
            edit_distance=0.1,
            kl=0.02,
        ),
        _row(
            policy="oracle_topk",
            exact=1.0,
            token_match=1.0,
            edit_distance=0.0,
            kl=0.005,
        ),
    ]

    gap = module.calculate_oracle_gaps(rows)[0]

    assert gap["query_minus_oracle_kl"] == pytest.approx(0.015)
    assert gap["oracle_minus_query_exact_match"] == pytest.approx(0.2)
    assert gap["oracle_minus_query_token_match"] == pytest.approx(0.1)
    assert gap["query_minus_oracle_edit_distance"] == pytest.approx(0.1)


def test_best_deployable_excludes_oracle() -> None:
    module = _load_module()
    rows = [
        _row(policy="oracle_topk", kl=0.0),
        _row(policy="query_key_block_score", kl=0.001),
        _row(
            policy="recent",
            exact=0.8,
            token_match=0.9,
            edit_distance=0.1,
            kl=0.02,
        ),
    ]

    best = module.best_deployable_config(rows)

    assert best is not None
    assert best["policy"] == "query_key_block_score"


def test_failure_flags_cover_thresholds() -> None:
    module = _load_module()
    row = _row(
        policy="query_key_block_score",
        exact=0.8,
        token_match=0.98,
        edit_distance=0.1,
        kl=0.02,
        top1=0.9,
    )

    assert set(module.failure_flags(row)) == {
        "exact_sequence_match_below_1",
        "token_match_below_0.99",
        "normalized_edit_distance_above_0",
        "average_kl_above_0.01",
        "per_step_top1_below_1",
    }


def test_markdown_contains_required_caveats() -> None:
    module = _load_module()
    summary = module.build_summary(
        [
            _row(policy="query_key_block_score"),
            _row(policy="oracle_topk"),
        ],
        config={"model": "gpt2"},
    )

    markdown = module.render_markdown(summary)

    assert "outside vLLM" in markdown
    assert "No vLLM integration or active routing" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No latency claim" in markdown
    assert "generation-quality probe" in markdown


def test_prompt_sets_have_expected_sizes() -> None:
    module = _load_module()
    default = [f"prompt {index}" for index in range(5)]

    assert len(module.read_prompts(
        prompts_file=None,
        prompt_set="default",
        default_prompts=default,
    )) == 5
    extended = module.read_prompts(
        prompts_file=None,
        prompt_set="extended",
        default_prompts=default,
    )
    assert 10 <= len(extended) <= 15
    assert extended[:5] == default


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_adaptive_multilayer_generation_sweep.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--prompts-file",
        "--layer-budget-maps",
        "--policies",
        "--max-new-tokens-values",
        "--prompt-set",
        "--teacher-forced-context",
        "--output-dir",
        "--dry-run",
        "--continue-on-error",
    ):
        assert flag in process.stdout
