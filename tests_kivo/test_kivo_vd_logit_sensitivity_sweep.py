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
        / "run_logit_sensitivity_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_logit_sensitivity_sweep",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    policy: str,
    layer: int = 0,
    budget: int = 8,
    sketch_dim: int | None = None,
    top1: float = 1.0,
    kl: float = 0.001,
    logits_l2: float = 0.01,
    top5: float = 5.0,
    attention_l2: float = 0.05,
) -> dict:
    return {
        "policy": policy,
        "layer_index": layer,
        "candidate_budget_blocks": budget,
        "block_size": 16,
        "sketch_dim": sketch_dim,
        "status": "succeeded",
        "warning": None,
        "num_prompts": 5,
        "average_logits_cosine_similarity": 0.999,
        "average_logits_relative_l2_error": logits_l2,
        "average_kl_divergence": kl,
        "top1_match_rate": top1,
        "average_top5_overlap": top5,
        "average_top10_overlap": 10.0,
        "average_attention_output_cosine": 0.99,
        "average_attention_output_relative_l2": attention_l2,
        "failure_flags": [],
    }


def test_parses_layers_budgets_and_policies() -> None:
    module = _load_module()

    assert module._parse_int_csv("0,5,8,11") == [0, 5, 8, 11]
    assert module._parse_int_csv("8,16,32") == [8, 16, 32]
    assert module.parse_policies(
        "query_key_block_score,oracle_topk,count_sketch"
    ) == ["query_key_block_score", "oracle_topk", "count_sketch"]


def test_builds_sweep_combinations_and_expands_sketch_dims() -> None:
    module = _load_module()

    combinations = module.build_combinations(
        layers=[0, 5],
        budgets=[8],
        block_sizes=[16],
        policies=["query_key_block_score", "count_sketch"],
        sketch_dims=[16, 32],
    )

    assert len(combinations) == 6
    assert {
        row["sketch_dim"]
        for row in combinations
        if row["policy"] == "query_key_block_score"
    } == {None}
    assert {
        row["sketch_dim"]
        for row in combinations
        if row["policy"] == "count_sketch"
    } == {16, 32}


def test_dry_run_writes_planned_rows_without_model_download(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_logit_sensitivity_sweep.py"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--layers",
            "0,5",
            "--budgets",
            "8",
            "--policies",
            "query_key_block_score,oracle_topk",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(process.stdout)
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "logit_sensitivity_runs.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]

    assert payload["dry_run"] is True
    assert payload["summary"]["num_runs"] == 4
    assert len(rows) == 4
    assert all(row["status"] == "planned" for row in rows)
    assert (tmp_path / "logit_sensitivity_summary.json").exists()
    assert (tmp_path / "logit_sensitivity_summary.md").exists()


def test_summary_aggregation_and_best_deployable() -> None:
    module = _load_module()
    rows = [
        _row(policy="oracle_topk", kl=0.0001, logits_l2=0.001),
        _row(
            policy="query_key_block_score",
            kl=0.001,
            logits_l2=0.01,
        ),
        _row(
            policy="recent",
            top1=0.8,
            kl=0.02,
            logits_l2=0.08,
            top5=3.0,
        ),
    ]

    summary = module.summarize_rows(rows)

    assert summary["num_runs"] == 3
    assert summary["num_succeeded"] == 3
    assert summary["per_policy"][0]["count"] == 1
    assert summary["best_deployable_policy"]["policy"] == (
        "query_key_block_score"
    )
    assert summary["best_deployable_policy"]["policy"] != "oracle_topk"
    assert summary["phase11_2_ready"] is True


def test_oracle_gap_calculation() -> None:
    module = _load_module()
    oracle = _row(
        policy="oracle_topk",
        kl=0.001,
        logits_l2=0.01,
        attention_l2=0.04,
    )
    selector = _row(
        policy="query_key_block_score",
        top1=0.8,
        kl=0.004,
        logits_l2=0.03,
        attention_l2=0.10,
    )

    gap = module.calculate_oracle_gaps([oracle, selector])[0]

    assert gap["policy"] == "query_key_block_score"
    assert gap["kl_gap"] == pytest.approx(0.003)
    assert gap["top1_gap"] == pytest.approx(0.2)
    assert gap["logits_l2_gap"] == pytest.approx(0.02)
    assert gap["attention_l2_gap"] == pytest.approx(0.06)


def test_failure_flags() -> None:
    module = _load_module()
    row = _row(
        policy="recent",
        top1=0.90,
        kl=0.02,
        logits_l2=0.06,
        top5=3.5,
    )

    assert module.failure_flags(row) == [
        "top1_match_rate_below_0.95",
        "average_kl_above_0.01",
        "logits_relative_l2_above_0.05",
        "average_top5_overlap_below_4",
    ]


def test_markdown_contains_required_caveats() -> None:
    module = _load_module()
    summary = module.summarize_rows([
        _row(policy="oracle_topk"),
        _row(policy="query_key_block_score"),
    ])

    markdown = module.render_markdown(
        config={"model": "gpt2"},
        summary=summary,
    )

    assert "outside vLLM" in markdown
    assert "No vLLM integration" in markdown
    assert "one layer" in markdown
    assert "No active routing" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No latency improvement" in markdown
    assert "Full generation quality is not measured" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_logit_sensitivity_sweep.py"
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
        "--layers",
        "--budgets",
        "--block-sizes",
        "--policies",
        "--sketch-dims",
        "--block-score-reduction",
        "--max-length",
        "--dtype",
        "--device",
        "--seed",
        "--output-dir",
        "--dry-run",
        "--continue-on-error",
    ):
        assert flag in process.stdout
