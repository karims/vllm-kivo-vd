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
        / "run_real_qkv_policy_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_real_qkv_policy_sweep",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    policy: str,
    layer: int,
    budget: int,
    cosine: float,
    min_cosine: float,
    relative_l2: float,
    max_relative_l2: float,
    mass: float,
) -> dict:
    return {
        "policy": policy,
        "layer_index": layer,
        "candidate_budget_blocks": budget,
        "block_size": 16,
        "status": "succeeded",
        "num_prompts": 2,
        "average_cosine_similarity": cosine,
        "min_cosine_similarity": min_cosine,
        "average_relative_l2_error": relative_l2,
        "max_relative_l2_error": max_relative_l2,
        "average_attention_mass_captured": mass,
        "average_selected_block_ratio": 0.25,
        "average_selected_token_ratio": 0.25,
        "failure_flags": [],
    }


def test_parses_comma_separated_sweep_values() -> None:
    module = _load_module()

    assert module._parse_int_csv("0,5,11") == [0, 5, 11]
    assert module._parse_int_csv("4,8,16") == [4, 8, 16]
    assert module.parse_policies(
        "recent,random,oracle_topk"
    ) == ["recent", "random", "oracle_topk"]


def test_builds_all_sweep_combinations() -> None:
    module = _load_module()

    combinations = module.build_combinations(
        layers=[0, 5],
        budgets=[4, 8],
        block_sizes=[8, 16],
        policies=["recent", "oracle_topk"],
    )

    assert len(combinations) == 16
    assert {
        (
            row["layer_index"],
            row["candidate_budget_blocks"],
            row["block_size"],
            row["policy"],
        )
        for row in combinations
    } == {
        (layer, budget, block_size, policy)
        for layer in (0, 5)
        for budget in (4, 8)
        for block_size in (8, 16)
        for policy in ("recent", "oracle_topk")
    }


def test_failure_flags_use_research_thresholds() -> None:
    module = _load_module()
    row = _row(
        policy="recent",
        layer=5,
        budget=4,
        cosine=0.80,
        min_cosine=0.70,
        relative_l2=0.50,
        max_relative_l2=0.80,
        mass=0.60,
    )

    flags = module.failure_flags(row)

    assert flags == [
        "average_cosine_below_0.95",
        "min_cosine_below_0.90",
        "average_relative_l2_above_0.25",
        "max_relative_l2_above_0.50",
    ]


def test_summary_and_oracle_gap_calculation() -> None:
    module = _load_module()
    oracle = _row(
        policy="oracle_topk",
        layer=5,
        budget=4,
        cosine=0.99,
        min_cosine=0.98,
        relative_l2=0.05,
        max_relative_l2=0.10,
        mass=0.95,
    )
    recent = _row(
        policy="recent",
        layer=5,
        budget=4,
        cosine=0.80,
        min_cosine=0.70,
        relative_l2=0.50,
        max_relative_l2=0.80,
        mass=0.60,
    )
    recent["failure_flags"] = module.failure_flags(recent)

    summary = module.summarize_rows([oracle, recent])

    assert summary["num_runs"] == 2
    assert summary["num_succeeded"] == 2
    assert summary["best_by_average_cosine"]["policy"] == "oracle_topk"
    assert summary["worst_by_max_relative_l2"]["policy"] == "recent"
    assert summary["per_policy"][0]["policy"] == "oracle_topk"
    gap = summary["oracle_gaps"][0]
    assert gap["policy"] == "recent"
    assert gap["cosine_gap"] == pytest.approx(0.19)
    assert gap["relative_l2_gap"] == pytest.approx(0.45)
    assert gap["attention_mass_gap"] == pytest.approx(0.35)


def test_dry_run_writes_planned_runs_without_model_download(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_real_qkv_policy_sweep.py"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--layers",
            "0,5",
            "--budgets",
            "4",
            "--block-sizes",
            "16",
            "--policies",
            "recent,oracle_topk",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(process.stdout)
    runs_path = tmp_path / "policy_sweep_runs.jsonl"
    summary_path = tmp_path / "policy_sweep_summary.json"
    markdown_path = tmp_path / "policy_sweep_summary.md"
    rows = [
        json.loads(line)
        for line in runs_path.read_text(encoding="utf-8").splitlines()
    ]

    assert payload["dry_run"] is True
    assert payload["summary"]["num_runs"] == 4
    assert len(rows) == 4
    assert all(row["status"] == "planned" for row in rows)
    assert summary_path.exists()
    assert markdown_path.exists()


def test_markdown_contains_required_caveats() -> None:
    module = _load_module()
    summary = module.summarize_rows([
        _row(
            policy="oracle_topk",
            layer=0,
            budget=4,
            cosine=0.99,
            min_cosine=0.98,
            relative_l2=0.05,
            max_relative_l2=0.10,
            mass=0.95,
        )
    ])

    markdown = module.render_markdown(
        config={"model": "gpt2"},
        summary=summary,
    )

    assert "real GPT-2-style model" in markdown
    assert "outside vLLM" in markdown
    assert "No logits or generation quality is measured" in markdown
    assert "No active routing is implemented" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_real_qkv_policy_sweep.py"
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
        "--max-length",
        "--dtype",
        "--device",
        "--seed",
        "--output-dir",
        "--run-name",
        "--continue-on-error",
        "--dry-run",
    ):
        assert flag in process.stdout
