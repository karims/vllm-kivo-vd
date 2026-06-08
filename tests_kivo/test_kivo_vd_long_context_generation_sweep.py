# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


class FakeTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def __call__(self, text: str, **_kwargs):
        ids = []
        for token in text.split():
            if token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            ids.append(self.token_to_id[token])
        return {"input_ids": ids}

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.id_to_token[token_id] for token_id in token_ids)


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_long_context_adaptive_generation_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_long_context_adaptive_generation_sweep",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    policy: str,
    target: int = 768,
    exact: float = 1.0,
    token_match: float = 1.0,
    edit_distance: float = 0.0,
    kl: float = 0.001,
    top1: float = 1.0,
    selected_ratio: float = 0.4,
) -> dict:
    return {
        "status": "succeeded",
        "policy": policy,
        "layer_budget_map": "0:12,5:8,8:8,11:12",
        "max_new_tokens": 32,
        "target_token_length": target,
        "actual_prompt_token_lengths": [target, target],
        "num_prompts": 2,
        "exact_sequence_match_rate": exact,
        "average_token_match_rate": token_match,
        "average_prefix_match_length": 32.0,
        "average_normalized_edit_distance": edit_distance,
        "average_per_step_kl_divergence": kl,
        "average_per_step_top1_match_rate": top1,
        "average_selected_block_ratio_across_patched_layers": selected_ratio,
        "estimated_active_block_reduction_ratio": 1.0 - selected_ratio,
        "failure_flags": [],
        "warnings": [],
    }


def test_parses_target_token_lengths() -> None:
    module = _load_module()

    assert module.parse_target_token_lengths("768,896") == [768, 896]

    with pytest.raises(ValueError, match="positive"):
        module.parse_target_token_lengths("768,0")


def test_synthetic_prompt_generation_is_deterministic() -> None:
    module = _load_module()
    tokenizer = FakeTokenizer()

    first = module.generate_synthetic_prompts(
        tokenizer=tokenizer,
        target_token_length=80,
        num_prompts=3,
        max_prompt_tokens=96,
        seed=7,
    )
    second = module.generate_synthetic_prompts(
        tokenizer=tokenizer,
        target_token_length=80,
        num_prompts=3,
        max_prompt_tokens=96,
        seed=7,
    )

    assert first == second
    assert len({row["prompt_type"] for row in first}) == 3


def test_generated_prompt_lengths_stay_under_limit() -> None:
    module = _load_module()
    tokenizer = FakeTokenizer()

    prompts = module.generate_synthetic_prompts(
        tokenizer=tokenizer,
        target_token_length=90,
        num_prompts=2,
        max_prompt_tokens=64,
        seed=0,
    )

    assert [row["actual_prompt_token_length"] for row in prompts] == [64, 64]
    assert all(
        len(tokenizer(row["prompt"])["input_ids"]) <= 64 for row in prompts
    )


def test_dry_run_creates_planned_rows_without_model_download(
    tmp_path: Path,
) -> None:
    module = _load_module()
    args = module._parse_args([
        "--dry-run",
        "--target-token-lengths",
        "64,80",
        "--max-new-tokens-values",
        "16",
        "--max-length",
        "128",
        "--num-prompts-per-length",
        "2",
        "--output-dir",
        str(tmp_path),
    ])

    result = module.run_sweep(args)

    assert result["summary"]["counts"] == {
        "total": 4,
        "succeeded": 0,
        "failed": 0,
        "planned": 4,
    }
    rows = [
        json.loads(line)
        for line in Path(result["rows_path"]).read_text().splitlines()
    ]
    assert {row["target_token_length"] for row in rows} == {64, 80}
    assert result["prompts_json"] is None


def test_summary_aggregation_and_selected_reduction() -> None:
    module = _load_module()
    rows = [
        _row(policy="query_key_block_score", selected_ratio=0.4),
        _row(policy="oracle_topk", selected_ratio=0.4),
    ]

    summary = module.build_summary(rows, config={"model": "gpt2"})

    assert summary["counts"]["succeeded"] == 2
    assert summary["readiness"]["phase11_6_ready"] is True
    assert summary["readiness"]["phase12_ready"] is False
    query = next(
        row
        for row in summary["selected_ratio_reduction"]
        if row["policy"] == "query_key_block_score"
    )
    assert query["estimated_active_block_reduction_ratio"] == pytest.approx(
        0.6
    )


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
        _row(policy="oracle_topk", kl=0.005),
    ]

    gap = module.calculate_oracle_gaps(rows)[0]

    assert gap["target_token_length"] == 768
    assert gap["query_minus_oracle_kl"] == pytest.approx(0.015)
    assert gap["oracle_minus_query_exact_match"] == pytest.approx(0.2)
    assert gap["oracle_minus_query_token_match"] == pytest.approx(0.1)
    assert gap["query_minus_oracle_edit_distance"] == pytest.approx(0.1)


def test_best_deployable_excludes_oracle_and_prefers_reduction() -> None:
    module = _load_module()
    rows = [
        _row(policy="oracle_topk", selected_ratio=0.1),
        _row(policy="query_key_block_score", selected_ratio=0.5),
        _row(policy="recent", selected_ratio=0.4),
    ]

    best = module.best_deployable_config(rows)

    assert best is not None
    assert best["policy"] == "recent"


def test_failure_flags_cover_quality_ratio_and_length() -> None:
    module = _load_module()
    row = _row(
        policy="query_key_block_score",
        exact=0.8,
        token_match=0.98,
        edit_distance=0.1,
        kl=0.02,
        top1=0.9,
        selected_ratio=0.9,
    )
    row["actual_prompt_token_lengths"] = [700]

    assert set(module.failure_flags(row)) == {
        "exact_sequence_match_below_1",
        "token_match_below_0.99",
        "normalized_edit_distance_above_0",
        "average_kl_above_0.01",
        "per_step_top1_below_1",
        "selected_ratio_above_0.85",
        "actual_prompt_length_too_short",
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
    assert "Prompts are synthetic unless a file is provided" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No latency claim" in markdown
    assert "generation-quality probe" in markdown
    assert "GPT-2's context limit applies" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_long_context_adaptive_generation_sweep.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--prompt-source",
        "--prompts-file",
        "--target-token-lengths",
        "--num-prompts-per-length",
        "--layer-budget-maps",
        "--policies",
        "--max-new-tokens-values",
        "--max-length",
        "--teacher-forced-context",
        "--output-dir",
        "--dry-run",
        "--continue-on-error",
    ):
        assert flag in process.stdout
