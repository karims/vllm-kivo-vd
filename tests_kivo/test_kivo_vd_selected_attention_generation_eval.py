# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_selected_attention_generation_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_selected_attention_generation_eval",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_match_length() -> None:
    module = _load_module()

    assert module.prefix_match_length([1, 2, 3], [1, 2, 4]) == 2
    assert module.prefix_match_length([1, 2], [1, 2, 3]) == 2
    assert module.prefix_match_length([], [1]) == 0


def test_normalized_edit_distance() -> None:
    module = _load_module()

    assert module.normalized_edit_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert module.normalized_edit_distance([1, 2, 3], [1, 4, 3]) == (
        pytest.approx(1 / 3)
    )
    assert module.normalized_edit_distance([], []) == 0.0


def test_token_sequence_comparison_metrics() -> None:
    module = _load_module()

    metrics = module.compare_token_sequences(
        [1, 2, 3, 4],
        [1, 2, 9, 4],
    )

    assert metrics["exact_token_sequence_match"] is False
    assert metrics["prefix_match_length"] == 2
    assert metrics["token_match_rate"] == pytest.approx(0.75)
    assert metrics["first_mismatch_index"] == 2
    assert metrics["normalized_edit_distance"] == pytest.approx(0.25)


def test_aggregate_generation_metrics() -> None:
    module = _load_module()
    rows = [
        {
            "exact_token_sequence_match": True,
            "token_match_rate": 1.0,
            "prefix_match_length": 4,
            "normalized_edit_distance": 0.0,
            "average_per_step_kl_divergence": 0.001,
            "average_per_step_top1_match": 1.0,
            "average_selected_block_ratio": 0.5,
        },
        {
            "exact_token_sequence_match": False,
            "token_match_rate": 0.5,
            "prefix_match_length": 2,
            "normalized_edit_distance": 0.5,
            "average_per_step_kl_divergence": 0.01,
            "average_per_step_top1_match": 0.5,
            "average_selected_block_ratio": 0.4,
        },
    ]

    aggregate = module.aggregate_rows(rows)

    assert aggregate["num_prompts"] == 2
    assert aggregate["exact_sequence_match_rate"] == 0.5
    assert aggregate["average_token_match_rate"] == 0.75
    assert aggregate["average_prefix_match_length"] == 3.0
    assert aggregate["average_normalized_edit_distance"] == 0.25
    assert aggregate["average_per_step_kl_divergence"] == pytest.approx(
        0.0055
    )


def test_report_schema_and_caveats() -> None:
    module = _load_module()
    row = {
        "exact_token_sequence_match": True,
        "token_match_rate": 1.0,
        "prefix_match_length": 2,
        "normalized_edit_distance": 0.0,
        "average_per_step_kl_divergence": 0.001,
        "average_per_step_top1_match": 1.0,
        "average_selected_block_ratio": 0.5,
    }

    report = module.build_report(config={"model": "gpt2"}, rows=[row])

    assert set(report) == {"config", "aggregate", "per_prompt", "caveats"}
    assert report["per_prompt"] == [row]
    assert report["caveats"]["outside_vllm"] is True
    assert report["caveats"]["no_vllm_integration"] is True
    assert report["caveats"]["single_layer_patch_only"] is True
    assert report["caveats"]["greedy_generation_only"] is True


def test_tiny_random_gpt2_generation_smoke() -> None:
    transformers = pytest.importorskip("transformers")
    module = _load_module()
    phase11 = module._load_phase11()
    helpers = phase11._load_selected_attention_helpers()
    config = transformers.GPT2Config(
        vocab_size=97,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    input_ids = torch.randint(0, 97, (1, 12))
    args = argparse.Namespace(
        max_new_tokens=3,
        max_length=32,
        teacher_forced_context=False,
        layer_idx=0,
        block_size=4,
        candidate_budget_blocks=4,
        selection_policy="query_key_block_score",
        sketch_dim=8,
        block_score_reduction="max",
        seed=0,
    )

    result = module.generate_sequences(
        model=model,
        input_ids=input_ids,
        args=args,
        phase11=phase11,
        helpers=helpers,
    )

    assert len(result["baseline_generated_token_ids"]) == 3
    assert len(result["patched_generated_token_ids"]) == 3
    assert len(result["per_step_kl_divergence"]) == 3
    assert len(result["per_step_selected_block_ratio"]) == 3
    assert result["baseline_generated_token_ids"] == (
        result["patched_generated_token_ids"]
    )


def test_markdown_contains_required_caveats() -> None:
    module = _load_module()
    report = {
        "config": {"model": "gpt2"},
        "aggregate": {"num_prompts": 1},
        "per_prompt": [{
            "prompt_index": 0,
            "prompt_token_length": 12,
            "num_generated_tokens": 2,
            "exact_token_sequence_match": True,
            "prefix_match_length": 2,
            "token_match_rate": 1.0,
            "normalized_edit_distance": 0.0,
            "average_per_step_kl_divergence": 0.0,
            "average_per_step_top1_match": 1.0,
            "average_selected_block_ratio": 0.5,
            "baseline_generated_text": " hello",
            "patched_generated_text": " hello",
            "final_generated_text_comparison_note": "exact token sequence match",
        }],
        "caveats": {},
    }

    markdown = module.render_markdown(report)

    assert "outside vLLM" in markdown
    assert "No vLLM integration" in markdown
    assert "one layer" in markdown
    assert "greedy decoding only" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No latency claim" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_selected_attention_generation_eval.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--prompt",
        "--prompts-file",
        "--layer-idx",
        "--block-size",
        "--candidate-budget-blocks",
        "--selection-policy",
        "--sketch-dim",
        "--block-score-reduction",
        "--max-length",
        "--max-new-tokens",
        "--teacher-forced-context",
        "--dtype",
        "--device",
        "--seed",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
