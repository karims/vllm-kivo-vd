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
        / "run_multilayer_selected_attention_generation_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_multilayer_selected_attention_generation_eval",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parses_layers_and_single_budget() -> None:
    module = _load_module()

    result = module.resolve_layer_budget_map(
        layers_value="5,8,11",
        budgets_value="8",
        map_value=None,
    )

    assert result == {5: 8, 8: 8, 11: 8}


def test_parses_layer_budget_map() -> None:
    module = _load_module()

    result = module.resolve_layer_budget_map(
        layers_value="5,8",
        budgets_value="8",
        map_value="0:12,5:8,8:8,11:8",
    )

    assert result == {0: 12, 5: 8, 8: 8, 11: 8}


def test_matching_budgets_zip_to_layers() -> None:
    module = _load_module()

    result = module.resolve_layer_budget_map(
        layers_value="0,5,8",
        budgets_value="12,8,16",
        map_value=None,
    )

    assert result == {0: 12, 5: 8, 8: 16}


def test_mismatched_layers_and_budgets_error() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="match the number of layers"):
        module.resolve_layer_budget_map(
            layers_value="0,5,8",
            budgets_value="8,16",
            map_value=None,
        )


def test_sequence_metrics_are_reused_correctly() -> None:
    module = _load_module()
    generation_helpers = module._load_phase11_generation()

    metrics = generation_helpers.compare_token_sequences(
        [1, 2, 3, 4],
        [1, 2, 9, 4],
    )

    assert metrics["prefix_match_length"] == 2
    assert metrics["token_match_rate"] == pytest.approx(0.75)
    assert metrics["normalized_edit_distance"] == pytest.approx(0.25)


def test_report_schema_and_caveats() -> None:
    module = _load_module()
    layer_map = {5: 8, 8: 8}
    row = {
        "exact_token_sequence_match": True,
        "token_match_rate": 1.0,
        "prefix_match_length": 2,
        "normalized_edit_distance": 0.0,
        "average_per_step_kl_divergence": 0.001,
        "average_per_step_top1_match": 1.0,
        "per_layer_selected_block_summary": {
            "5": {
                "budget": 8,
                "average_selected_block_ratio": 0.5,
                "average_selected_blocks": 8.0,
            },
            "8": {
                "budget": 8,
                "average_selected_block_ratio": 0.5,
                "average_selected_blocks": 8.0,
            },
        },
    }

    report = module.build_report(
        config={"model": "gpt2"},
        layer_budget_map=layer_map,
        rows=[row],
    )

    assert set(report) == {
        "config",
        "layer_budget_map",
        "aggregate",
        "per_prompt",
        "caveats",
    }
    assert report["layer_budget_map"] == {"5": 8, "8": 8}
    assert report["caveats"]["outside_vllm"] is True
    assert report["caveats"]["no_vllm_integration"] is True
    assert report["caveats"]["multilayer_patch"] is True


def test_tiny_random_gpt2_two_layer_smoke() -> None:
    transformers = pytest.importorskip("transformers")
    module = _load_module()
    phase11 = module._load_phase11()
    helpers = phase11._load_selected_attention_helpers()
    config = transformers.GPT2Config(
        vocab_size=97,
        n_positions=32,
        n_embd=16,
        n_layer=3,
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
        block_size=4,
        selection_policy="query_key_block_score",
        sketch_dim=8,
        block_score_reduction="max",
        seed=0,
    )

    result = module.generate_sequences(
        model=model,
        input_ids=input_ids,
        args=args,
        layer_budget_map={0: 4, 1: 4},
        phase11=phase11,
        helpers=helpers,
    )

    assert result["baseline_generated_token_ids"] == (
        result["patched_generated_token_ids"]
    )
    assert set(result["per_layer_selected_block_summary"]) == {"0", "1"}
    assert all(
        values["average_selected_block_ratio"] == 1.0
        for values in result["per_layer_selected_block_summary"].values()
    )


def test_markdown_contains_required_sections_and_caveats() -> None:
    module = _load_module()
    report = {
        "config": {"model": "gpt2"},
        "layer_budget_map": {"5": 8, "8": 8},
        "aggregate": {
            "num_prompts": 1,
            "exact_sequence_match_rate": 1.0,
            "per_layer_selected_block_summary": {
                "5": {
                    "budget": 8,
                    "average_selected_blocks": 8.0,
                    "average_selected_block_ratio": 0.5,
                },
                "8": {
                    "budget": 8,
                    "average_selected_blocks": 8.0,
                    "average_selected_block_ratio": 0.5,
                },
            },
        },
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
            "baseline_generated_text": " hello",
            "patched_generated_text": " hello",
            "final_generated_text_comparison_note": "exact token sequence match",
        }],
        "caveats": {},
    }

    markdown = module.render_markdown(report)

    assert "Layer-Budget Map" in markdown
    assert "outside vLLM" in markdown
    assert "No vLLM integration" in markdown
    assert "Multiple layers are patched" in markdown
    assert "greedy decoding only" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_multilayer_selected_attention_generation_eval.py"
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
        "--layers",
        "--budgets",
        "--layer-budget-map",
        "--block-size",
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
