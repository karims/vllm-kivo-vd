# SPDX-License-Identifier: Apache-2.0

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
        / "run_real_qkv_selected_attention_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_real_qkv_selected_attention_eval",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpt2_fused_qkv_split_shapes() -> None:
    module = _load_module()
    fused = torch.arange(2 * 5 * 24, dtype=torch.float32).reshape(2, 5, 24)

    query, keys, values = module.split_gpt2_fused_qkv(fused, num_heads=2)

    assert query.shape == (2, 2, 5, 4)
    assert keys.shape == (2, 2, 5, 4)
    assert values.shape == (2, 2, 5, 4)


def test_block_attention_mass_handles_partial_final_block() -> None:
    module = _load_module()
    probabilities = torch.tensor([[[[0.1, 0.2, 0.3, 0.15, 0.25]]]])

    masses = module.block_attention_mass(probabilities, block_size=2)

    assert masses.tolist() == pytest.approx([0.3, 0.45, 0.25])
    assert masses.sum().item() == pytest.approx(1.0)


def test_oracle_selects_highest_mass_blocks() -> None:
    module = _load_module()
    masses = torch.tensor([0.1, 0.5, 0.3, 0.1])

    selected = module.select_block_ids(
        policy="oracle_topk",
        num_blocks=4,
        candidate_budget_blocks=2,
        seed=0,
        masses=masses,
    )

    assert selected == [1, 2]
    assert module.captured_attention_mass(masses, selected) == pytest.approx(
        0.8
    )


def test_selected_attention_output_shape_with_partial_block() -> None:
    module = _load_module()
    query = torch.randn(1, 2, 5, 4)
    keys = torch.randn(1, 2, 5, 4)
    values = torch.randn(1, 2, 5, 4)

    selected_keys = module.gather_selected_blocks(keys, [0, 2], 2)
    selected_values = module.gather_selected_blocks(values, [0, 2], 2)
    output, probabilities = module.last_query_attention(
        query,
        selected_keys,
        selected_values,
    )

    assert selected_keys.shape == (1, 2, 3, 4)
    assert output.shape == (1, 2, 1, 4)
    assert probabilities.shape == (1, 2, 1, 3)


def test_metric_calculations() -> None:
    module = _load_module()
    full = torch.tensor([[[[1.0, 2.0]]]])
    selected = torch.tensor([[[[1.0, 1.0]]]])

    metrics = module.calculate_metrics(full, selected)

    assert metrics["cosine_similarity"] == pytest.approx(
        3 / (5**0.5 * 2**0.5)
    )
    assert metrics["relative_l2_error"] == pytest.approx(1 / 5**0.5)
    assert metrics["mean_absolute_error"] == 0.5
    assert metrics["max_absolute_error"] == 1.0


def test_aggregate_metrics() -> None:
    module = _load_module()
    rows = [
        {
            "cosine_similarity": 0.8,
            "relative_l2_error": 0.4,
            "attention_mass_captured": 0.7,
        },
        {
            "cosine_similarity": 1.0,
            "relative_l2_error": 0.2,
            "attention_mass_captured": 0.9,
        },
    ]

    aggregate = module.aggregate_rows(rows)

    assert aggregate["num_prompts"] == 2
    assert aggregate["average_cosine_similarity"] == pytest.approx(0.9)
    assert aggregate["average_relative_l2_error"] == pytest.approx(0.3)
    assert aggregate["average_attention_mass_captured"] == pytest.approx(0.8)
    assert aggregate["min_cosine_similarity"] == 0.8
    assert aggregate["max_relative_l2_error"] == 0.4


def test_markdown_contains_required_caveats() -> None:
    module = _load_module()
    report = {
        "config": {
            "model": "gpt2",
            "selection_policy": "oracle_topk",
        },
        "aggregate_metrics": {
            "num_prompts": 1,
            "average_cosine_similarity": 0.9,
        },
        "per_prompt_rows": [{
            "prompt_index": 0,
            "token_length": 32,
            "block_count": 2,
            "selected_block_count": 1,
            "selected_block_ratio": 0.5,
            "attention_mass_captured": 0.8,
            "cosine_similarity": 0.9,
            "relative_l2_error": 0.2,
            "mean_absolute_error": 0.1,
            "max_absolute_error": 0.3,
        }],
    }

    markdown = module.render_markdown(report)

    assert "real GPT-2-style model" in markdown
    assert "outside vLLM" in markdown
    assert "No generation quality is measured" in markdown
    assert "No active routing is implemented" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_real_qkv_selected_attention_eval.py"
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
        "--selected-blocks",
        "--max-length",
        "--dtype",
        "--device",
        "--seed",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
