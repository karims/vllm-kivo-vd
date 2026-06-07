# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
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
        / "run_selected_attention_equivalence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_selected_attention_equivalence",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_attention_output_shape() -> None:
    module = _load_module()
    query = torch.randn(1, 4, 2, 8)
    keys = torch.randn(1, 4, 16, 8)
    values = torch.randn(1, 4, 16, 8)

    output, weights = module.scaled_dot_product_attention(
        query,
        keys,
        values,
    )

    assert output.shape == (1, 4, 2, 8)
    assert weights.shape == (1, 4, 2, 16)


def test_selected_block_gather_and_attention_shapes() -> None:
    module = _load_module()
    query = torch.randn(1, 4, 1, 8)
    keys = torch.randn(1, 2, 32, 8)
    values = torch.randn(1, 2, 32, 8)
    expanded_keys = module.expand_kv_heads(keys, 4)
    expanded_values = module.expand_kv_heads(values, 4)

    selected_keys = module.gather_selected_blocks(
        expanded_keys,
        [1, 3],
        block_size=8,
    )
    selected_values = module.gather_selected_blocks(
        expanded_values,
        [1, 3],
        block_size=8,
    )
    output, _ = module.scaled_dot_product_attention(
        query,
        selected_keys,
        selected_values,
    )

    assert selected_keys.shape == (1, 4, 16, 8)
    assert selected_values.shape == (1, 4, 16, 8)
    assert output.shape == (1, 4, 1, 8)


def test_all_blocks_match_full_attention() -> None:
    module = _load_module()
    query = torch.randn(1, 2, 1, 4)
    keys = torch.randn(1, 2, 12, 4)
    values = torch.randn(1, 2, 12, 4)
    full_output, _ = module.scaled_dot_product_attention(
        query,
        keys,
        values,
    )
    selected_keys = module.gather_selected_blocks(keys, [0, 1, 2], 4)
    selected_values = module.gather_selected_blocks(values, [0, 1, 2], 4)
    selected_output, _ = module.scaled_dot_product_attention(
        query,
        selected_keys,
        selected_values,
    )

    assert torch.allclose(full_output, selected_output)


def test_oracle_captures_at_least_random_attention_mass() -> None:
    module = _load_module()
    block_mass = torch.tensor([0.05, 0.50, 0.10, 0.35])
    oracle = module.select_block_ids(
        policy="oracle_topk",
        num_blocks=4,
        candidate_budget_blocks=2,
        seed=0,
        block_attention_mass=block_mass,
    )
    random = module.select_block_ids(
        policy="random",
        num_blocks=4,
        candidate_budget_blocks=2,
        seed=0,
    )

    oracle_mass = module.captured_attention_mass(block_mass, oracle)
    random_mass = module.captured_attention_mass(block_mass, random)

    assert oracle_mass >= random_mass
    assert oracle == [1, 3]


def test_metric_calculations() -> None:
    module = _load_module()
    full = torch.tensor([[[[1.0, 2.0]]]])
    selected = torch.tensor([[[[1.0, 1.0]]]])

    metrics = module.calculate_metrics(full, selected)

    assert metrics["cosine_similarity"] == pytest.approx(
        3 / (5**0.5 * 2**0.5)
    )
    assert metrics["relative_l2_error"] == pytest.approx(1 / 5**0.5)
    assert metrics["max_absolute_error"] == 1.0
    assert metrics["mean_absolute_error"] == 0.5


def test_cpu_smoke_run_writes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_md = tmp_path / "result.md"

    return_code = module.main([
        "--num-query-heads",
        "4",
        "--num-kv-heads",
        "2",
        "--head-dim",
        "8",
        "--block-size",
        "4",
        "--num-blocks",
        "8",
        "--candidate-budget-blocks",
        "2",
        "--selection-policy",
        "oracle_topk",
        "--device",
        "cpu",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ])

    assert return_code == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["full_output_shape"] == [1, 4, 1, 8]
    assert report["selected_output_shape"] == [1, 4, 1, 8]
    assert report["selected_block_count"] == 2
    assert report["selected_token_count"] == 8
    assert report["caveats"]["synthetic_qkv"] is True


def test_markdown_contains_required_caveats() -> None:
    module = _load_module()
    args = module._parse_args([
        "--num-query-heads",
        "2",
        "--num-kv-heads",
        "2",
        "--head-dim",
        "4",
        "--block-size",
        "2",
        "--num-blocks",
        "4",
        "--candidate-budget-blocks",
        "2",
        "--device",
        "cpu",
    ])
    report = module.run_experiment(args)

    markdown = module.render_markdown(report)

    assert "Q/K/V tensors are synthetic" in markdown
    assert "outside vLLM" in markdown
    assert "No real model quality is measured" in markdown
    assert "No active routing is implemented" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_selected_attention_equivalence.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--num-query-heads",
        "--num-kv-heads",
        "--head-dim",
        "--block-size",
        "--num-blocks",
        "--query-len",
        "--selected-blocks",
        "--selection-policy",
        "--candidate-budget-blocks",
        "--dtype",
        "--device",
        "--seed",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
