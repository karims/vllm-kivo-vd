# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "run_sk3_2_attention_mass_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_sk3_2_attention_mass_eval",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_true_attention_mass_per_block_sums_to_one() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    keys = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    attn = module.compute_true_attention_weights(query, keys)
    mass = module.aggregate_block_values(attn, block_size=2)

    assert pytest.approx(float(attn.sum().item()), rel=1e-6) == 1.0
    assert pytest.approx(float(mass.sum().item()), rel=1e-6) == 1.0


def test_block_aggregation_handles_uneven_final_block() -> None:
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

    mass = module.aggregate_block_values(values, block_size=2)

    assert torch.allclose(
        mass,
        torch.tensor([0.3, 0.7, 0.5], dtype=torch.float32),
    )


def test_mean_key_score_returns_one_score_per_block() -> None:
    query = torch.tensor([1.0, 2.0], dtype=torch.float32)
    keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    scores = module.mean_key_block_scores(query, keys, block_size=2)

    assert tuple(scores.shape) == (3,)


def test_countsketch_scoring_is_deterministic_with_seed() -> None:
    query = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    keys = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    a = module.sketch_block_scores(
        query,
        keys,
        block_size=2,
        backend="countsketch",
        sketch_dim=3,
        seed=123,
    )
    b = module.sketch_block_scores(
        query,
        keys,
        block_size=2,
        backend="countsketch",
        sketch_dim=3,
        seed=123,
    )

    assert torch.allclose(a, b)


def test_topk_mass_recovered_is_computed_correctly() -> None:
    true_mass = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)

    recovered = module.attention_mass_recovered(true_mass, [2, 0])

    assert pytest.approx(recovered, rel=1e-6) == 0.7


def test_perfect_ranking_gives_perfect_topk_overlap() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    keys = torch.tensor(
        [
            [5.0, 0.0],
            [4.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    result = module.evaluate_backend_scores(
        backend="mean_key",
        query=query,
        keys=keys,
        block_size=2,
        sketch_dim=2,
        seed=1,
        topks=[1, 2],
    )

    assert result["topk_metrics"]["1"]["topk_overlap_with_true_topk"] == 1
    assert result["topk_metrics"]["2"]["topk_overlap_with_true_topk"] == 2


def test_output_schema_can_be_built_from_synthetic_tensors() -> None:
    backend_results = {
        "mean_key": {
            "score_by_block": [0.3, 0.2],
            "true_mass_by_block": [0.6, 0.4],
            "top_blocks_by_true_mass": [0, 1],
            "top_blocks_by_score": [0, 1],
            "topk_metrics": {
                "1": {
                    "attention_mass_recovered_by_score_topk": 0.6,
                    "topk_overlap_with_true_topk": 1,
                    "true_topk_mass": 0.6,
                }
            },
            "rank_correlation_spearman": 1.0,
            "warnings": [],
        }
    }

    payload = module.build_output_payload(
        model="synthetic",
        prompt_kind="synthetic_long",
        token_count=12,
        block_size=4,
        num_blocks=3,
        layer=1,
        head=0,
        query_position=11,
        backend_results=backend_results,
    )

    assert payload["model"] == "synthetic"
    assert payload["prompt_kind"] == "synthetic_long"
    assert payload["backend_results"]["mean_key"]["top_blocks_by_score"] == [0, 1]


def test_invalid_backend_fails_safely() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported backend"):
        module.evaluate_backend_scores(
            backend="not_real",
            query=query,
            keys=keys,
            block_size=1,
            sketch_dim=2,
            seed=1,
            topks=[1],
        )
