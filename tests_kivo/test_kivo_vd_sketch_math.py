# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "vllm" / "v1" / "core" / "kivo_vd_sketch_math.py"
    spec = importlib.util.spec_from_file_location("kivo_vd_sketch_math", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_projection_shape() -> None:
    m = _load_module()
    proj = m.make_random_projection(input_dim=8, sketch_dim=3, seed=1)
    assert proj.shape == (8, 3)


def test_count_sketch_shape() -> None:
    m = _load_module()
    spec = m.make_count_sketch(input_dim=10, sketch_dim=4, seed=2)
    assert spec.bucket_index.shape == (10,)
    assert spec.bucket_sign.shape == (10,)


def test_exact_score_shape() -> None:
    m = _load_module()
    q = np.array([1.0, 2.0, 3.0])
    k = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scores = m.compute_exact_scores(q, k)
    assert scores.shape == (2,)


def test_topk_recall_known_case() -> None:
    m = _load_module()
    exact = np.array([1, 2, 3])
    approx = np.array([2, 3, 9])
    assert m.topk_recall(exact, approx) == (2 / 3)


def test_block_scoring_correctness() -> None:
    m = _load_module()
    scores = np.array([1.0, 5.0, 2.0, 6.0])
    by_block_max = m.block_scores_from_token_scores(scores, block_size=2, mode="max")
    by_block_mean = m.block_scores_from_token_scores(scores, block_size=2, mode="mean")
    assert np.allclose(by_block_max, np.array([5.0, 6.0]))
    assert np.allclose(by_block_mean, np.array([3.0, 4.0]))


def test_cli_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_offline_sketch_eval.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--num-tokens",
            "64",
            "--input-dim",
            "32",
            "--sketch-dim",
            "8",
            "--topk",
            "8",
            "--block-size",
            "8",
            "--seed",
            "7",
            "--sketch-type",
            "random_projection",
            "--mode",
            "gaussian",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout.strip()
    assert "token_topk_recall" in out
    assert "block_topk_recall" in out


def test_generator_shapes_all_modes() -> None:
    m = _load_module()
    modes = ["gaussian", "clustered", "smooth_sequence", "needle_blocks", "mixed"]
    for mode in modes:
        keys, query = m.generate_synthetic_keys_and_query(
            num_tokens=64,
            input_dim=32,
            seed=11,
            mode=mode,
            block_size=8,
        )
        assert keys.shape == (64, 32)
        assert query.shape == (32,)


def test_generator_deterministic_same_seed() -> None:
    m = _load_module()
    k1, q1 = m.generate_synthetic_keys_and_query(
        num_tokens=32, input_dim=16, seed=5, mode="mixed", block_size=8
    )
    k2, q2 = m.generate_synthetic_keys_and_query(
        num_tokens=32, input_dim=16, seed=5, mode="mixed", block_size=8
    )
    assert np.allclose(k1, k2)
    assert np.allclose(q1, q2)


def test_needle_blocks_has_high_scoring_block() -> None:
    m = _load_module()
    num_tokens = 128
    block_size = 16
    keys, query = m.generate_synthetic_keys_and_query(
        num_tokens=num_tokens,
        input_dim=64,
        seed=19,
        mode="needle_blocks",
        needle_strength=4.0,
        num_needle_blocks=2,
        block_size=block_size,
    )
    scores = m.compute_exact_scores(query, keys)
    block_scores = m.block_scores_from_token_scores(
        scores, block_size=block_size, mode="max"
    )
    top_block = int(m.topk_indices(block_scores, 1)[0])
    median_block = float(np.median(block_scores))
    assert block_scores[top_block] > median_block


def test_recall_at_budget_simple_case() -> None:
    m = _load_module()
    exact_top = np.array([10, 11, 12, 13])
    approx_ranked = np.array([99, 10, 50, 11, 42, 12, 13, 7])
    assert m.recall_at_budget(exact_top, approx_ranked, 2) == 0.25
    assert m.recall_at_budget(exact_top, approx_ranked, 4) == 0.5
    by_budgets = m.recall_at_budgets(exact_top, approx_ranked, [2, 4, 8])
    assert by_budgets[2] == 0.25
    assert by_budgets[4] == 0.5
    assert by_budgets[8] == 1.0


def test_mrr_simple_case() -> None:
    m = _load_module()
    exact_top = np.array([5, 6])
    approx_ranked = np.array([7, 6, 8, 5])
    # rr: 6 at rank 2 -> 1/2, 5 at rank 4 -> 1/4; mean = 3/8
    assert np.isclose(m.mean_reciprocal_rank(exact_top, approx_ranked), 0.375)


def test_pearson_identical_arrays_high() -> None:
    m = _load_module()
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(m.pearson_correlation(a, b), 1.0)
