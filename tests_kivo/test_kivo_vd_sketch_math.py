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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout.strip()
    assert "token_topk_recall" in out
    assert "block_topk_recall" in out
