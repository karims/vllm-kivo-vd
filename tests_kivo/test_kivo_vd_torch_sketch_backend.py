# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.core.kivo_vd_torch_sketch_backend import (  # noqa: E402
    TorchCountSketchBackend,
    TorchRandomProjectionBackend,
    TorchSRHTBackend,
    make_torch_sketch_backend,
)


def test_torch_backend_factory_works() -> None:
    count = make_torch_sketch_backend(
        "count_sketch", 8, 4, 1, "cpu", torch.float32
    )
    rp = make_torch_sketch_backend(
        "random_projection", 8, 4, 1, "cpu", torch.float32
    )
    srht = make_torch_sketch_backend("srht", 10, 4, 1, "cpu", torch.float32)
    assert isinstance(count, TorchCountSketchBackend)
    assert isinstance(rp, TorchRandomProjectionBackend)
    assert isinstance(srht, TorchSRHTBackend)


def test_torch_count_sketch_deterministic_same_seed() -> None:
    a = TorchCountSketchBackend(8, 4, 7, "cpu", torch.float32)
    b = TorchCountSketchBackend(8, 4, 7, "cpu", torch.float32)
    assert torch.equal(a.bucket_index, b.bucket_index)
    assert torch.equal(a.bucket_sign, b.bucket_sign)


def test_torch_random_projection_deterministic_same_seed() -> None:
    a = TorchRandomProjectionBackend(8, 4, 9, "cpu", torch.float32)
    b = TorchRandomProjectionBackend(8, 4, 9, "cpu", torch.float32)
    assert torch.allclose(a.projection, b.projection)


def test_torch_srht_deterministic_same_seed() -> None:
    a = TorchSRHTBackend(10, 4, 9, "cpu", torch.float32)
    b = TorchSRHTBackend(10, 4, 9, "cpu", torch.float32)
    x = torch.arange(10, dtype=torch.float32)
    assert a.padded_dim == 16
    assert torch.equal(a.signs, b.signs)
    assert torch.equal(a.sampled_indices, b.sampled_indices)
    assert torch.allclose(a.sketch_query(x), b.sketch_query(x))


def test_torch_backend_output_shapes() -> None:
    backend = TorchCountSketchBackend(8, 4, 3, "cpu", torch.float32)
    keys = torch.randn(16, 8)
    query = torch.randn(8)
    key_sketches = backend.sketch_keys(keys)
    query_sketch = backend.sketch_query(query)
    block_sketches = backend.block_sketches_from_key_sketches(
        key_sketches, block_size=4
    )
    scores = backend.score_blocks(query_sketch, block_sketches)
    ranks = backend.rank_blocks(scores)

    assert key_sketches.shape == (16, 4)
    assert query_sketch.shape == (4,)
    assert block_sketches.shape == (4, 4)
    assert scores.shape == (4,)
    assert ranks.shape == (4,)


def test_torch_benchmark_script_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "benchmark_torch_sketch_backend.py"
    output = tmp_path / "bench.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sketch-types",
            "srht",
            "--sketch-dims",
            "8",
            "--num-tokens",
            "64",
            "--head-dim",
            "16",
            "--block-size",
            "8",
            "--num-queries",
            "2",
            "--topk-blocks",
            "3",
            "--block-score-mode",
            "mean",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(proc.stdout)
    assert summary["num_rows"] == 1
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["sketch_type"] == "srht"
    assert row["sketch_dim"] == 8
    assert row["topk_blocks"] == 3
    assert row["block_score_mode"] == "mean"
    assert "key_sketch_build_ms" in row
    assert "block_aggregation_ms" in row
    assert "query_sketch_ms" in row
    assert "block_scoring_ms" in row
    assert "ranking_ms" in row
    assert "total_time_ms" in row
    assert "sketch_memory_ratio" in row
    assert "sketch_memory_bytes" in row
