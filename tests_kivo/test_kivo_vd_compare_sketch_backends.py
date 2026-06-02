# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_compare_sketch_backends_summarizes_tiny_jsonl(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "compare_sketch_backends.py"
    input_path = tmp_path / "hf.jsonl"
    output_path = tmp_path / "summary.json"
    _write_jsonl(
        input_path,
        [
            {
                "sketch_type": "count_sketch",
                "sketch_dim": 64,
                "effective_sketch_dim": 64,
                "sketch_compression_ratio": 1.0,
                "is_full_dimensional_sketch": True,
                "block_topk_recall": 0.8,
                "block_recall_at_2x_budget": 0.9,
                "block_recall_at_4x_budget": 1.0,
                "block_score_correlation": 0.91,
            },
            {
                "sketch_type": "srht",
                "sketch_dim": 64,
                "effective_sketch_dim": 32,
                "sketch_compression_ratio": 0.5,
                "is_full_dimensional_sketch": False,
                "block_topk_recall": 0.7,
                "block_recall_at_2x_budget": 0.85,
                "block_recall_at_4x_budget": 0.95,
                "block_score_correlation": 0.88,
            },
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert output_path.exists()
    summary = payload["summary"]
    assert {row["sketch_type"] for row in summary} == {"count_sketch", "srht"}
    srht = next(row for row in summary if row["sketch_type"] == "srht")
    assert srht["effective_sketch_dim"] == 32
    assert srht["sketch_compression_ratio"] == 0.5
    assert srht["is_full_dimensional_sketch"] is False
    assert srht["avg_block_topk_recall"] == 0.7
    assert srht["avg_block_recall_at_2x_budget"] == 0.85
