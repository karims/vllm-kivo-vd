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


def test_structured_param_sweep_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_structured_sketch_param_sweep.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    out = proc.stdout
    assert "--sketch-types" in out
    assert "--alphas" in out
    assert "--coordinate-strategies" in out
    assert "bidiagonal_sign_subsample" in out
    assert "tridiagonal_sign" in out


def test_structured_param_summary_groups_tiny_jsonl(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "summarize_structured_sketch_param_sweep.py"
    )
    input_path = tmp_path / "sweep.jsonl"
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"
    _write_jsonl(
        input_path,
        [
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "num_query_heads": 14,
                "num_key_value_heads": 2,
                "selected_query_head": 0,
                "selected_kv_head": 0,
                "head_dim": 128,
                "effective_sketch_dim": 32,
                "sketch_compression_ratio": 0.25,
                "is_full_dimensional_sketch": False,
                "sketch_type": "bidiagonal_sign_subsample",
                "sketch_dim": 32,
                "structured_alpha": 0.5,
                "structured_coordinate_strategy": "uniform",
                "block_topk_recall": 0.7,
                "block_recall_at_2x_budget": 0.9,
                "block_recall_at_4x_budget": 1.0,
                "block_score_correlation": 0.8,
            },
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "num_query_heads": 14,
                "num_key_value_heads": 2,
                "selected_query_head": 1,
                "selected_kv_head": 0,
                "head_dim": 128,
                "effective_sketch_dim": 32,
                "sketch_compression_ratio": 0.25,
                "is_full_dimensional_sketch": False,
                "sketch_type": "bidiagonal_sign_subsample",
                "sketch_dim": 32,
                "structured_alpha": 0.5,
                "structured_coordinate_strategy": "uniform",
                "block_topk_recall": 0.8,
                "block_recall_at_2x_budget": 1.0,
                "block_recall_at_4x_budget": 1.0,
                "block_score_correlation": 0.9,
            },
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "num_query_heads": 14,
                "num_key_value_heads": 2,
                "selected_query_head": 0,
                "selected_kv_head": 0,
                "head_dim": 128,
                "effective_sketch_dim": 32,
                "sketch_compression_ratio": 0.25,
                "is_full_dimensional_sketch": False,
                "sketch_type": "tridiagonal_sign",
                "sketch_dim": 32,
                "structured_alpha": 0.25,
                "structured_coordinate_strategy": "stride",
                "block_topk_recall": 0.6,
                "block_recall_at_2x_budget": 0.85,
                "block_recall_at_4x_budget": 0.95,
                "block_score_correlation": 0.7,
            },
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output-json",
            str(json_output),
            "--output-md",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["num_input_rows"] == 3
    assert payload["num_summary_rows"] == 2
    assert json_output.exists()
    assert markdown_output.exists()

    summary = json.loads(json_output.read_text(encoding="utf-8"))["summary"]
    best = summary[0]
    assert best["sketch_type"] == "bidiagonal_sign_subsample"
    assert best["structured_alpha"] == 0.5
    assert best["structured_coordinate_strategy"] == "uniform"
    assert best["model_name"] == "Qwen/Qwen2.5-0.5B"
    assert best["extraction_mode"] == "separate_qk_proj"
    assert best["qk_space"] == "pre_rope_projection"
    assert best["num_query_heads"] == 14
    assert best["num_key_value_heads"] == 2
    assert best["selected_query_head"] == [0, 1]
    assert best["selected_kv_head"] == 0
    assert best["head_dim"] == 128
    assert best["effective_sketch_dim"] == 32
    assert best["sketch_compression_ratio"] == 0.25
    assert best["is_full_dimensional_sketch"] is False
    assert best["count"] == 2
    assert best["avg_block_topk_recall"] == 0.75
    assert best["avg_block_recall_at_2x_budget"] == 0.95
    assert "offline retrieval summary only" in markdown_output.read_text(
        encoding="utf-8"
    )
