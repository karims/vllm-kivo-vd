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


def test_benchmark_report_generator_writes_markdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "generate_kivo_benchmark_report.py"
    hf_path = tmp_path / "hf.jsonl"
    policy_path = tmp_path / "policy.jsonl"
    output_path = tmp_path / "report.md"

    _write_jsonl(
        hf_path,
        [
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "num_query_heads": 14,
                "num_key_value_heads": 2,
                "sketch_type": "count_sketch",
                "sketch_dim": 64,
                "block_topk_recall": 0.8,
                "block_recall_at_2x_budget": 0.95,
                "block_recall_at_4x_budget": 1.0,
                "block_score_correlation": 0.9,
            },
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "num_query_heads": 14,
                "num_key_value_heads": 2,
                "sketch_type": "random_projection",
                "sketch_dim": 64,
                "block_topk_recall": 0.7,
                "block_recall_at_2x_budget": 0.9,
                "block_recall_at_4x_budget": 0.98,
                "block_score_correlation": 0.88,
            },
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "num_query_heads": 14,
                "num_key_value_heads": 2,
                "sketch_type": "srht",
                "sketch_dim": 64,
                "block_topk_recall": 0.75,
                "block_recall_at_2x_budget": 0.93,
                "block_recall_at_4x_budget": 0.99,
                "block_score_correlation": 0.89,
            },
        ],
    )
    _write_jsonl(
        policy_path,
        [
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "sketch_type": "count_sketch",
                "sketch_dim": 64,
                "recent_window_blocks": 8,
                "candidate_budget_blocks": 16,
                "active_block_ratio": 0.61,
                "estimated_kv_reduction": 0.39,
                "exact_top_recall_in_active": 0.99,
            },
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "sketch_type": "count_sketch",
                "sketch_dim": 64,
                "recent_window_blocks": 4,
                "candidate_budget_blocks": 8,
                "active_block_ratio": 0.34,
                "estimated_kv_reduction": 0.66,
                "exact_top_recall_in_active": 0.96,
            },
            {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "extraction_mode": "separate_qk_proj",
                "qk_space": "pre_rope_projection",
                "sketch_type": "srht",
                "sketch_dim": 64,
                "recent_window_blocks": 8,
                "candidate_budget_blocks": 16,
                "active_block_ratio": 0.62,
                "estimated_kv_reduction": 0.38,
                "exact_top_recall_in_active": 0.98,
            },
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--hf-sweep",
            str(hf_path),
            "--policy-sim",
            str(policy_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(proc.stdout)
    assert summary["hf_rows"] == 3
    assert output_path.exists()
    report = output_path.read_text(encoding="utf-8")
    assert "Kivo-VD Offline Benchmark Report" in report
    assert "Executive Summary" in report
    assert "Retrieval Benchmark Summary" in report
    assert "Model and Extraction Metadata" in report
    assert "Qwen/Qwen2.5-0.5B" in report
    assert "pre_rope_projection" in report
    assert "Runtime post-RoPE attention behavior may differ" in report
    assert "Active KV Policy Simulation Summary" in report
    assert "SRHT should be compared against CountSketch" in report
    assert "Conservative Recommended Policy" in report
    assert "Aggressive Policy Notes" in report
    assert "What Is Proven vs Not Proven" in report
    assert "| count_sketch | 64 | 8 | 16 |" in report
    assert "| count_sketch | 64 | 4 | 8 |" in report
    assert "| srht | 64 | 8 | 16 |" in report


def test_benchmark_report_generator_missing_input_is_clear(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "generate_kivo_benchmark_report.py"
    missing = tmp_path / "missing.jsonl"
    output_path = tmp_path / "report.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--hf-sweep",
            str(missing),
            "--policy-sim",
            str(missing),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "input file not found" in proc.stderr
