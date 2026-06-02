# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "simulate_active_kv_policy.py"
    spec = importlib.util.spec_from_file_location(
        "simulate_active_kv_policy", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_row() -> dict:
    return {
        "exact_top_block_ids": [1, 5, 6, 7],
        "approx_ranked_block_ids": [5, 2, 3, 9, 1, 4, 6, 7],
        "num_keys_used": 160,
        "block_size": 16,
        "sketch_type": "count_sketch",
        "sketch_dim": 64,
        "layer": 0,
        "head": 1,
        "query_position": 159,
        "model_name": "Qwen/Qwen2.5-0.5B",
        "extraction_mode": "separate_qk_proj",
        "qk_space": "pre_rope_projection",
        "num_query_heads": 14,
        "num_key_value_heads": 2,
        "selected_query_head": 3,
        "selected_kv_head": 0,
        "head_dim": 64,
        "effective_input_dim": 64,
        "effective_sketch_dim": 32,
        "sketch_compression_ratio": 0.5,
        "is_full_dimensional_sketch": False,
    }


def test_active_block_union_and_reduction() -> None:
    m = _load_module()
    out = m.simulate_policy_for_row(
        _tiny_row(),
        recent_window_blocks=2,
        candidate_budget_blocks=3,
        topk_blocks=4,
        min_total_blocks=1,
    )

    assert out is not None
    assert out["num_total_blocks"] == 10
    # candidates {5, 2, 3} + recent {8, 9}
    assert out["active_block_count"] == 5
    assert out["active_block_ratio"] == 0.5
    assert out["estimated_kv_reduction"] == 0.5
    assert out["exact_top_recall_in_active"] == 0.25
    assert out["model_name"] == "Qwen/Qwen2.5-0.5B"
    assert out["qk_space"] == "pre_rope_projection"
    assert out["selected_query_head"] == 3
    assert out["selected_kv_head"] == 0
    assert out["head_dim"] == 64
    assert out["effective_sketch_dim"] == 32
    assert out["sketch_compression_ratio"] == 0.5
    assert out["is_full_dimensional_sketch"] is False


def test_missing_ranked_blocks_gives_clear_error() -> None:
    m = _load_module()
    row = _tiny_row()
    row.pop("approx_ranked_block_ids")

    try:
        m.simulate_policy_for_row(
            row,
            recent_window_blocks=4,
            candidate_budget_blocks=8,
            topk_blocks=4,
            min_total_blocks=1,
        )
    except ValueError as exc:
        assert "--include-ranked-blocks" in str(exc)
    else:
        raise AssertionError("Expected missing ranked blocks to raise ValueError")


def test_simulator_cli_writes_jsonl(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "simulate_active_kv_policy.py"
    input_path = tmp_path / "hf_rows.jsonl"
    output_path = tmp_path / "policy.jsonl"
    input_path.write_text(json.dumps(_tiny_row()) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--recent-window-blocks",
            "2",
            "--candidate-budget-blocks",
            "3",
            "--topk-blocks",
            "4",
            "--min-total-blocks",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(proc.stdout)
    assert summary["output_rows"] == 1
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["active_block_count"] == 5
    assert rows[0]["estimated_kv_reduction"] == 0.5
