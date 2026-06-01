# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


def test_sketch_sweep_quick_creates_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_sketch_sweep.py"
    output = tmp_path / "sweep.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--quick",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0


def test_sketch_sweep_jsonl_fields(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_sketch_sweep.py"
    output = tmp_path / "sweep_fields.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--quick",
            "--output",
            str(output),
            "--seed",
            "3",
            "--num-tokens",
            "128",
            "--input-dim",
            "32",
            "--block-size",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    first_row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    required = {
        "mode",
        "sketch_type",
        "sketch_dim",
        "num_tokens",
        "input_dim",
        "block_size",
        "topk_blocks",
        "seed",
        "token_topk_recall",
        "block_topk_recall",
        "block_recall_at_2x_budget",
        "block_recall_at_4x_budget",
        "block_mrr",
        "token_score_correlation",
        "block_score_correlation",
        "exact_top_block_ids",
        "approx_top_block_ids",
        "runtime_ms",
    }
    assert required.issubset(first_row.keys())
