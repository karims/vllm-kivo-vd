# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import importlib.util
from pathlib import Path

import numpy as np


def _load_hf_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "run_hf_qk_sketch_eval.py"
    spec = importlib.util.spec_from_file_location("run_hf_qk_sketch_eval", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hf_qk_script_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_hf_qk_sketch_eval.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    out = proc.stdout
    assert "--model-name" in out
    assert "--sketch-type" in out
    assert "--topk-blocks" in out
    assert "--max-tokens" in out
    assert "--truncate-side" in out
    assert "--query-position" in out
    assert "--sweep-query-positions" in out
    assert "--extraction-mode" in out
    assert "srht" in out


def test_truncate_input_ids_right_and_left() -> None:
    m = _load_hf_script_module()
    tensor = np.array([[1, 2, 3, 4, 5, 6]])
    right, right_truncated = m._truncate_input_ids(tensor, 4, "right")
    left, left_truncated = m._truncate_input_ids(tensor, 4, "left")

    assert right_truncated is True
    assert left_truncated is True
    assert right.tolist() == [[1, 2, 3, 4]]
    assert left.tolist() == [[3, 4, 5, 6]]


def test_resolve_model_max_context_tokens_prefers_model() -> None:
    m = _load_hf_script_module()

    class Model:
        class config:
            n_positions = 1024

    class Tokenizer:
        model_max_length = 2048

    assert m._resolve_model_max_context_tokens(Model(), Tokenizer()) == 1024


def test_resolve_query_position() -> None:
    m = _load_hf_script_module()
    assert m._resolve_query_position("last", 10) == 9
    assert m._resolve_query_position("3", 10) == 3
    assert m._resolve_query_position("-1", 10) == 9


def test_sweep_query_positions() -> None:
    m = _load_hf_script_module()
    positions = m._sweep_query_positions(20)
    assert positions[-1] == 19
    assert all(1 <= p <= 19 for p in positions)


def test_map_query_head_to_kv_head_for_gqa() -> None:
    m = _load_hf_script_module()
    assert m._map_query_head_to_kv_head(0, 8, 2) == 0
    assert m._map_query_head_to_kv_head(3, 8, 2) == 0
    assert m._map_query_head_to_kv_head(4, 8, 2) == 1
    assert m._map_query_head_to_kv_head(7, 8, 2) == 1


def test_detect_extraction_mode_with_fake_attention() -> None:
    m = _load_hf_script_module()

    class GPT2Attention:
        c_attn = object()

    class SeparateAttention:
        q_proj = object()
        k_proj = object()

    assert m._detect_extraction_mode(GPT2Attention(), "auto") == "gpt2_fused_c_attn"
    assert m._detect_extraction_mode(SeparateAttention(), "auto") == "separate_qk_proj"


def test_sketch_compression_metadata_marks_full_dimensional() -> None:
    m = _load_hf_script_module()
    compressed = m._sketch_compression_metadata(
        head_dim=64,
        sketch_type="srht",
        sketch_dim=32,
    )
    full_dim = m._sketch_compression_metadata(
        head_dim=64,
        sketch_type="srht",
        sketch_dim=64,
    )
    expanded = m._sketch_compression_metadata(
        head_dim=64,
        sketch_type="random_projection",
        sketch_dim=128,
    )

    assert compressed["head_dim"] == 64
    assert compressed["effective_sketch_dim"] == 32
    assert compressed["sketch_compression_ratio"] == 0.5
    assert compressed["is_full_dimensional_sketch"] is False
    assert full_dim["effective_sketch_dim"] == 64
    assert full_dim["is_full_dimensional_sketch"] is True
    assert expanded["effective_sketch_dim"] == 64
    assert expanded["is_full_dimensional_sketch"] is True
