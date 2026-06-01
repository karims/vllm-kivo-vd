# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "run_hf_qk_head_sweep.py"
    spec = importlib.util.spec_from_file_location("run_hf_qk_head_sweep", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hf_head_sweep_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_hf_qk_head_sweep.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    out = proc.stdout
    assert "--layers" in out
    assert "--heads" in out
    assert "--query-positions" in out
    assert "--output" in out
    assert "--sketch-types" in out
    assert "--sketch-dims" in out


def test_parse_index_list() -> None:
    m = _load_module()
    assert m._parse_index_list("all", 3, "layer") == [0, 1, 2]
    assert m._parse_index_list("0,2", 3, "layer") == [0, 2]


def test_parse_query_positions() -> None:
    m = _load_module()
    hf_eval = m._load_hf_eval_module()
    assert m._parse_query_positions("sweep", 20, hf_eval)[-1] == 19
    assert m._parse_query_positions("3,-1", 20, hf_eval) == [3, 19]


def test_parse_sketch_types() -> None:
    m = _load_module()
    assert m._parse_sketch_types("random_projection", None) == ["random_projection"]
    assert m._parse_sketch_types(
        "random_projection", "count_sketch,random_projection"
    ) == ["count_sketch", "random_projection"]


def test_parse_sketch_dims() -> None:
    m = _load_module()
    assert m._parse_sketch_dims(64, None) == [64]
    assert m._parse_sketch_dims(64, "16,32,64,128") == [16, 32, 64, 128]
