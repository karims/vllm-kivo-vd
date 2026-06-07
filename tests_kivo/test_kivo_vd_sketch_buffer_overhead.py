# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "measure_sketch_buffer_overhead.py"
    )
    spec = importlib.util.spec_from_file_location(
        "measure_sketch_buffer_overhead", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_kv_bytes_formula() -> None:
    module = _load_module()

    result = module.full_kv_bytes(
        num_layers=12,
        num_kv_heads=12,
        head_dim=64,
        block_size=16,
        num_blocks=256,
        dtype_bytes=2,
    )

    assert result == 150_994_944


def test_sketch_buffer_bytes_formula() -> None:
    module = _load_module()

    result = module.sketch_buffer_bytes(
        num_layers=12,
        num_kv_heads=12,
        sketch_dim=32,
        num_blocks=256,
        dtype_bytes=2,
    )

    assert result == 2_359_296


def test_sketch_overhead_ratio() -> None:
    module = _load_module()

    ratio = module.sketch_overhead_ratio(2_359_296, 150_994_944)

    assert ratio == pytest.approx(0.015625)


def test_report_schema_and_markdown_caveats() -> None:
    module = _load_module()
    row = {
        "sketch_type": "count_sketch",
        "sketch_dim": 16,
        "sketch_buffer_shape": [12, 12, 256, 16],
        "theoretical_sketch_bytes": 1_179_648,
        "sketch_overhead_ratio_vs_full_kv": 0.0078125,
        "measured_allocated_delta_bytes": None,
        "measured_reserved_delta_bytes": None,
    }
    report = module.build_report(
        model="gpt2",
        num_layers=12,
        num_kv_heads=12,
        head_dim=64,
        block_size=16,
        num_blocks=256,
        dtype_bytes=2,
        sketch_types=["count_sketch"],
        sketch_dims=[16],
        device_name="cpu",
        cuda_available=False,
        rows=[row],
    )

    assert report["overhead_only"] is True
    assert report["replaces_full_kv"] is False
    assert report["active_routing"] is False
    assert report["measured_runtime_reduction"] is False
    assert report["measured_cuda_available"] is False
    assert report["aggregate"]["recommended_small_config"]["sketch_dim"] == 16

    markdown = module.render_markdown(report)
    assert "overhead only" in markdown.lower()
    assert "do not replace full KV" in markdown
    assert "No active routing" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_cpu_mode_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "measure_sketch_buffer_overhead.py"
    )
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--num-layers",
            "2",
            "--num-kv-heads",
            "2",
            "--head-dim",
            "8",
            "--block-size",
            "4",
            "--num-blocks",
            "4",
            "--sketch-types",
            "count_sketch,bidiagonal_sign_subsample",
            "--sketch-dims",
            "2,4",
            "--device",
            "cpu",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(process.stdout)
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["num_configurations"] == 4
    assert report["device"] == "cpu"
    assert report["cuda_available"] in (True, False)
    assert report["measured_cuda_available"] is False
    assert len(report["rows"]) == 4
    assert all(
        row["measured_allocated_delta_bytes"] is None
        for row in report["rows"]
    )
    assert output_md.is_file()


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "measure_sketch_buffer_overhead.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--num-layers",
        "--num-kv-heads",
        "--head-dim",
        "--block-size",
        "--num-blocks",
        "--dtype-bytes",
        "--sketch-dims",
        "--sketch-types",
        "--device",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
