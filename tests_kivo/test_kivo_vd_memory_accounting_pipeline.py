# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_memory_accounting_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_memory_accounting_pipeline", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = {
        "model": "gpt2",
        "prompt": "test prompt",
        "max_tokens": 32,
        "gpu_memory_utilization": 0.05,
        "max_model_len": 256,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
        "num_layers": 12,
        "num_kv_heads": 12,
        "head_dim": 64,
        "block_size": 16,
        "dtype_bytes": 2,
        "run_name": "test-run",
        "output_dir": None,
        "dry_run": True,
        "continue_on_error": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_pipeline_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_memory_accounting_pipeline.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--prompt",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--num-layers",
        "--num-kv-heads",
        "--head-dim",
        "--block-size",
        "--dtype-bytes",
        "--run-name",
        "--output-dir",
        "--dry-run",
        "--continue-on-error",
    ):
        assert flag in process.stdout


def test_output_paths_are_under_run_dir(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")

    assert paths["baseline_memory"] == str(
        tmp_path / "run" / "baseline_memory.json"
    )
    assert paths["comparison_markdown"] == str(
        tmp_path / "run" / "memory_comparison.md"
    )
    assert paths["pipeline_summary"] == str(
        tmp_path / "run" / "pipeline_summary.json"
    )


def test_stage_commands_pass_runtime_and_kv_metadata(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")
    stages = module.build_stage_commands(_args(), paths)

    assert [stage["name"] for stage in stages] == [
        "baseline_memory_measurement",
        "kivo_dry_run_memory_measurement",
        "event_memory_estimate",
        "memory_comparison_report",
    ]
    baseline_command = stages[0]["command"]
    assert "--model" in baseline_command
    assert "gpt2" in baseline_command
    assert "--gpu-memory-utilization" in baseline_command
    assert "0.05" in baseline_command
    assert "--max-model-len" in baseline_command
    assert "--enable-kivo-vd" not in baseline_command

    kivo_command = stages[1]["command"]
    assert "--enable-kivo-vd" in kivo_command
    assert "--event-output" in kivo_command

    estimate_command = stages[2]["command"]
    assert estimate_command[estimate_command.index("--num-layers") + 1] == "12"
    assert estimate_command[estimate_command.index("--num-kv-heads") + 1] == "12"
    assert estimate_command[estimate_command.index("--head-dim") + 1] == "64"
    assert "--block-size" in estimate_command
    assert "--dtype-bytes" in estimate_command


def test_dry_run_writes_planned_summary_without_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_memory_accounting_pipeline.py"
    )
    run_dir = tmp_path / "planned-run"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--run-name",
            "planned-run",
            "--output-dir",
            str(run_dir),
            "--num-layers",
            "12",
            "--num-kv-heads",
            "12",
            "--head-dim",
            "64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    compact = json.loads(process.stdout)
    summary_path = run_dir / "pipeline_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert compact["dry_run"] is True
    assert compact["savings_are_theoretical_only"] is True
    assert summary["success"] is True
    assert [stage["status"] for stage in summary["stages"]] == [
        "planned",
        "planned",
        "planned",
        "planned",
    ]
    assert [stage["name"] for stage in summary["stages"]] == [
        "baseline_memory_measurement",
        "kivo_dry_run_memory_measurement",
        "event_memory_estimate",
        "memory_comparison_report",
    ]
    assert not (run_dir / "baseline_memory.json").exists()
    assert not (run_dir / "memory_comparison.md").exists()


def test_initial_summary_schema_contains_outputs(tmp_path: Path) -> None:
    module = _load_module()
    args = _args()
    paths = module.output_paths(tmp_path)
    summary = module.build_initial_summary(
        args,
        "test-run",
        tmp_path,
        paths,
        "2026-01-01T00:00:00Z",
    )

    assert summary["run_name"] == "test-run"
    assert summary["output_files"]["event_estimate_json"].endswith(
        "kivo_event_memory_estimate.json"
    )
    assert summary["savings_are_theoretical_only"] is True
    assert summary["measured_runtime_reduction"] is False
