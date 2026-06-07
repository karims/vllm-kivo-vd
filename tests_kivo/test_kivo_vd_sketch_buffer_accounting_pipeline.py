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
        / "run_sketch_buffer_accounting_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_sketch_buffer_accounting_pipeline", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = {
        "event_estimate": "phase7_event_estimate.json",
        "memory_comparison": "phase7_memory_comparison.json",
        "model": "gpt2",
        "num_layers": 12,
        "num_kv_heads": 12,
        "head_dim": 64,
        "block_size": 16,
        "num_blocks": 256,
        "dtype_bytes": 2,
        "sketch_types": (
            "count_sketch,random_projection,"
            "bidiagonal_sign_subsample"
        ),
        "sketch_dims": "16,32,64",
        "device": "cpu",
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
        / "run_sketch_buffer_accounting_pipeline.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--event-estimate",
        "--memory-comparison",
        "--model",
        "--num-layers",
        "--num-kv-heads",
        "--head-dim",
        "--block-size",
        "--num-blocks",
        "--dtype-bytes",
        "--sketch-types",
        "--sketch-dims",
        "--device",
        "--run-name",
        "--output-dir",
        "--dry-run",
        "--continue-on-error",
    ):
        assert flag in process.stdout


def test_output_paths_are_under_run_dir(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")

    assert paths["sketch_overhead_json"] == str(
        tmp_path / "run" / "sketch_buffer_overhead.json"
    )
    assert paths["event_accounting_markdown"] == str(
        tmp_path
        / "run"
        / "event_aware_sketch_buffer_accounting.md"
    )
    assert paths["pipeline_summary"] == str(
        tmp_path / "run" / "pipeline_summary.json"
    )


def test_commands_pass_inputs_metadata_and_output_paths(
    tmp_path: Path,
) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")
    stages = module.build_stage_commands(_args(), paths)

    assert [stage["name"] for stage in stages] == [
        "sketch_buffer_overhead_measurement",
        "overhead_vs_savings_comparison",
        "event_aware_sketch_buffer_accounting",
    ]
    overhead = stages[0]["command"]
    for flag, value in (
        ("--model", "gpt2"),
        ("--num-layers", "12"),
        ("--num-kv-heads", "12"),
        ("--head-dim", "64"),
        ("--block-size", "16"),
        ("--num-blocks", "256"),
        ("--dtype-bytes", "2"),
        ("--sketch-dims", "16,32,64"),
        ("--device", "cpu"),
    ):
        assert overhead[overhead.index(flag) + 1] == value
    assert "bidiagonal_sign_subsample" in overhead[
        overhead.index("--sketch-types") + 1
    ]
    assert paths["sketch_overhead_json"] in overhead

    for stage in stages[1:]:
        command = stage["command"]
        assert command[command.index("--event-estimate") + 1] == (
            "phase7_event_estimate.json"
        )
        assert command[command.index("--memory-comparison") + 1] == (
            "phase7_memory_comparison.json"
        )
        assert command[command.index("--sketch-overhead") + 1] == (
            paths["sketch_overhead_json"]
        )


def test_optional_memory_comparison_is_omitted(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")
    stages = module.build_stage_commands(
        _args(memory_comparison=None), paths
    )

    assert "--memory-comparison" not in stages[1]["command"]
    assert "--memory-comparison" not in stages[2]["command"]


def test_dry_run_writes_plan_without_executing_stages(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_sketch_buffer_accounting_pipeline.py"
    )
    run_dir = tmp_path / "planned-run"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--event-estimate",
            str(tmp_path / "missing-event-estimate.json"),
            "--memory-comparison",
            str(tmp_path / "missing-memory-comparison.json"),
            "--output-dir",
            str(run_dir),
            "--run-name",
            "planned-run",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    compact = json.loads(process.stdout)
    summary = json.loads(
        (run_dir / "pipeline_summary.json").read_text(encoding="utf-8")
    )
    assert compact["success"] is True
    assert compact["dry_run"] is True
    assert compact["savings_are_theoretical_only"] is True
    assert compact["measured_runtime_reduction"] is False
    assert compact["active_routing"] is False
    assert [stage["status"] for stage in summary["stages"]] == [
        "planned",
        "planned",
        "planned",
    ]
    assert not (run_dir / "sketch_buffer_overhead.json").exists()
    assert not (
        run_dir / "event_aware_sketch_buffer_accounting.md"
    ).exists()


def test_initial_summary_has_expected_schema(tmp_path: Path) -> None:
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

    assert summary["parameters"]["event_estimate"] == (
        "phase7_event_estimate.json"
    )
    assert summary["parameters"]["device"] == "cpu"
    assert summary["output_files"]["event_accounting_json"].endswith(
        "event_aware_sketch_buffer_accounting.json"
    )
    assert summary["savings_are_theoretical_only"] is True
    assert summary["measured_runtime_reduction"] is False
    assert summary["active_routing"] is False
