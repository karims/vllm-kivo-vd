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
        / "run_selected_kv_materialization_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_selected_kv_materialization_pipeline", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = {
        "events": "events.jsonl",
        "event_estimate": "event_estimate.json",
        "sketch_accounting": "sketch_accounting.json",
        "model": "gpt2",
        "num_layers": 12,
        "num_kv_heads": 12,
        "head_dim": 64,
        "block_size": 16,
        "dtype_bytes": 2,
        "device": "cuda",
        "max_events": 32,
        "num_pool_blocks": 256,
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
        / "run_selected_kv_materialization_pipeline.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--events",
        "--event-estimate",
        "--sketch-accounting",
        "--model",
        "--num-layers",
        "--num-kv-heads",
        "--head-dim",
        "--block-size",
        "--dtype-bytes",
        "--device",
        "--max-events",
        "--num-pool-blocks",
        "--run-name",
        "--output-dir",
        "--dry-run",
        "--continue-on-error",
    ):
        assert flag in process.stdout


def test_output_paths_are_under_run_dir(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")

    assert paths["materialization_json"] == str(
        tmp_path / "run" / "selected_kv_materialization.json"
    )
    assert paths["comparison_markdown"] == str(
        tmp_path
        / "run"
        / "selected_kv_materialization_comparison.md"
    )
    assert paths["pipeline_summary"] == str(
        tmp_path / "run" / "pipeline_summary.json"
    )


def test_commands_pass_inputs_metadata_and_outputs(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")
    stages = module.build_stage_commands(_args(), paths)

    assert [stage["name"] for stage in stages] == [
        "selected_kv_materialization",
        "selected_kv_materialization_comparison",
    ]
    materialization = stages[0]["command"]
    for flag, value in (
        ("--events", "events.jsonl"),
        ("--model", "gpt2"),
        ("--num-layers", "12"),
        ("--num-kv-heads", "12"),
        ("--head-dim", "64"),
        ("--block-size", "16"),
        ("--dtype-bytes", "2"),
        ("--device", "cuda"),
        ("--max-events", "32"),
        ("--num-pool-blocks", "256"),
    ):
        assert materialization[materialization.index(flag) + 1] == value
    assert paths["materialization_json"] in materialization

    comparison = stages[1]["command"]
    assert comparison[comparison.index("--materialization") + 1] == (
        paths["materialization_json"]
    )
    assert comparison[comparison.index("--event-estimate") + 1] == (
        "event_estimate.json"
    )
    assert comparison[comparison.index("--sketch-accounting") + 1] == (
        "sketch_accounting.json"
    )
    assert paths["comparison_json"] in comparison


def test_optional_inputs_are_omitted(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.output_paths(tmp_path / "run")
    stages = module.build_stage_commands(
        _args(sketch_accounting=None, num_pool_blocks=None), paths
    )

    assert "--num-pool-blocks" not in stages[0]["command"]
    assert "--sketch-accounting" not in stages[1]["command"]


def test_dry_run_writes_plan_without_executing_stages(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_selected_kv_materialization_pipeline.py"
    )
    run_dir = tmp_path / "planned-run"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--events",
            str(tmp_path / "missing-events.jsonl"),
            "--event-estimate",
            str(tmp_path / "missing-estimate.json"),
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
    assert compact["synthetic_kv"] is True
    assert compact["outside_attention_path"] is True
    assert compact["full_kv_still_allocated"] is True
    assert compact["active_routing"] is False
    assert compact["measured_runtime_reduction"] is False
    assert [stage["status"] for stage in summary["stages"]] == [
        "planned",
        "planned",
    ]
    assert [stage["name"] for stage in summary["stages"]] == [
        "selected_kv_materialization",
        "selected_kv_materialization_comparison",
    ]
    assert not (run_dir / "selected_kv_materialization.json").exists()
    assert not (
        run_dir / "selected_kv_materialization_comparison.md"
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

    assert summary["parameters"]["events"] == "events.jsonl"
    assert summary["parameters"]["device"] == "cuda"
    assert summary["output_files"]["comparison_json"].endswith(
        "selected_kv_materialization_comparison.json"
    )
    assert summary["synthetic_kv"] is True
    assert summary["outside_attention_path"] is True
    assert summary["full_kv_still_allocated"] is True
    assert summary["active_routing"] is False
    assert summary["measured_runtime_reduction"] is False
