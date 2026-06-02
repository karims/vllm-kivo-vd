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
        repo_root / "scripts" / "kivo_vd" / "run_offline_benchmark_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_offline_benchmark_pipeline", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = {
        "model_name": "gpt2",
        "prompt_mode": "blue_orchid",
        "sketch_types": "count_sketch,random_projection",
        "sketch_dims": "32,64",
        "layers": "0",
        "heads": "0",
        "extraction_mode": "auto",
        "block_size": 16,
        "topk_blocks": 4,
        "max_tokens": 128,
        "recent_window_blocks": "4,8",
        "candidate_budget_blocks": "8,16",
        "run_torch_benchmark": False,
        "output_dir": "outputs/kivo_vd/runs",
        "run_name": "unit-test-run",
        "dry_run": True,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_blue_orchid_prompt_builder() -> None:
    m = _load_module()
    prompt = m.build_blue_orchid_prompt()

    assert "BLUE ORCHID" in prompt
    assert "Question:" in prompt
    assert prompt.count("ordinary facts") == 48


def test_run_dir_path_generation(tmp_path: Path) -> None:
    m = _load_module()
    run_dir = m.resolve_run_dir(tmp_path, "my-run")

    assert run_dir == tmp_path / "my-run"


def test_summary_json_writing(tmp_path: Path) -> None:
    m = _load_module()
    summary_path = tmp_path / "nested" / "pipeline_summary.json"
    m.write_pipeline_summary({"success": True}, summary_path)

    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"success": True}


def test_dry_run_pipeline_writes_planned_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_offline_benchmark_pipeline.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "dry-run",
            "--layers",
            "0",
            "--heads",
            "0",
            "--sketch-dims",
            "32",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    summary_path = tmp_path / "dry-run" / "pipeline_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert summary["dry_run"] is True
    assert summary["parameters"]["extraction_mode"] == "auto"
    assert all(stage["status"] == "planned" for stage in summary["stages"])
    assert not (tmp_path / "dry-run" / "hf_qk_head_sweep_ranked.jsonl").exists()


def test_build_stage_commands_includes_ranked_sweep(tmp_path: Path) -> None:
    m = _load_module()
    stages = m.build_stage_commands(_args(), tmp_path / "run")
    hf_stage = stages[0]

    assert hf_stage["name"] == "hf_qk_head_sweep"
    assert "--include-ranked-blocks" in hf_stage["command"]
    assert str(tmp_path / "run" / "hf_qk_head_sweep_ranked.jsonl") in hf_stage[
        "command"
    ]


def test_pipeline_stage_commands_include_extraction_mode(tmp_path: Path) -> None:
    m = _load_module()
    stages = m.build_stage_commands(
        _args(extraction_mode="separate_qk_proj"), tmp_path / "run"
    )
    command = stages[0]["command"]

    assert "--extraction-mode" in command
    assert "separate_qk_proj" in command
