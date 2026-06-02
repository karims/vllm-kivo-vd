# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "run_quality_benchmark_plan.py"
    spec = importlib.util.spec_from_file_location(
        "run_quality_benchmark_plan", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_needle_prompt_generator_contains_needle_and_query() -> None:
    m = _load_module()
    prompt, expected_answer = m.generate_needle_prompt(
        num_filler_repeats=2,
        needle="GREEN LANTERN",
        query="Which phrase matters?",
    )

    assert "GREEN LANTERN" in prompt
    assert "Which phrase matters?" in prompt
    assert expected_answer == "GREEN LANTERN"
    assert prompt.count("ordinary project updates") == 2


def test_quality_benchmark_scaffold_writes_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_quality_benchmark_plan.py"
    output_path = tmp_path / "quality_plan.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--benchmark",
            "needle_synthetic",
            "--model",
            "gpt2",
            "--needle",
            "BLUE ORCHID",
            "--num-filler-repeats",
            "3",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_payload = json.loads(proc.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload["benchmark"] == "needle_synthetic"
    assert file_payload["expected_answer"] == "BLUE ORCHID"
    assert "prompt" in file_payload


def test_dry_run_equality_plan_has_command() -> None:
    m = _load_module()

    class Args:
        benchmark = "dry_run_equality"
        model = "sshleifer/tiny-gpt2"

    plan = m.build_plan(Args())

    assert plan["benchmark"] == "dry_run_equality"
    assert "scripts/kivo_vd/run_vllm_kivo_dry_run.py" in plan["command"]
    assert "--enable-kivo-vd" in plan["command"]
