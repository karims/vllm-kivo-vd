# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "run_kivo_serving_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_kivo_serving_benchmark", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_help_includes_serving_benchmark_flags() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_kivo_serving_benchmark.py"
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--prompt-token-targets",
        "--output-tokens",
        "--concurrency-levels",
        "--gpu-memory-utilizations",
        "--duration-seconds",
        "--ignore-eos",
        "--kivo-mode",
        "--dry-run",
    ):
        assert flag in process.stdout


def test_plan_builds_baseline_and_kivo_commands(tmp_path: Path) -> None:
    module = _load_module()
    args = module._parse_args(
        [
            "--prompt-token-targets",
            "2000",
            "--output-tokens",
            "128",
            "--concurrency-levels",
            "16",
            "--gpu-memory-utilizations",
            "0.1",
            "--duration-seconds",
            "3",
            "--run-name",
            "unit",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    run_dir = module.resolve_run_dir(args, "unit")
    plan = module.build_plan(args, run_dir)

    assert len(plan) == 2
    assert [case["variant"] for case in plan] == ["baseline", "kivo"]
    assert plan[0]["kivo_env"] == {}
    assert plan[1]["kivo_env"]["KIVO_KV_FREE_TO_POOL_ENABLE"] == "1"
    assert plan[0]["num_prompts"] == 48
    assert "--input-len" in plan[0]["benchmark_command"]
    assert "2000" in plan[0]["benchmark_command"]
    assert "--ignore-eos" in plan[0]["benchmark_command"]
    assert "--gpu-memory-utilization" in plan[1]["server_command"]
    assert "0.1" in plan[1]["server_command"]


def test_dry_run_writes_plan_csv_and_markdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_kivo_serving_benchmark.py"
    run_dir = tmp_path / "dry-run"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--prompt-token-targets",
            "2000",
            "--output-tokens",
            "128",
            "--concurrency-levels",
            "16",
            "--gpu-memory-utilizations",
            "0.10",
            "--duration-seconds",
            "1",
            "--run-name",
            "dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    compact = json.loads(process.stdout)
    summary = json.loads((run_dir / "pipeline_summary.json").read_text())

    assert compact["dry_run"] is True
    assert compact["case_count"] == 2
    assert summary["success"] is True
    assert summary["cases"][0]["status"] == "planned"
    assert (run_dir / "serving_benchmark_plan.json").exists()
    assert (run_dir / "serving_benchmark_summary.csv").exists()
    assert (run_dir / "serving_benchmark_summary.md").exists()
    markdown = (run_dir / "serving_benchmark_summary.md").read_text()
    assert "Baseline Command Pattern" in markdown
    assert "Kivo Command Pattern" in markdown


def test_summarize_case_reads_synthetic_benchmark_json(tmp_path: Path) -> None:
    module = _load_module()
    benchmark_path = tmp_path / "case.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "completed": 9,
                "total_input_tokens": 18000,
                "total_output_tokens": 1152,
                "output_throughput": 100.5,
                "total_token_throughput": 1500.25,
                "request_throughput": 0.75,
                "median_e2el_ms": 1000.0,
                "p95_e2el_ms": 2000.0,
                "p99_e2el_ms": 3000.0,
                "median_ttft_ms": 50.0,
                "p95_ttft_ms": 75.0,
                "p99_ttft_ms": 100.0,
                "errors": ["", "timeout"],
            }
        ),
        encoding="utf-8",
    )
    run_result = {
        "case_id": "baseline_isl2000_osl128_c16_gpu0p1",
        "variant": "baseline",
        "status": "succeeded",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "gpu_memory_utilization": 0.1,
        "concurrency": 16,
        "prompt_token_target": 2000,
        "output_tokens": 128,
        "duration_seconds": 180,
        "request_rate": 16.0,
        "num_prompts": 10,
        "paths": {
            "benchmark_json": str(benchmark_path),
            "run_json": str(tmp_path / "run.json"),
            "server_log": str(tmp_path / "server.log"),
            "benchmark_log": str(tmp_path / "benchmark.log"),
        },
    }

    row = module.summarize_case(run_result)

    assert row["completed_requests"] == 9
    assert row["failed_requests"] == 1
    assert row["timeout_or_error_count"] == 1
    assert row["total_input_tokens"] == 18000
    assert row["output_tokens_per_second"] == 100.5
    assert row["p95_latency_ms"] == 2000.0
    assert row["p95_ttft_ms"] == 75.0
