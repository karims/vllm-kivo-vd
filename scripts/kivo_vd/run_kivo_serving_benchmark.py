#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run baseline-vs-Kivo OpenAI-compatible serving benchmarks.

This script is intentionally an orchestration/reporting wrapper. It starts a
vLLM OpenAI-compatible server, runs vLLM's existing serve benchmark against it,
and summarizes the resulting JSON files. It does not modify Kivo or vLLM runtime
logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT_TOKEN_TARGETS = "2000,3000"
DEFAULT_OUTPUT_TOKENS = "128,256"
DEFAULT_CONCURRENCY_LEVELS = "16,32,64"
DEFAULT_GPU_MEMORY_UTILIZATIONS = "0.10,0.06,0.04"


@dataclass(frozen=True)
class ServingCase:
    variant: str
    prompt_token_target: int
    output_tokens: int
    concurrency: int
    gpu_memory_utilization: float

    @property
    def case_id(self) -> str:
        util = str(self.gpu_memory_utilization).replace(".", "p")
        return (
            f"{self.variant}_isl{self.prompt_token_target}_"
            f"osl{self.output_tokens}_c{self.concurrency}_gpu{util}"
        )


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline and Kivo OpenAI-compatible vLLM serving benchmarks. "
            "This runner only orchestrates servers and reports metrics."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt-token-targets",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_PROMPT_TOKEN_TARGETS),
        help="Comma-separated random-dataset input token targets.",
    )
    parser.add_argument(
        "--output-tokens",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_OUTPUT_TOKENS),
        help="Comma-separated output token counts.",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_CONCURRENCY_LEVELS),
        help="Comma-separated max-concurrency levels.",
    )
    parser.add_argument(
        "--gpu-memory-utilizations",
        type=parse_float_list,
        default=parse_float_list(DEFAULT_GPU_MEMORY_UTILIZATIONS),
        help="Comma-separated gpu_memory_utilization values.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=180,
        help="Steady-arrival benchmark duration target in seconds.",
    )
    parser.add_argument(
        "--request-rate-multiplier",
        type=float,
        default=1.0,
        help="request_rate = concurrency * multiplier.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--server-ready-timeout", type=float, default=600.0)
    parser.add_argument("--server-shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tokenizer")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward --ignore-eos to vllm/benchmarks/serve.py.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument("--keep-recent-blocks", type=int, default=4)
    parser.add_argument("--max-full-blocks", type=int, default=64)
    parser.add_argument(
        "--kivo-mode",
        choices=("full", "apply-only"),
        default="full",
        help="Kivo env preset for the Kivo server variant.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kivo_vd/serving_benchmark",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def resolve_run_name(run_name: str | None) -> str:
    return run_name or f"serving_benchmark_{_timestamp()}"


def resolve_run_dir(args: argparse.Namespace, run_name: str) -> Path:
    return Path(args.output_dir) / run_name


def build_cases(args: argparse.Namespace) -> list[ServingCase]:
    cases: list[ServingCase] = []
    for variant in ("baseline", "kivo"):
        for gpu_util in args.gpu_memory_utilizations:
            for concurrency in args.concurrency_levels:
                for input_len in args.prompt_token_targets:
                    for output_len in args.output_tokens:
                        cases.append(
                            ServingCase(
                                variant=variant,
                                prompt_token_target=input_len,
                                output_tokens=output_len,
                                concurrency=concurrency,
                                gpu_memory_utilization=gpu_util,
                            )
                        )
    return cases


def _script(path: str) -> str:
    return str(REPO_ROOT / path)


def _quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _request_rate(case: ServingCase, args: argparse.Namespace) -> float:
    return case.concurrency * args.request_rate_multiplier


def _num_prompts(case: ServingCase, args: argparse.Namespace) -> int:
    return max(1, int(round(_request_rate(case, args) * args.duration_seconds)))


def kivo_env_for_case(case: ServingCase, args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    if case.variant != "kivo":
        return {}
    counters_file = run_dir / f"{case.case_id}_kivo_counters.json"
    env = {
        "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE": "1",
        "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY": "recent_only",
        "KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS": str(
            args.keep_recent_blocks
        ),
        "KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS": str(args.max_full_blocks),
        "KIVO_KV_RUNTIME_BLOCK_TABLE_REQUIRE_SLOT_MAPPING_REFRESH": "1",
        "KIVO_KV_DEMOTION_COUNTERS_ENABLE": "1",
        "KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE": str(counters_file),
    }
    if args.kivo_mode == "apply-only":
        env["KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION"] = "apply_block_table_only"
        return env

    env.update(
        {
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION": "apply_block_table_only",
            "KIVO_KV_DEMOTION_TRANSPORT_ENABLE": "1",
            "KIVO_KV_DEMOTION_TRANSPORT_ACTION": "apply_core_mark_demoted",
            "KIVO_KV_CORE_DEMOTION_ENABLE": "1",
            "KIVO_KV_CORE_DEMOTION_ACTION": "mark_demoted_only",
            "KIVO_KV_OWNERSHIP_REMOVE_ENABLE": "1",
            "KIVO_KV_OWNERSHIP_REMOVE_ACTION": "remove_marked_demoted_only",
            "KIVO_KV_FREE_TO_POOL_ENABLE": "1",
            "KIVO_KV_FREE_TO_POOL_ACTION": "free_removed_demoted_only",
        }
    )
    return env


def build_server_command(
    case: ServingCase,
    args: argparse.Namespace,
    port: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(case.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(case.concurrency),
        "--dtype",
        args.dtype,
    ]
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def build_benchmark_command(
    case: ServingCase,
    args: argparse.Namespace,
    port: int,
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        _script("vllm/benchmarks/serve.py"),
        "--backend",
        "openai",
        "--base-url",
        f"http://{args.host}:{port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--input-len",
        str(case.prompt_token_target),
        "--output-len",
        str(case.output_tokens),
        "--num-prompts",
        str(_num_prompts(case, args)),
        "--request-rate",
        str(_request_rate(case, args)),
        "--max-concurrency",
        str(case.concurrency),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--num-warmups",
        str(args.num_warmups),
        "--metric-percentiles",
        "50,95,99",
        "--percentile-metrics",
        "ttft,e2el",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
        "--metadata",
        f"variant={case.variant}",
        f"prompt_token_target={case.prompt_token_target}",
        f"requested_output_tokens={case.output_tokens}",
        f"gpu_memory_utilization={case.gpu_memory_utilization}",
        f"concurrency={case.concurrency}",
        f"duration_seconds={args.duration_seconds}",
    ]
    if args.ignore_eos:
        command.append("--ignore-eos")
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def build_case_plan(
    case: ServingCase,
    args: argparse.Namespace,
    run_dir: Path,
    index: int,
) -> dict[str, Any]:
    port = args.base_port + index
    benchmark_json = run_dir / f"{case.case_id}.benchmark.json"
    run_json = run_dir / f"{case.case_id}.run.json"
    server_log = run_dir / f"{case.case_id}.server.log"
    benchmark_log = run_dir / f"{case.case_id}.benchmark.log"
    return {
        "case_id": case.case_id,
        "variant": case.variant,
        "model": args.model,
        "prompt_token_target": case.prompt_token_target,
        "output_tokens": case.output_tokens,
        "concurrency": case.concurrency,
        "gpu_memory_utilization": case.gpu_memory_utilization,
        "duration_seconds": args.duration_seconds,
        "request_rate": _request_rate(case, args),
        "num_prompts": _num_prompts(case, args),
        "port": port,
        "server_command": build_server_command(case, args, port),
        "benchmark_command": build_benchmark_command(
            case, args, port, benchmark_json
        ),
        "kivo_env": kivo_env_for_case(case, args, run_dir),
        "paths": {
            "benchmark_json": str(benchmark_json),
            "run_json": str(run_json),
            "server_log": str(server_log),
            "benchmark_log": str(benchmark_log),
        },
    }


def build_plan(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    return [
        build_case_plan(case, args, run_dir, index)
        for index, case in enumerate(build_cases(args))
    ]


def _wait_for_server(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return True
        except URLError:
            time.sleep(2)
        except TimeoutError:
            time.sleep(2)
    return False


def _terminate_process(process: subprocess.Popen[Any], timeout: float) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def _preview_file(path: str | Path, limit: int = 4000) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) <= limit:
        return data
    return data[-limit:]


def run_case(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started_at = _iso_now()
    env = os.environ.copy()
    env.update(plan["kivo_env"])
    server_process: subprocess.Popen[Any] | None = None
    server_ready = False
    benchmark_returncode: int | None = None
    benchmark_started_at: str | None = None
    benchmark_ended_at: str | None = None
    error: str | None = None

    Path(plan["paths"]["server_log"]).parent.mkdir(parents=True, exist_ok=True)
    with open(plan["paths"]["server_log"], "w", encoding="utf-8") as server_log:
        try:
            server_process = subprocess.Popen(
                plan["server_command"],
                cwd=REPO_ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            server_ready = _wait_for_server(
                args.host,
                plan["port"],
                args.server_ready_timeout,
            )
            if not server_ready:
                error = "server did not become ready before timeout"
            elif server_process.poll() is not None:
                error = f"server exited early with {server_process.returncode}"
            else:
                benchmark_started_at = _iso_now()
                with open(
                    plan["paths"]["benchmark_log"], "w", encoding="utf-8"
                ) as benchmark_log:
                    benchmark = subprocess.run(
                        plan["benchmark_command"],
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=benchmark_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                benchmark_ended_at = _iso_now()
                benchmark_returncode = benchmark.returncode
                if benchmark.returncode != 0:
                    error = f"benchmark exited with {benchmark.returncode}"
        except Exception as exc:  # pragma: no cover - defensive runtime wrapper.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if server_process is not None:
                _terminate_process(server_process, args.server_shutdown_timeout)

    ended_at = _iso_now()
    result = {
        **{key: plan[key] for key in (
            "case_id",
            "variant",
            "model",
            "prompt_token_target",
            "output_tokens",
            "concurrency",
            "gpu_memory_utilization",
            "duration_seconds",
            "request_rate",
            "num_prompts",
            "port",
        )},
        "status": "succeeded" if error is None else "failed",
        "error": error,
        "started_at": started_at,
        "ended_at": ended_at,
        "server_ready": server_ready,
        "benchmark_returncode": benchmark_returncode,
        "benchmark_started_at": benchmark_started_at,
        "benchmark_ended_at": benchmark_ended_at,
        "server_command": plan["server_command"],
        "benchmark_command": plan["benchmark_command"],
        "kivo_env": plan["kivo_env"],
        "paths": plan["paths"],
        "server_log_preview": _preview_file(plan["paths"]["server_log"]),
        "benchmark_log_preview": _preview_file(plan["paths"]["benchmark_log"]),
    }
    Path(plan["paths"]["run_json"]).write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _load_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _first_number(data: dict[str, Any] | None, names: tuple[str, ...]) -> Any:
    if not data:
        return None
    for name in names:
        value = data.get(name)
        if isinstance(value, int | float):
            return value
    return None


def _failed_requests(data: dict[str, Any] | None, planned: int) -> int | None:
    if not data:
        return None
    completed = _first_number(data, ("completed", "successful_requests"))
    if isinstance(completed, int | float):
        return max(planned - int(completed), 0)
    errors = data.get("errors")
    if isinstance(errors, list):
        return len([error for error in errors if error])
    return None


def summarize_case(run_result: dict[str, Any]) -> dict[str, Any]:
    benchmark = _load_json(run_result["paths"]["benchmark_json"])
    planned = int(run_result.get("num_prompts", 0) or 0)
    completed = _first_number(benchmark, ("completed", "successful_requests"))
    failed = _failed_requests(benchmark, planned)
    timeout_or_error = failed
    if benchmark and isinstance(benchmark.get("errors"), list):
        timeout_or_error = len([error for error in benchmark["errors"] if error])

    return {
        "case_id": run_result["case_id"],
        "variant": run_result["variant"],
        "status": run_result["status"],
        "model": run_result["model"],
        "gpu_memory_utilization": run_result["gpu_memory_utilization"],
        "concurrency": run_result["concurrency"],
        "prompt_token_target": run_result["prompt_token_target"],
        "output_tokens_requested": run_result["output_tokens"],
        "duration_seconds": run_result["duration_seconds"],
        "request_rate": run_result["request_rate"],
        "num_prompts": planned,
        "completed_requests": completed,
        "failed_requests": failed,
        "total_input_tokens": _first_number(benchmark, ("total_input_tokens",)),
        "total_output_tokens": _first_number(benchmark, ("total_output_tokens",)),
        "output_tokens_per_second": _first_number(
            benchmark, ("output_throughput", "output_tokens_per_second")
        ),
        "total_tokens_per_second": _first_number(
            benchmark, ("total_token_throughput", "total_tokens_per_second")
        ),
        "requests_per_second": _first_number(
            benchmark, ("request_throughput", "requests_per_second")
        ),
        "p50_latency_ms": _first_number(
            benchmark, ("median_e2el_ms", "p50_e2el_ms")
        ),
        "p95_latency_ms": _first_number(benchmark, ("p95_e2el_ms",)),
        "p99_latency_ms": _first_number(benchmark, ("p99_e2el_ms",)),
        "p50_ttft_ms": _first_number(
            benchmark, ("median_ttft_ms", "p50_ttft_ms")
        ),
        "p95_ttft_ms": _first_number(benchmark, ("p95_ttft_ms",)),
        "p99_ttft_ms": _first_number(benchmark, ("p99_ttft_ms",)),
        "timeout_or_error_count": timeout_or_error,
        "benchmark_json": run_result["paths"]["benchmark_json"],
        "run_json": run_result["paths"]["run_json"],
        "server_log": run_result["paths"]["server_log"],
        "benchmark_log": run_result["paths"]["benchmark_log"],
    }


CSV_COLUMNS = [
    "case_id",
    "variant",
    "status",
    "model",
    "gpu_memory_utilization",
    "concurrency",
    "prompt_token_target",
    "output_tokens_requested",
    "duration_seconds",
    "request_rate",
    "num_prompts",
    "completed_requests",
    "failed_requests",
    "total_input_tokens",
    "total_output_tokens",
    "output_tokens_per_second",
    "total_tokens_per_second",
    "requests_per_second",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "timeout_or_error_count",
    "benchmark_json",
    "run_json",
    "server_log",
    "benchmark_log",
]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    columns = [
        "variant",
        "gpu_memory_utilization",
        "concurrency",
        "prompt_token_target",
        "output_tokens_requested",
        "completed_requests",
        "failed_requests",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "p95_latency_ms",
        "p95_ttft_ms",
        "status",
    ]
    lines = [
        "# Kivo-VD Serving Benchmark Summary",
        "",
        "This report compares baseline vLLM serving against a Kivo-enabled "
        "server using the same OpenAI-compatible benchmark workload.",
        "",
        "Important: this runner changes no Kivo runtime logic. Any Kivo behavior "
        "comes only from the environment flags passed to the Kivo server process.",
        "",
        "## Parameters",
        "",
        f"- model: `{args.model}`",
        f"- duration_seconds: `{args.duration_seconds}`",
        f"- prompt_token_targets: `{args.prompt_token_targets}`",
        f"- output_tokens: `{args.output_tokens}`",
        f"- concurrency_levels: `{args.concurrency_levels}`",
        f"- gpu_memory_utilizations: `{args.gpu_memory_utilizations}`",
        f"- ignore_eos: `{args.ignore_eos}`",
        "",
        "## Summary Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_value(row.get(col)) for col in columns) + " |")
    lines.extend(
        [
            "",
            "## Baseline Command Pattern",
            "",
            "```bash",
            "python -m vllm.entrypoints.openai.api_server \\",
            "  --model Qwen/Qwen2.5-0.5B-Instruct \\",
            "  --gpu-memory-utilization 0.10 \\",
            "  --max-model-len 4096 --max-num-batched-tokens 4096",
            "```",
            "",
            "## Kivo Command Pattern",
            "",
            "```bash",
            "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE=1 \\",
            "KIVO_KV_DEMOTION_TRANSPORT_ENABLE=1 \\",
            "KIVO_KV_CORE_DEMOTION_ENABLE=1 \\",
            "KIVO_KV_OWNERSHIP_REMOVE_ENABLE=1 \\",
            "KIVO_KV_FREE_TO_POOL_ENABLE=1 \\",
            "python -m vllm.entrypoints.openai.api_server \\",
            "  --model Qwen/Qwen2.5-0.5B-Instruct \\",
            "  --gpu-memory-utilization 0.10 \\",
            "  --max-model-len 4096 --max-num-batched-tokens 4096",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pipeline_summary(
    args: argparse.Namespace,
    run_name: str,
    run_dir: Path,
    plan: list[dict[str, Any]],
    run_results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "dry_run": args.dry_run,
        "started_at": started_at,
        "ended_at": ended_at,
        "success": all(row.get("status") in ("planned", "succeeded") for row in summary_rows),
        "parameters": {
            "model": args.model,
            "prompt_token_targets": args.prompt_token_targets,
            "output_tokens": args.output_tokens,
            "concurrency_levels": args.concurrency_levels,
            "gpu_memory_utilizations": args.gpu_memory_utilizations,
            "duration_seconds": args.duration_seconds,
            "request_rate_multiplier": args.request_rate_multiplier,
            "ignore_eos": args.ignore_eos,
            "kivo_mode": args.kivo_mode,
            "keep_recent_blocks": args.keep_recent_blocks,
            "max_full_blocks": args.max_full_blocks,
        },
        "output_files": {
            "csv_summary": str(run_dir / "serving_benchmark_summary.csv"),
            "markdown_summary": str(run_dir / "serving_benchmark_summary.md"),
            "plan_json": str(run_dir / "serving_benchmark_plan.json"),
            "pipeline_summary": str(run_dir / "pipeline_summary.json"),
        },
        "case_count": len(plan),
        "cases": run_results,
        "summary_rows": summary_rows,
    }
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def planned_result_from_case(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: plan[key] for key in (
            "case_id",
            "variant",
            "model",
            "prompt_token_target",
            "output_tokens",
            "concurrency",
            "gpu_memory_utilization",
            "duration_seconds",
            "request_rate",
            "num_prompts",
            "port",
        )},
        "status": "planned",
        "error": None,
        "server_command": plan["server_command"],
        "benchmark_command": plan["benchmark_command"],
        "server_command_string": _quote(plan["server_command"]),
        "benchmark_command_string": _quote(plan["benchmark_command"]),
        "kivo_env": plan["kivo_env"],
        "paths": plan["paths"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_name = resolve_run_name(args.run_name)
    run_dir = resolve_run_dir(args, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = _iso_now()
    plan = build_plan(args, run_dir)
    (run_dir / "serving_benchmark_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
    )

    run_results: list[dict[str, Any]] = []
    if args.dry_run:
        run_results = [planned_result_from_case(case) for case in plan]
    else:
        for case in plan:
            result = run_case(case, args)
            run_results.append(result)
            if result["status"] != "succeeded" and not args.continue_on_error:
                break

    summary_rows = [summarize_case(result) for result in run_results]
    for row, result in zip(summary_rows, run_results, strict=False):
        if result["status"] == "planned":
            row["status"] = "planned"

    csv_path = run_dir / "serving_benchmark_summary.csv"
    md_path = run_dir / "serving_benchmark_summary.md"
    write_csv(summary_rows, csv_path)
    write_markdown(summary_rows, md_path, args)
    ended_at = _iso_now()
    summary = write_pipeline_summary(
        args=args,
        run_name=run_name,
        run_dir=run_dir,
        plan=plan,
        run_results=run_results,
        summary_rows=summary_rows,
        started_at=started_at,
        ended_at=ended_at,
    )

    compact = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "dry_run": args.dry_run,
        "success": summary["success"],
        "case_count": len(plan),
        "csv_summary": str(csv_path),
        "markdown_summary": str(md_path),
        "pipeline_summary": str(run_dir / "pipeline_summary.json"),
        "note": "Kivo runtime logic is unchanged; this is serving orchestration only.",
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if summary["success"] or args.continue_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
