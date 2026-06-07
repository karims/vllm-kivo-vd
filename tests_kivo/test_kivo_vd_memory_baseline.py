# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "run_vllm_memory_baseline.py"
    spec = importlib.util.spec_from_file_location(
        "run_vllm_memory_baseline", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True

    def memory_allocated(self) -> int:
        return 10

    def memory_reserved(self) -> int:
        return 20

    def max_memory_allocated(self) -> int:
        return 30

    def max_memory_reserved(self) -> int:
        return 40

    def mem_get_info(self) -> tuple[int, int]:
        return 50, 60


def _checkpoints() -> list[dict[str, int | float | str | None]]:
    names = [
        "process_start",
        "before_llm_init",
        "after_llm_init",
        "before_generate",
        "after_generate",
        "after_request_or_cleanup",
    ]
    checkpoints = []
    for index, name in enumerate(names):
        checkpoints.append({
            "name": name,
            "timestamp": float(index),
            "memory_allocated_bytes": index * 10,
            "memory_reserved_bytes": index * 20,
            "max_memory_allocated_bytes": index * 30,
            "max_memory_reserved_bytes": index * 40,
            "free_memory_bytes": 1000 - index,
            "total_memory_bytes": 1000,
        })
    return checkpoints


def test_memory_baseline_help_includes_runtime_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_vllm_memory_baseline.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--enable-kivo-vd",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--output",
    ):
        assert flag in proc.stdout


def test_memory_checkpoint_uses_cuda_metrics() -> None:
    module = _load_module()
    cuda = _FakeCuda()

    checkpoint = module._capture_memory_checkpoint(
        "before_generate",
        cuda,
        timestamp_fn=lambda: 123.5,
    )

    assert cuda.synchronized
    assert checkpoint == {
        "name": "before_generate",
        "timestamp": 123.5,
        "memory_allocated_bytes": 10,
        "memory_reserved_bytes": 20,
        "max_memory_allocated_bytes": 30,
        "max_memory_reserved_bytes": 40,
        "free_memory_bytes": 50,
        "total_memory_bytes": 60,
    }


def test_result_schema_marks_kivo_as_dry_run_only() -> None:
    module = _load_module()

    result = module._build_result(
        config={"model": "gpt2"},
        runtime={"torch_version": "test"},
        output_text=" output",
        prompt_token_length=4,
        checkpoints=_checkpoints(),
        kivo_enabled=True,
        observer_counters={"num_dry_run_select_calls": 1},
        event_output="events.jsonl",
        num_events_exported=2,
        observer_note=None,
    )

    assert result["dry_run_only"] is True
    assert result["kivo_enabled"] is True
    assert result["config"]["model"] == "gpt2"
    assert len(result["memory_checkpoints"]) == 6
    assert result["peak_deltas"]["llm_init_allocated_delta_bytes"] == 10
    assert result["event_output"] == "events.jsonl"


def test_llm_kwargs_use_conservative_defaults() -> None:
    module = _load_module()
    args = module._parse_args([])

    kwargs = module._build_llm_kwargs(args)

    assert kwargs["gpu_memory_utilization"] == 0.05
    assert kwargs["max_model_len"] == 256
    assert kwargs["max_num_batched_tokens"] == 256
    assert kwargs["max_num_seqs"] == 1
