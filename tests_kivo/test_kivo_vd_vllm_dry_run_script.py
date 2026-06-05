# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "run_vllm_kivo_dry_run.py"
    spec = importlib.util.spec_from_file_location("run_vllm_kivo_dry_run", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vllm_kivo_dry_run_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_vllm_kivo_dry_run.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    out = proc.stdout
    assert "--model" in out
    assert "--enable-kivo-vd" in out
    assert "--event-output" in out
    assert "--compare-baseline" in out
    assert "--gpu-memory-utilization" in out
    assert "--max-model-len" in out
    assert "--max-num-batched-tokens" in out
    assert "--max-num-seqs" in out


def test_vllm_kivo_dry_run_runtime_limits_are_parsed() -> None:
    m = _load_module()

    args = m._parse_args([
        "--gpu-memory-utilization",
        "0.07",
        "--max-model-len",
        "256",
        "--max-num-batched-tokens",
        "192",
        "--max-num-seqs",
        "2",
    ])

    assert args.gpu_memory_utilization == 0.07
    assert args.max_model_len == 256
    assert args.max_num_batched_tokens == 192
    assert args.max_num_seqs == 2


def test_vllm_kivo_dry_run_runtime_limits_are_llm_kwargs() -> None:
    m = _load_module()

    kwargs = m._build_llm_kwargs(
        model="sshleifer/tiny-gpt2",
        seed=123,
        dtype="auto",
        device="cuda",
        gpu_memory_utilization=0.05,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
    )

    assert kwargs["model"] == "sshleifer/tiny-gpt2"
    assert kwargs["gpu_memory_utilization"] == 0.05
    assert kwargs["max_model_len"] == 128
    assert kwargs["max_num_batched_tokens"] == 128
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["device"] == "cuda"


def test_extract_generation_text_handles_empty_outputs() -> None:
    m = _load_module()
    assert m._extract_generation_text([]) == ""


def test_extract_prompt_token_length_handles_missing_ids() -> None:
    m = _load_module()

    class Output:
        prompt_token_ids = None

    assert m._extract_prompt_token_length([Output()]) is None
