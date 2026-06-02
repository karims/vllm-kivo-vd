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


def test_extract_generation_text_handles_empty_outputs() -> None:
    m = _load_module()
    assert m._extract_generation_text([]) == ""


def test_extract_prompt_token_length_handles_missing_ids() -> None:
    m = _load_module()

    class Output:
        prompt_token_ids = None

    assert m._extract_prompt_token_length([Output()]) is None
