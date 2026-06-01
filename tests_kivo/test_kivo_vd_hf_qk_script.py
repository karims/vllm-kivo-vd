# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path


def test_hf_qk_script_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "run_hf_qk_sketch_eval.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    out = proc.stdout
    assert "--model-name" in out
    assert "--sketch-type" in out
    assert "--topk-blocks" in out
