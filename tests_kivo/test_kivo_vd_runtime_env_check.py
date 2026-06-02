# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


def test_runtime_env_check_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "check_vllm_runtime_env.py"

    proc = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert "python_version" in payload
    assert "torch" in payload
    assert "vllm" in payload
    assert "compiled_extension" in payload
