# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


def test_debug_export_dry_run_events_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "debug_export_dry_run_events.py"
    output = tmp_path / "nested" / "debug_events.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--num-blocks",
            "8",
            "--candidate-budget-blocks",
            "3",
            "--recent-window-blocks",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(proc.stdout)
    assert summary["export_path"] == str(output)
    assert summary["num_events_written"] == 3
    assert "counters" in summary
    assert summary["counters"]["num_dry_run_select_calls"] == 1

    assert output.exists()
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [row["event_type"] for row in rows]
    assert "dry_run_routing_decision" in event_types
    assert all("event_type" in row for row in rows)

    exported_text = output.read_text(encoding="utf-8")
    assert "shape" not in exported_text
    assert "dtype" not in exported_text
