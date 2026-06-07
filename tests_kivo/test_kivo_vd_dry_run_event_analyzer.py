# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "analyze_dry_run_events.py"
    spec = importlib.util.spec_from_file_location("analyze_dry_run_events", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyzer_summarizes_synthetic_events(tmp_path: Path) -> None:
    m = _load_module()
    input_path = tmp_path / "events.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "event_type": "after_allocate_slots",
                "request_id": "r1",
                "source": "waiting",
            },
            {
                "event_type": "dry_run_routing_decision",
                "request_id": "r1",
                "source": "waiting",
                "selected_block_count": 6,
                "recent_block_count": 2,
                "skipped_block_count": 1,
                "candidate_budget_blocks": 8,
                "recent_window_blocks": 4,
                "selected_block_preview": [1, 2, 3],
                "selected_block_ids_full": [1, 2, 3, 4, 5, 6],
                "full_block_ids_exported": True,
            },
            {
                "event_type": "free_request",
                "request_id": "r1",
                "source": "free_blocks",
            },
        ],
    )

    summary = m.analyze_events(input_path)

    assert summary["total_events"] == 3
    assert summary["event_counts"]["dry_run_routing_decision"] == 1
    assert summary["num_dry_run_routing_decision_events"] == 1
    assert summary["avg_selected_block_count"] == 6
    assert summary["avg_recent_block_count"] == 2
    assert summary["avg_skipped_block_count"] == 1
    assert summary["candidate_budget_blocks"] == [8]
    assert summary["recent_window_blocks"] == [4]
    assert summary["full_block_ids_exported_count"] == 1
    assert summary["preview_only_routing_event_count"] == 0
    assert summary["all_routing_events_have_full_block_ids"] is True
    assert summary["request_ids_seen"] == ["r1"]
    assert "waiting" in summary["sources_seen"]
    assert not summary["warnings"]


def test_analyzer_missing_file_warns(tmp_path: Path) -> None:
    m = _load_module()
    summary = m.analyze_events(tmp_path / "missing.jsonl")

    assert summary["total_events"] == 0
    assert "event file is missing" in summary["warnings"][0]


def test_analyzer_counts_malformed_rows(tmp_path: Path) -> None:
    m = _load_module()
    input_path = tmp_path / "events.jsonl"
    input_path.write_text(
        '{"event_type":"after_allocate_slots"}\nnot json\n',
        encoding="utf-8",
    )

    summary = m.analyze_events(input_path)

    assert summary["total_events"] == 1
    assert summary["malformed_rows"] == 1
    assert any("malformed JSONL row" in warning for warning in summary["warnings"])


def test_analyzer_warns_when_no_routing_events(tmp_path: Path) -> None:
    m = _load_module()
    input_path = tmp_path / "events.jsonl"
    _write_jsonl(
        input_path,
        [
            {"event_type": "after_allocate_slots", "request_id": "r1"},
            {"event_type": "free_request", "request_id": "r1"},
        ],
    )

    summary = m.analyze_events(input_path)

    assert "no dry_run_routing_decision events found" in summary["warnings"]
    assert "only allocation/free events found" in summary["warnings"]


def test_analyzer_cli_writes_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "analyze_dry_run_events.py"
    input_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "summary.json"
    _write_jsonl(
        input_path,
        [
            {
                "event_type": "dry_run_routing_decision",
                "request_id": "r1",
                "selected_block_count": 0,
                "recent_block_count": 0,
                "skipped_block_count": 0,
            }
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_summary = json.loads(proc.stdout)
    file_summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_summary["total_events"] == 1
    assert file_summary["total_events"] == 1
    assert "selected block count is always zero" in file_summary["warnings"]
