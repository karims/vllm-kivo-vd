# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "materialize_selected_kv.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_selected_kv", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_reads_routing_events_and_ignores_other_rows(
    tmp_path: Path,
) -> None:
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            {"event_type": "after_allocate_slots"},
            {
                "event_type": "dry_run_routing_decision",
                "selected_block_ids": [1, 2],
            },
            {"name": "dry-run-routing-decision", "selected_block_ids": [3]},
        ],
    )

    events, warnings = module.read_routing_events(events_path, 32)

    assert len(events) == 2
    assert warnings == []


def test_extracts_full_ids_and_marks_preview_fallback() -> None:
    module = _load_module()

    full = module.extract_selected_blocks({
        "selected_block_count": 3,
        "selected_block_ids": [4, 2, 1],
    })
    preview = module.extract_selected_blocks({
        "selected_block_count": 16,
        "selected_block_preview": list(range(8)),
    })

    assert full == ([4, 2, 1], 3, False, None)
    assert preview[:3] == (list(range(8)), 16, True)
    assert "preview-only" in preview[3]


def test_prefers_explicit_full_block_ids() -> None:
    module = _load_module()

    result = module.extract_selected_blocks({
        "selected_block_count": 4,
        "selected_block_ids_full": [4, 3, 2, 1],
        "selected_block_preview": [4, 3],
        "full_block_ids_exported": True,
    })

    assert result == ([4, 3, 2, 1], 4, False, None)


def test_bytes_per_block_formula() -> None:
    module = _load_module()

    result = module.bytes_per_kv_block(
        num_layers=12,
        num_kv_heads=12,
        head_dim=64,
        block_size=16,
        dtype_bytes=2,
    )

    assert result == 589_824


def test_cpu_synthetic_materialization_and_aggregates(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "event_type": "dry_run_routing_decision",
                "event_id": 1,
                "request_id": "r1",
                "selected_block_count": 2,
                "selected_block_ids_full": [1, 3],
                "full_block_ids_exported": True,
                "skipped_block_count": 2,
            },
            {
                "event_type": "dry_run_routing_decision",
                "event_id": 2,
                "request_id": "r1",
                "selected_block_count": 1,
                "selected_block_ids_full": [0],
                "full_block_ids_exported": True,
                "skipped_block_count": 3,
            },
        ],
    )

    report = module.materialize_selected_kv(
        events_path=events_path,
        model="tiny",
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=2,
        dtype_bytes=4,
        device_name="cpu",
        max_events=32,
        num_pool_blocks=4,
    )

    assert report["num_events_processed"] == 2
    assert report["bytes_per_block"] == 64
    assert report["aggregate"]["average_selected_blocks"] == 1.5
    assert report["aggregate"]["average_selected_kv_bytes"] == 96
    assert report["aggregate"]["average_materialization_ratio"] == 0.375
    assert report["aggregate"][
        "total_selected_kv_bytes_materialized"
    ] == 192
    assert report["aggregate"]["full_block_ids_exported_count"] == 2
    assert report["aggregate"]["preview_only_event_count"] == 0
    assert not any("preview-only" in item for item in report["warnings"])
    assert all(
        row["copy_time_ms"] >= 0 for row in report["per_event_rows"]
    )
    assert report["caveats"]["synthetic_kv"] is True
    markdown = module.render_markdown(report)
    assert "Complete selected block IDs were exported" in markdown
    assert "Preview-only events undercount" not in markdown


def test_missing_selected_ids_produce_empty_report_warning(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [{
            "event_type": "dry_run_routing_decision",
            "selected_block_count": 4,
            "skipped_block_count": 2,
        }],
    )

    report = module.materialize_selected_kv(
        events_path=events_path,
        model="tiny",
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=2,
        dtype_bytes=4,
        device_name="cpu",
        max_events=32,
        num_pool_blocks=4,
    )

    assert report["num_routing_events_read"] == 1
    assert report["num_events_processed"] == 0
    assert any("lacks selected block IDs" in item for item in report["warnings"])


def test_markdown_contains_required_caveats(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    module = _load_module()
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(events_path, [])
    report = module.materialize_selected_kv(
        events_path=events_path,
        model="tiny",
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=2,
        dtype_bytes=4,
        device_name="cpu",
        max_events=32,
        num_pool_blocks=4,
    )

    markdown = module.render_markdown(report)

    assert "KV tensors are synthetic" in markdown
    assert "outside the attention path" in markdown
    assert "Full KV is still allocated" in markdown
    assert "No active routing" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_markdown_keeps_preview_only_warning() -> None:
    module = _load_module()
    report = {
        "model_kv_metadata": {},
        "aggregate": {"preview_only_event_count": 1},
        "num_events_processed": 1,
        "per_event_rows": [{
            "event_id": 1,
            "requested_selected_block_count": 16,
            "materialized_selected_block_count": 8,
            "skipped_block_count": 24,
            "selected_kv_bytes": 8,
            "materialization_ratio": 0.2,
            "copy_time_ms": 0.1,
            "selected_ids_preview_only": True,
        }],
        "warnings": [],
    }

    markdown = module.render_markdown(report)

    assert "Preview-only events undercount" in markdown
    assert "Complete selected block IDs were exported" not in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "kivo_vd" / "materialize_selected_kv.py"

    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--events",
        "--model",
        "--num-layers",
        "--num-kv-heads",
        "--head-dim",
        "--block-size",
        "--dtype-bytes",
        "--device",
        "--max-events",
        "--num-pool-blocks",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
