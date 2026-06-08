# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "validate_phase12_shadow_event.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_phase12_shadow_event",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event() -> dict:
    return {
        "event_type": "kivo_vd_shadow_selection",
        "version": "12.0",
        "request_id": "request-1",
        "layer_idx": 5,
        "context_token_count": 128,
        "block_size": 16,
        "total_context_blocks": 8,
        "candidate_budget_blocks": 4,
        "selected_block_ids_by_score": [6, 1, 4],
        "selected_block_ids_for_gather": [1, 4, 6],
        "selected_block_count": 3,
        "selected_ratio": 0.375,
        "ordering_valid": True,
        "causal_valid": True,
        "shadow_only": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
    }


def test_valid_fixture_passes() -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]
    fixture = (
        repo_root
        / "docs"
        / "kivo_vd"
        / "examples"
        / "phase12_shadow_event_example.jsonl"
    )

    report = module.validate_events(module.load_events(fixture))

    assert report["validation_passed"] is True
    assert report["total_events"] == 3
    assert report["invalid_events"] == 0


def test_unsorted_gather_ids_fail() -> None:
    module = _load_module()
    event = _event()
    event["selected_block_ids_for_gather"] = [4, 1, 6]

    report = module.validate_events([event])

    assert report["validation_passed"] is False
    assert any(
        error["check"] == "gather_order" for error in report["errors"]
    )


def test_score_and_gather_id_mismatch_fails() -> None:
    module = _load_module()
    event = _event()
    event["selected_block_ids_for_gather"] = [1, 4, 7]

    report = module.validate_events([event])

    assert report["validation_passed"] is False
    assert any(
        error["check"] == "selected_id_set_match"
        for error in report["errors"]
    )


def test_duplicate_ids_fail() -> None:
    module = _load_module()
    event = _event()
    event["selected_block_ids_by_score"] = [6, 1, 1]
    event["selected_block_ids_for_gather"] = [1, 1, 6]

    report = module.validate_events([event])

    assert report["validation_passed"] is False
    assert any(
        error["check"] == "duplicate_ids" for error in report["errors"]
    )


def test_active_routing_true_fails() -> None:
    module = _load_module()
    event = _event()
    event["active_routing"] = True

    report = module.validate_events([event])

    assert report["validation_passed"] is False
    assert any(
        error["check"] == "active_routing" for error in report["errors"]
    )


def test_measured_runtime_reduction_true_fails() -> None:
    module = _load_module()
    event = _event()
    event["measured_runtime_reduction"] = True

    report = module.validate_events([event])

    assert report["validation_passed"] is False
    assert any(
        error["check"] == "measured_runtime_reduction"
        for error in report["errors"]
    )


def test_selected_ratio_mismatch_warns_but_remains_valid() -> None:
    module = _load_module()
    event = _event()
    event["selected_ratio"] = 0.5

    report = module.validate_events([event])

    assert report["validation_passed"] is True
    assert report["valid_events"] == 1
    assert any(
        warning["check"] == "selected_ratio"
        for warning in report["warnings"]
    )


def test_cli_writes_outputs_for_valid_fixture(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "validate_phase12_shadow_event.py"
    )
    fixture = (
        repo_root
        / "docs"
        / "kivo_vd"
        / "examples"
        / "phase12_shadow_event_example.jsonl"
    )
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(fixture),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(process.stdout)["validation_passed"] is True
    assert json.loads(output_json.read_text())["valid_events"] == 3
    assert "Validation Status" in output_md.read_text()
