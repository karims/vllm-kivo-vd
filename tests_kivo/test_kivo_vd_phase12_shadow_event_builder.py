# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_ratio_policy_and_derive_budget() -> None:
    module = _load_module(
        "phase12_shadow_events.py",
        "phase12_shadow_events_parse_test",
    )
    policy = module.parse_ratio_policy("balanced=0:0.60,5:0.45")

    assert policy.name == "balanced"
    assert policy.layer_ratios == {0: 0.6, 5: 0.45}
    assert module.derive_layer_budget(32, 0, policy) == 20
    assert module.derive_layer_budget(32, 5, policy) == 15


def test_budget_clamps_to_minimum_maximum_and_total() -> None:
    module = _load_module(
        "phase12_shadow_events.py",
        "phase12_shadow_events_budget_test",
    )
    low = module.parse_ratio_policy("low=0:0.01")
    high = module.parse_ratio_policy("high=0:1.0")

    assert module.derive_layer_budget(32, 0, low, min_budget=4) == 4
    assert module.derive_layer_budget(32, 0, high, max_budget=8) == 8
    assert module.derive_layer_budget(3, 0, low, min_budget=8) == 3


def test_score_order_is_converted_to_sequence_gather_order() -> None:
    module = _load_module(
        "phase12_shadow_events.py",
        "phase12_shadow_events_order_test",
    )
    policy = module.parse_ratio_policy("balanced=5:0.5")
    event = module.build_phase12_shadow_event(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        block_size=16,
        total_context_blocks=8,
        ratio_policy=policy,
        candidate_budget_blocks=4,
        selected_block_ids_by_score=[6, 1, 4],
        selector_policy="synthetic_score",
        scores=[0.2, 0.8, -0.1],
    )

    assert event.selected_block_ids_by_score == [6, 1, 4]
    assert event.selected_block_ids_for_gather == [1, 4, 6]
    assert event.ordering_valid is True
    assert event.selected_ratio == 3 / 8
    assert event.estimated_active_block_reduction_ratio == 5 / 8


def test_ordering_validator_detects_set_mismatch() -> None:
    module = _load_module(
        "phase12_shadow_events.py",
        "phase12_shadow_events_mismatch_test",
    )

    assert module.validate_ordering([6, 1, 4], [1, 4, 7]) is False
    assert module.validate_ordering([6, 1, 4], [4, 1, 6]) is False


def test_generated_events_pass_phase12_validator(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generator = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "generate_phase12_synthetic_shadow_events.py"
    )
    validator = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "validate_phase12_shadow_event.py"
    )
    events_path = tmp_path / "events.jsonl"
    events_md = tmp_path / "events.md"
    report_path = tmp_path / "validation.json"
    report_md = tmp_path / "validation.md"

    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--num-events",
            "8",
            "--output-jsonl",
            str(events_path),
            "--output-md",
            str(events_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--input",
            str(events_path),
            "--output-json",
            str(report_path),
            "--output-md",
            str(report_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["validation_passed"] is True
    assert report["valid_events"] == 8
    assert all(event["shadow_only"] is True for event in events)
    assert all(event["active_routing"] is False for event in events)
    assert all(
        event["measured_runtime_reduction"] is False for event in events
    )
