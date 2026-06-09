# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts" / "kivo_vd"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _env(output: Path, *, enabled: bool = True) -> dict[str, str]:
    return {
        "KIVO_PHASE12_SHADOW_ENABLED": "1" if enabled else "0",
        "KIVO_PHASE12_SHADOW_OUTPUT": str(output),
        "KIVO_PHASE12_RATIO_POLICY": "balanced=0:0.60,5:0.45",
        "KIVO_PHASE12_PREVIEW_ONLY": "0",
    }


def test_disabled_helper_returns_noop_and_writes_nothing(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "phase12_vllm_runtime_touchpoint.py",
        "phase12_runtime_touchpoint_disabled_test",
    )
    output = tmp_path / "events.jsonl"

    result = module.observe_phase12_decode_shadow_metadata(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        env=_env(output, enabled=False),
    )

    assert result["enabled"] is False
    assert result["event_written"] is False
    assert result["reason"] == "disabled"
    assert output.exists() is False


def test_enabled_helper_emits_valid_event(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_runtime_touchpoint.py",
        "phase12_runtime_touchpoint_enabled_test",
    )
    validator = _load_module(
        "validate_phase12_shadow_event.py",
        "phase12_runtime_touchpoint_validator_test",
    )
    output = tmp_path / "events.jsonl"

    result = module.observe_phase12_decode_shadow_metadata(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=list(range(8)),
        scores={block_id: float(block_id) for block_id in range(8)},
        env=_env(output),
    )
    report = validator.validate_events(validator.load_events(output))
    event = json.loads(output.read_text(encoding="utf-8"))

    assert result["enabled"] is True
    assert result["event_written"] is True
    assert report["validation_passed"] is True
    assert event["active_routing"] is False
    assert event["measured_runtime_reduction"] is False


def test_block_table_helper_creates_preview_event(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_runtime_touchpoint.py",
        "phase12_runtime_touchpoint_block_table_test",
    )
    output = tmp_path / "events.jsonl"

    result = module.observe_phase12_block_table_shadow_metadata(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=list(range(8)),
        env=_env(output),
    )
    event = json.loads(output.read_text(encoding="utf-8"))

    assert result["event_written"] is True
    assert event["preview_only"] is True
    assert event["selected_block_ids_for_gather"] == [4, 5, 6, 7]


def test_input_block_ids_are_not_mutated(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_runtime_touchpoint.py",
        "phase12_runtime_touchpoint_input_test",
    )
    output = tmp_path / "events.jsonl"
    block_ids = list(range(8))
    original = block_ids.copy()

    module.observe_phase12_decode_shadow_metadata(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=block_ids,
        scores={block_id: float(block_id) for block_id in block_ids},
        env=_env(output),
    )

    assert block_ids == original


def test_exceptions_are_caught_and_fail_closed(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_runtime_touchpoint.py",
        "phase12_runtime_touchpoint_fail_closed_test",
    )
    output = tmp_path / "events.jsonl"

    result = module.observe_phase12_decode_shadow_metadata(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=[0, 1],
        env=_env(output),
    )

    assert result["event_written"] is False
    assert result["active_routing"] is False
    assert result["measured_runtime_reduction"] is False
    assert output.exists() is False


def test_import_is_lightweight_without_vllm_runtime() -> None:
    module = _load_module(
        "phase12_vllm_runtime_touchpoint.py",
        "phase12_runtime_touchpoint_import_test",
    )

    assert module.is_phase12_shadow_enabled({}) is False
    assert "vllm" not in module.__dict__
