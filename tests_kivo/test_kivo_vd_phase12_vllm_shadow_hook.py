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


def _call(hook, *, scores=None, block_ids=None):
    effective_ids = list(range(8)) if block_ids is None else block_ids
    return hook.observe_decode_metadata(
        request_id="request-1",
        sequence_id="sequence-1",
        layer_idx=5,
        step_idx=2,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=effective_ids,
        scores=scores,
        metadata={"source": "test"},
    )


def test_env_config_is_disabled_by_default() -> None:
    module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_default_env_test",
    )

    config = module.build_config_from_env({})

    assert config.enabled is False
    assert config.preview_only is True
    assert config.block_size == 16
    assert config.max_budget is None


def test_env_parses_booleans_integers_and_max_budget() -> None:
    module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_env_test",
    )
    config = module.build_config_from_env({
        "KIVO_PHASE12_SHADOW_ENABLED": "yes",
        "KIVO_PHASE12_PREVIEW_ONLY": "0",
        "KIVO_PHASE12_BLOCK_SIZE": "32",
        "KIVO_PHASE12_MIN_BUDGET": "2",
        "KIVO_PHASE12_MAX_BUDGET": "7",
    })

    assert config.enabled is True
    assert config.preview_only is False
    assert config.block_size == 32
    assert config.min_budget == 2
    assert config.max_budget == 7


def test_disabled_hook_writes_no_event(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_disabled_test",
    )
    output = tmp_path / "events.jsonl"
    hook = module.Phase12VllmShadowHook(
        module.Phase12VllmShadowHookConfig(output_jsonl=str(output))
    )

    result = _call(hook)

    assert result["enabled"] is False
    assert result["event_written"] is False
    assert result["reason"] == "disabled"
    assert output.exists() is False


def test_enabled_hook_writes_validator_compatible_event(
    tmp_path: Path,
) -> None:
    hook_module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_enabled_test",
    )
    validator_module = _load_module(
        "validate_phase12_shadow_event.py",
        "phase12_vllm_hook_validator_test",
    )
    output = tmp_path / "events.jsonl"
    hook = hook_module.Phase12VllmShadowHook(
        hook_module.Phase12VllmShadowHookConfig(
            enabled=True,
            output_jsonl=str(output),
            preview_only=False,
        )
    )
    scores = {
        0: 0.1,
        1: 0.8,
        2: -0.2,
        3: 0.4,
        4: 0.9,
        5: 0.3,
        6: -0.5,
        7: 0.7,
    }

    result = _call(hook, scores=scores)
    report = validator_module.validate_events(
        validator_module.load_events(output)
    )
    event = json.loads(output.read_text(encoding="utf-8"))

    assert result["event_written"] is True
    assert result["event_summary"]["preview_only"] is False
    assert result["event_summary"]["metadata_keys"] == ["source"]
    assert report["validation_passed"] is True
    assert event["active_routing"] is False
    assert event["measured_runtime_reduction"] is False


def test_hook_catches_observer_rejection(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_error_test",
    )
    hook = module.Phase12VllmShadowHook(
        module.Phase12VllmShadowHookConfig(
            enabled=True,
            output_jsonl=str(tmp_path / "events.jsonl"),
        )
    )

    result = _call(hook, block_ids=[0, 1])

    assert result["event_written"] is False
    assert result["reason"] == "observer_rejected_metadata"
    assert result["error"]
    assert hook.errors == 1


def test_hook_catches_unexpected_observer_exception(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_exception_test",
    )
    hook = module.Phase12VllmShadowHook(
        module.Phase12VllmShadowHookConfig(
            enabled=True,
            output_jsonl=str(tmp_path / "events.jsonl"),
        )
    )
    assert hook.observer is not None

    def raise_unexpected(*args, **kwargs):
        raise RuntimeError("synthetic observer failure")

    hook.observer.observe = raise_unexpected
    result = _call(hook)

    assert result["event_written"] is False
    assert result["reason"] == "hook_error"
    assert result["error"] == "synthetic observer failure"
    assert hook.errors == 1


def test_invalid_env_returns_fail_closed_disabled_hook() -> None:
    module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_invalid_env_test",
    )

    hook = module.maybe_get_shadow_hook_from_env({
        "KIVO_PHASE12_SHADOW_ENABLED": "perhaps",
    })
    result = _call(hook)

    assert hook.config.enabled is False
    assert result["event_written"] is False
    assert result["reason"] == "disabled"
    assert result["error"]


def test_hook_does_not_mutate_inputs(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_vllm_shadow_hook.py",
        "phase12_vllm_hook_inputs_test",
    )
    hook = module.Phase12VllmShadowHook(
        module.Phase12VllmShadowHookConfig(
            enabled=True,
            output_jsonl=str(tmp_path / "events.jsonl"),
            preview_only=False,
        )
    )
    block_ids = list(range(8))
    scores = {block_id: float(block_id) for block_id in block_ids}
    metadata = {"source": "test", "nested": {"value": 1}}
    original_ids = block_ids.copy()
    original_scores = scores.copy()
    original_metadata = {"source": "test", "nested": {"value": 1}}

    hook.observe_decode_metadata(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=block_ids,
        scores=scores,
        metadata=metadata,
    )

    assert block_ids == original_ids
    assert scores == original_scores
    assert metadata == original_metadata
