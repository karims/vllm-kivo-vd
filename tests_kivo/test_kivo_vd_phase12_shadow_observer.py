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


def _observation(module, *, scores=None):
    return module.Phase12ShadowObservation(
        request_id="request-1",
        sequence_id="sequence-1",
        layer_idx=5,
        step_idx=3,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        scores=scores,
    )


def test_disabled_observer_writes_nothing(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_shadow_observer.py",
        "phase12_shadow_observer_disabled_test",
    )
    output = tmp_path / "events.jsonl"
    observer = module.Phase12ShadowObserver(
        module.Phase12ShadowObserverConfig(output_jsonl=str(output))
    )

    assert observer.observe(_observation(module)) is None
    assert output.exists() is False
    assert observer.get_counters() == {
        "events_seen": 1,
        "events_written": 0,
        "invalid_events": 0,
        "warnings": 0,
    }


def test_enabled_observer_writes_valid_scored_event(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "phase12_shadow_observer.py",
        "phase12_shadow_observer_scored_test",
    )
    output = tmp_path / "events.jsonl"
    observer = module.Phase12ShadowObserver(
        module.Phase12ShadowObserverConfig(
            enabled=True,
            output_jsonl=str(output),
            preview_only=False,
        )
    )
    scores = [0.1, 0.8, -0.2, 0.4, 0.9, 0.3, -0.5, 0.7]

    event = observer.observe(_observation(module, scores=scores))

    assert event is not None
    assert event.preview_only is False
    assert event.shadow_only is True
    assert event.active_routing is False
    assert event.measured_runtime_reduction is False
    assert event.selected_block_ids_by_score == [4, 1, 7, 3]
    assert event.selected_block_ids_for_gather == [1, 3, 4, 7]
    assert set(event.selected_block_ids_by_score) == set(
        event.selected_block_ids_for_gather
    )
    assert observer.get_counters()["events_written"] == 1


def test_missing_scores_creates_preview_event(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_shadow_observer.py",
        "phase12_shadow_observer_preview_test",
    )
    observer = module.Phase12ShadowObserver(
        module.Phase12ShadowObserverConfig(
            enabled=True,
            output_jsonl=str(tmp_path / "events.jsonl"),
            preview_only=False,
        )
    )

    event = observer.observe(_observation(module))

    assert event is not None
    assert event.preview_only is True
    assert event.selector_scores_summary is None
    assert event.selected_block_ids_for_gather == [4, 5, 6, 7]


def test_invalid_observation_updates_counters_without_writing(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "phase12_shadow_observer.py",
        "phase12_shadow_observer_invalid_test",
    )
    output = tmp_path / "events.jsonl"
    observer = module.Phase12ShadowObserver(
        module.Phase12ShadowObserverConfig(
            enabled=True,
            output_jsonl=str(output),
        )
    )
    invalid = module.Phase12ShadowObservation(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=[0, 1],
    )

    assert observer.observe(invalid) is None
    assert output.exists() is False
    assert observer.get_counters() == {
        "events_seen": 1,
        "events_written": 0,
        "invalid_events": 1,
        "warnings": 1,
    }


def test_output_events_pass_validator(tmp_path: Path) -> None:
    observer_module = _load_module(
        "phase12_shadow_observer.py",
        "phase12_shadow_observer_validation_test",
    )
    validator_module = _load_module(
        "validate_phase12_shadow_event.py",
        "phase12_shadow_validator_observer_test",
    )
    output = tmp_path / "events.jsonl"
    observer = observer_module.Phase12ShadowObserver(
        observer_module.Phase12ShadowObserverConfig(
            enabled=True,
            output_jsonl=str(output),
            preview_only=False,
        )
    )
    scores = [0.1, 0.8, -0.2, 0.4, 0.9, 0.3, -0.5, 0.7]

    observer.observe(_observation(observer_module, scores=scores))
    events = validator_module.load_events(output)
    report = validator_module.validate_events(events)

    assert report["validation_passed"] is True
    assert json.loads(output.read_text())["ordering_valid"] is True


def test_observer_does_not_mutate_input_sequences(tmp_path: Path) -> None:
    module = _load_module(
        "phase12_shadow_observer.py",
        "phase12_shadow_observer_input_test",
    )
    observer = module.Phase12ShadowObserver(
        module.Phase12ShadowObserverConfig(
            enabled=True,
            output_jsonl=str(tmp_path / "events.jsonl"),
            preview_only=False,
        )
    )
    block_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    scores = [0.1, 0.8, -0.2, 0.4, 0.9, 0.3, -0.5, 0.7]
    original_block_ids = block_ids.copy()
    original_scores = scores.copy()
    observation = module.Phase12ShadowObservation(
        request_id="request-1",
        layer_idx=5,
        context_token_count=128,
        total_context_blocks=8,
        block_ids=block_ids,
        scores=scores,
    )

    observer.observe(observation)

    assert block_ids == original_block_ids
    assert scores == original_scores


def test_config_rejects_active_routing() -> None:
    module = _load_module(
        "phase12_shadow_observer.py",
        "phase12_shadow_observer_config_test",
    )

    try:
        module.Phase12ShadowObserverConfig(active_routing=True)
    except ValueError as exc:
        assert "active_routing=false" in str(exc)
    else:
        raise AssertionError("unsafe active routing config was accepted")
