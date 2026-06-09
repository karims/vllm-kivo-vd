# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_plugin():
    plugin_root = _repo_root() / "plugins" / "kivo_vllm_shadow_plugin"
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    module = importlib.import_module("kivo_vllm_shadow_plugin.plugin")
    return importlib.reload(module)


def _load_shadow_events():
    plugin_root = _repo_root() / "plugins" / "kivo_vllm_shadow_plugin"
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    return importlib.import_module("kivo_vllm_shadow_plugin.shadow_events")


def _load_validator():
    path = (
        _repo_root()
        / "scripts"
        / "kivo_vd"
        / "validate_phase12_shadow_event.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase12_plugin_event_validator_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_probe():
    path = (
        _repo_root()
        / "scripts"
        / "kivo_vd"
        / "run_phase12_vllm_plugin_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase12_plugin_generate_probe_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_vllm(result):
    class FakeLLM:
        def generate(
            self,
            prompts,
            sampling_params=None,
            *args,
            **kwargs,
        ):
            return result

    return SimpleNamespace(
        LLM=FakeLLM,
        __version__="0.test",
        __file__="/site-packages/vllm/__init__.py",
    )


def test_default_register_does_not_patch_generate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    result = object()
    fake_vllm = _fake_vllm(result)
    original = fake_vllm.LLM.generate
    marker = tmp_path / "marker.json"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv(plugin.MARKER_ENV, str(marker))
    monkeypatch.delenv(plugin.PATCH_GENERATE_ENV, raising=False)

    plugin.register()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert fake_vllm.LLM.generate is original
    assert payload["patch_generate_requested"] is False
    assert payload["patch_generate_installed"] is False


def test_generate_patch_is_once_only_and_returns_original_object() -> None:
    plugin = _load_plugin()
    result = object()
    prompts = ["prompt"]
    fake_vllm = _fake_vllm(result)
    emitted: list[dict] = []

    def emitter(**kwargs):
        emitted.append(kwargs)
        return 1

    installed, _ = plugin.install_generate_patch(
        fake_vllm,
        emitter=emitter,
    )
    wrapper = fake_vllm.LLM.generate
    installed_again, _ = plugin.install_generate_patch(
        fake_vllm,
        emitter=emitter,
    )
    returned = fake_vllm.LLM().generate(prompts)

    assert installed is True
    assert installed_again is True
    assert fake_vllm.LLM.generate is wrapper
    assert returned is result
    assert prompts == ["prompt"]
    assert emitted == []


def test_generate_patch_emits_without_mutating_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    result = [SimpleNamespace(request_id="r1", prompt_token_ids=[1, 2, 3])]
    prompts = ["hello"]
    original_prompts = list(prompts)
    fake_vllm = _fake_vllm(result)
    emitted: list[dict] = []

    def emitter(**kwargs):
        emitted.append(kwargs)
        return 4

    monkeypatch.setenv(plugin.EVENTS_ENV, str(tmp_path / "events.jsonl"))
    plugin.install_generate_patch(fake_vllm, emitter=emitter)

    returned = fake_vllm.LLM().generate(prompts)

    assert returned is result
    assert prompts == original_prompts
    assert emitted[0]["prompts"] is prompts
    assert emitted[0]["result"] is result


def test_generate_patch_fails_closed_on_emitter_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    result = object()
    fake_vllm = _fake_vllm(result)
    marker = tmp_path / "marker.json"
    monkeypatch.setenv(plugin.EVENTS_ENV, str(tmp_path / "events.jsonl"))
    monkeypatch.setenv(plugin.MARKER_ENV, str(marker))
    plugin.write_load_marker(marker)

    def broken_emitter(**kwargs):
        raise RuntimeError("synthetic emission failure")

    plugin.install_generate_patch(fake_vllm, emitter=broken_emitter)
    returned = fake_vllm.LLM().generate(["prompt"])
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert returned is result
    assert "synthetic emission failure" in payload["runtime_warnings"][0]


def test_preview_events_validate(tmp_path: Path) -> None:
    shadow_events = _load_shadow_events()
    validator = _load_validator()
    output = SimpleNamespace(
        request_id="request-7",
        prompt_token_ids=list(range(65)),
    )
    path = tmp_path / "events.jsonl"

    written = shadow_events.emit_shadow_events_from_generate_call(
        prompts=["ignored because exact token IDs are available"],
        result=[output],
        events_path=path,
        layers=(0, 5),
        block_size=16,
        ratio_policy="balanced=0:0.60,5:0.45",
    )
    events = validator.load_events(path)
    validation = validator.validate_events(events)

    assert written == 2
    assert validation["validation_passed"] is True
    assert all(event["preview_only"] is True for event in events)
    assert all(event["active_routing"] is False for event in events)
    assert all(
        event["measured_runtime_reduction"] is False for event in events
    )


def test_register_records_generate_patch_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    fake_vllm = _fake_vllm([])
    marker = tmp_path / "marker.json"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv(plugin.MARKER_ENV, str(marker))
    monkeypatch.setenv(plugin.PATCH_GENERATE_ENV, "1")

    plugin.register()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["patch_generate_requested"] is True
    assert payload["patch_generate_installed"] is True
    assert payload["original_generate_qualname"].endswith("FakeLLM.generate")


def test_probe_requires_valid_events_for_internal_hook_candidate(
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    shadow_events = _load_shadow_events()
    probe = _load_probe()
    args = probe._parse_args([
        "--enable-generate-hook",
        "--marker-path",
        str(tmp_path / "marker.json"),
        "--events-jsonl",
        str(tmp_path / "events.jsonl"),
        "--output-json",
        str(tmp_path / "report.json"),
        "--output-md",
        str(tmp_path / "report.md"),
    ])

    def load_plugin():
        plugin.write_load_marker(
            args.marker_path,
            plugin.build_plugin_state(
                patch_generate_requested=True,
                patch_generate_installed=True,
                original_generate_qualname="FakeLLM.generate",
            ),
        )
        return {
            "vllm_version": "0.22.1",
            "vllm_file": "/site-packages/vllm/__init__.py",
        }

    def generate(_args):
        shadow_events.emit_shadow_events_from_generate_call(
            prompts=["hello"],
            result=[
                SimpleNamespace(
                    request_id="request-1",
                    prompt_token_ids=list(range(32)),
                )
            ],
            events_path=args.events_jsonl,
            layers=(0,),
            block_size=16,
            ratio_policy="balanced=0:0.5",
        )
        return {
            "status": "succeeded",
            "output_text": " generated",
            "error_type": None,
            "error": None,
        }

    report = probe.build_probe_report(
        args,
        load_fn=load_plugin,
        generation_fn=generate,
    )

    assert report["patch_generate_requested"] is True
    assert report["patch_generate_installed"] is True
    assert report["events_written"] == 1
    assert report["validation_passed"] is True
    assert report["phase12_6c_internal_hook_candidate"] is True
    assert report["active_routing"] is False
    assert report["measured_runtime_reduction"] is False
