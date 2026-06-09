# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _plugin_module(name: str):
    plugin_root = _repo_root() / "plugins" / "kivo_vllm_shadow_plugin"
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    return importlib.import_module(f"kivo_vllm_shadow_plugin.{name}")


def _load_plugin():
    return importlib.reload(_plugin_module("plugin"))


def _load_validator():
    path = (
        _repo_root()
        / "scripts"
        / "kivo_vd"
        / "validate_phase12_6d_kv_observation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase12_6d_kv_observation_validator_test",
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
        "phase12_6d_plugin_probe_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_kv_module(result):
    class FakeKVCacheManager:
        block_size = 16
        enable_caching = True

        def get_block_ids(self, request_id: str):
            return result

    return SimpleNamespace(KVCacheManager=FakeKVCacheManager)


def test_default_register_does_not_patch_kv_method(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    result = ([1, 2],)
    kv_module = _fake_kv_module(result)
    original = kv_module.KVCacheManager.get_block_ids
    fake_vllm = SimpleNamespace(
        __version__="0.test",
        __file__="/site-packages/vllm/__init__.py",
    )
    marker = tmp_path / "marker.json"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv(plugin.MARKER_ENV, str(marker))
    monkeypatch.delenv(plugin.PATCH_KV_GET_BLOCK_IDS_ENV, raising=False)

    plugin.register()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert kv_module.KVCacheManager.get_block_ids is original
    assert payload["patch_kv_get_block_ids_requested"] is False
    assert payload["patch_kv_get_block_ids_installed"] is False


def test_kv_patch_installs_once_and_returns_exact_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    result = ([1, 2], [3])
    kv_module = _fake_kv_module(result)
    observations: list[dict] = []

    def observer(**kwargs):
        observations.append(kwargs)
        return {}

    monkeypatch.setenv(
        plugin.KV_OBSERVATIONS_ENV,
        str(tmp_path / "observations.jsonl"),
    )
    installed, _ = plugin.install_kv_get_block_ids_patch(
        import_module=lambda name: kv_module,
        observer=observer,
    )
    wrapper = kv_module.KVCacheManager.get_block_ids
    installed_again, _ = plugin.install_kv_get_block_ids_patch(
        import_module=lambda name: kv_module,
        observer=observer,
    )
    manager = kv_module.KVCacheManager()
    returned = manager.get_block_ids("request-1")

    assert installed is True
    assert installed_again is True
    assert kv_module.KVCacheManager.get_block_ids is wrapper
    assert returned is result
    assert observations[0]["result"] is result
    assert observations[0]["args"] == ("request-1",)


def test_kv_patch_catches_observer_exception(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    result = ([4, 5],)
    kv_module = _fake_kv_module(result)
    marker = tmp_path / "marker.json"
    monkeypatch.setenv(
        plugin.KV_OBSERVATIONS_ENV,
        str(tmp_path / "observations.jsonl"),
    )
    monkeypatch.setenv(plugin.MARKER_ENV, str(marker))
    plugin.write_load_marker(marker)

    def broken_observer(**kwargs):
        raise RuntimeError("observer failed")

    plugin.install_kv_get_block_ids_patch(
        import_module=lambda name: kv_module,
        observer=broken_observer,
    )
    returned = kv_module.KVCacheManager().get_block_ids("request-2")
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert returned is result
    assert "observer failed" in payload["runtime_warnings"][0]


def test_observation_writer_flattens_and_truncates(tmp_path: Path) -> None:
    observation_module = _plugin_module("kv_observation")

    class TensorLike:
        def tolist(self):
            return [[9, 10], [11, 12]]

    instance = SimpleNamespace(block_size=16, enable_caching=True)
    observation = observation_module.build_kv_observation(
        instance=instance,
        args=("request-3",),
        kwargs={"extra": True},
        result=([1, 2, 3], TensorLike()),
        preview_limit=4,
    )

    assert observation["block_ids_preview"] == [1, 2, 3, 9]
    assert observation["block_id_count"] == 7
    assert observation["block_ids_preview_truncated"] is True
    assert observation["min_block_id"] == 1
    assert observation["max_block_id"] == 12
    assert len(observation["result_repr_preview"]) <= 320
    assert observation["active_routing"] is False
    assert observation["mutation"] is False


def test_validator_accepts_valid_record_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    observation_module = _plugin_module("kv_observation")
    validator = _load_validator()
    path = tmp_path / "observations.jsonl"
    observation = observation_module.build_kv_observation(
        instance=SimpleNamespace(block_size=16),
        args=("request-4",),
        kwargs={},
        result=([2, 5, 8],),
    )
    observation_module.append_kv_observation(path, observation)

    valid = validator.validate_observations(
        validator.load_observations(path)
    )
    assert valid["validation_passed"] is True

    invalid_observation = dict(observation)
    invalid_observation["mutation"] = True
    invalid_observation["active_routing"] = True
    invalid = validator.validate_observations([invalid_observation])
    assert invalid["validation_passed"] is False
    assert invalid["invalid_records"] == 1


def test_register_records_kv_patch_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = _load_plugin()
    kv_module = _fake_kv_module(([1],))
    fake_vllm = SimpleNamespace(
        __version__="0.test",
        __file__="/site-packages/vllm/__init__.py",
    )
    original_import = plugin.importlib.import_module

    def import_module(name: str):
        if name == "vllm.v1.core.kv_cache_manager":
            return kv_module
        return original_import(name)

    marker = tmp_path / "marker.json"
    observations = tmp_path / "observations.jsonl"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(plugin.importlib, "import_module", import_module)
    monkeypatch.setenv(plugin.MARKER_ENV, str(marker))
    monkeypatch.setenv(plugin.PATCH_KV_GET_BLOCK_IDS_ENV, "1")
    monkeypatch.setenv(plugin.KV_OBSERVATIONS_ENV, str(observations))

    plugin.register()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["patch_kv_get_block_ids_requested"] is True
    assert payload["patch_kv_get_block_ids_installed"] is True
    assert payload["kv_observations_path"] == str(observations)


def test_probe_counts_fake_kv_observations(tmp_path: Path) -> None:
    plugin = _load_plugin()
    observation_module = _plugin_module("kv_observation")
    probe = _load_probe()
    args = probe._parse_args([
        "--enable-kv-get-block-ids-hook",
        "--require-kv-observations",
        "--marker-path",
        str(tmp_path / "marker.json"),
        "--kv-observations-jsonl",
        str(tmp_path / "observations.jsonl"),
        "--output-json",
        str(tmp_path / "report.json"),
        "--output-md",
        str(tmp_path / "report.md"),
    ])

    def load_plugin():
        plugin.write_load_marker(
            args.marker_path,
            plugin.build_plugin_state(
                patch_kv_get_block_ids_requested=True,
                patch_kv_get_block_ids_installed=True,
                kv_get_block_ids_original_qualname=(
                    "FakeKVCacheManager.get_block_ids"
                ),
                kv_observations_path=args.kv_observations_jsonl,
            ),
        )
        return {
            "vllm_version": "0.22.1",
            "vllm_file": "/site-packages/vllm/__init__.py",
        }

    def generate(_args):
        observation = observation_module.build_kv_observation(
            instance=SimpleNamespace(block_size=16),
            args=("request-5",),
            kwargs={},
            result=([1, 4],),
        )
        observation_module.append_kv_observation(
            args.kv_observations_jsonl,
            observation,
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

    assert report["patch_kv_get_block_ids_installed"] is True
    assert report["kv_observations_written"] == 1
    assert report["kv_observation_requirement_satisfied"] is True
    assert report["phase12_7_active_experiment_candidate"] is True
    assert report["runtime_behavior_changed"] is False


def test_probe_cli_accepts_kv_observation_options() -> None:
    probe = _load_probe()
    args = probe._parse_args([
        "--enable-kv-get-block-ids-hook",
        "--kv-observations-jsonl",
        "/tmp/kv.jsonl",
        "--require-kv-observations",
    ])

    assert args.enable_kv_get_block_ids_hook is True
    assert args.kv_observations_jsonl == "/tmp/kv.jsonl"
    assert args.require_kv_observations is True
