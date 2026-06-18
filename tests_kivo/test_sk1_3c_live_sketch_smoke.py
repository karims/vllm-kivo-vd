# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "run_sk1_3c_live_sketch_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_sk1_3c_live_sketch_smoke",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "out.json"])
    assert args.model == "facebook/opt-125m"
    assert args.local_files_only is False
    assert args.sketch_dim == 16
    assert args.runtime_policy == "recent_only"
    assert args.max_full_blocks == 2
    assert args.counter_export_file is None


def test_parse_args_local_files_only_flag() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "out.json", "--local-files-only"])
    assert args.local_files_only is True


def test_parse_args_counter_export_and_max_full_blocks() -> None:
    module = _load_module()
    args = module.parse_args(
        [
            "--output",
            "out.json",
            "--counter-export-file",
            "/tmp/counters.json",
            "--max-full-blocks",
            "5",
        ]
    )
    assert args.counter_export_file == "/tmp/counters.json"
    assert args.max_full_blocks == 5


def test_resolve_model_reference_prefers_existing_local_path(tmp_path: Path) -> None:
    module = _load_module()
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    resolved, is_local, candidates = module.resolve_model_reference(
        str(model_dir),
        local_files_only=True,
    )
    assert resolved == str(model_dir)
    assert is_local is True
    assert candidates == []


def test_resolve_model_reference_uses_cached_snapshot_when_local_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    cache_root = tmp_path / "hub"
    snapshot = (
        cache_root
        / "models--sshleifer--tiny-gpt2"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(module, "HF_CACHE_ROOT", cache_root)

    resolved, is_local, candidates = module.resolve_model_reference(
        "sshleifer/tiny-gpt2",
        local_files_only=True,
    )
    assert resolved == str(snapshot)
    assert is_local is True
    assert candidates == [str(snapshot)]


def test_resolve_model_reference_fails_cleanly_without_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "HF_CACHE_ROOT", tmp_path / "missing-cache")

    try:
        module.resolve_model_reference(
            "missing/model",
            local_files_only=True,
        )
    except FileNotFoundError as exc:
        assert "No cached local snapshot found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing local cache")


def test_smoke_env_sets_offline_flags_when_requested() -> None:
    module = _load_module()
    args = module.parse_args(["--output", "out.json", "--local-files-only"])
    env = module._smoke_env(args)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_smoke_env_uses_explicit_counter_export_and_max_full_blocks() -> None:
    module = _load_module()
    args = module.parse_args(
        [
            "--output",
            "out.json",
            "--counter-export-file",
            "/tmp/custom-counters.json",
            "--max-full-blocks",
            "7",
        ]
    )
    env = module._smoke_env(args)
    assert env["KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE"] == "/tmp/custom-counters.json"
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS"] == "7"


def test_summarize_sketch_counters_success_case() -> None:
    module = _load_module()
    summary = module.summarize_sketch_counters(
        {
            "sketch_backend": "random_projection",
            "sketch_build_attempted": 4,
            "sketch_build_succeeded": 3,
            "sketch_build_failed": 1,
            "sketched_blocks_total": 3,
            "sketch_missing_prevented_demotion": 1,
            "sketch_missing_prevented_free": 0,
            "freed_after_sketch_blocks_total": 2,
            "ownership_remove_invariant_failed": 0,
            "free_to_pool_double_free_prevented": 0,
            "block_pool_free_accounting_rejected": 0,
        }
    )
    assert summary["sketch_backend"] == "random_projection"
    assert summary["sketch_build_attempted"] == 4
    assert summary["sketch_build_succeeded"] == 3
    assert summary["freed_after_sketch_blocks_total"] == 2
    assert summary["invariants_clean"] is True
    assert summary["warnings"] == []


def test_summarize_sketch_counters_warns_on_attempt_without_success() -> None:
    module = _load_module()
    summary = module.summarize_sketch_counters(
        {
            "sketch_backend": "random_projection",
            "sketch_build_attempted": 2,
            "sketch_build_succeeded": 0,
            "sketch_build_failed": 2,
        }
    )
    assert summary["sketch_build_attempted"] == 2
    assert summary["sketch_build_succeeded"] == 0
    assert any("kv_cache shape extraction issue" in warning for warning in summary["warnings"])
