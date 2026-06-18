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
    assert args.sketch_dim == 16
    assert args.runtime_policy == "recent_only"


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
