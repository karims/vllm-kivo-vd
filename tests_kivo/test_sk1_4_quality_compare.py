# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "kivo_vd" / "run_sk1_4_quality_compare.py"
    spec = importlib.util.spec_from_file_location(
        "run_sk1_4_quality_compare",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults() -> None:
    module = _load_module()
    args = module.parse_args([])
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.output == "/tmp/sk1_4_quality_compare.json"
    assert args.keep_recent_blocks == 2
    assert args.max_full_blocks == 2
    assert args.sketch_topk == 2
    assert args.sketch_dim == 16


def test_build_mode_env_baseline_only_has_counter_export() -> None:
    module = _load_module()
    args = module.parse_args([])
    env = module.build_mode_env(
        module.BASELINE_MODE,
        args=args,
        counter_export_file="/tmp/counters.json",
    )
    assert env["KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE"] == "/tmp/counters.json"
    assert "KIVO_KV_SKETCH_ENABLE" not in env
    assert "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE" not in env


def test_build_mode_env_random_projection_contains_sketch_and_free_flags() -> None:
    module = _load_module()
    args = module.parse_args(
        ["--sketch-dim", "32", "--max-full-blocks", "4", "--sketch-topk", "3"]
    )
    env = module.build_mode_env(
        module.RANDOM_PROJECTION_MODE,
        args=args,
        counter_export_file="/tmp/counters.json",
    )
    assert env["KIVO_KV_SKETCH_ENABLE"] == "1"
    assert env["KIVO_KV_SKETCH_BACKEND"] == "random_projection"
    assert env["KIVO_KV_SKETCH_DIM"] == "32"
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS"] == "4"
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_TOPK"] == "3"
    assert env["KIVO_KV_FREE_TO_POOL_ENABLE"] == "1"


def test_build_mode_env_sketch_topk_sets_policy_and_sketch_flags() -> None:
    module = _load_module()
    args = module.parse_args(["--sketch-topk", "4"])
    env = module.build_mode_env(
        module.SKETCH_TOPK_MODE,
        args=args,
        counter_export_file="/tmp/counters.json",
    )
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY"] == "sketch_topk"
    assert env["KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_TOPK"] == "4"
    assert env["KIVO_KV_SKETCH_ENABLE"] == "1"
    assert env["KIVO_KV_SKETCH_BACKEND"] == "random_projection"


def test_build_quality_prompts_has_expected_schema() -> None:
    module = _load_module()
    prompts = module.build_quality_prompts(repeats=3)
    assert len(prompts) == 4
    assert {item["prompt_name"] for item in prompts} == {
        "factual_recall",
        "code_context",
        "summarization",
        "instruction_following",
    }
    assert all(item["prompt_text"] for item in prompts)


def test_build_mode_summary_warns_when_random_projection_has_zero_sketch_success():
    module = _load_module()
    summary = module.build_mode_summary(
        module.RANDOM_PROJECTION_MODE,
        [
            {
                "generation_success": True,
                "latency_seconds": 1.0,
                "counter_summary": {
                    "freed_after_sketch_blocks_total": 0,
                    "invariants_clean": True,
                    "sketch_build_succeeded": 0,
                },
            }
        ],
    )
    assert summary["random_projection_sketch_success_count"] == 0
    assert "random_projection_zero_sketch_success" in summary["warnings"]


def test_build_mode_summary_warns_on_invariant_failure() -> None:
    module = _load_module()
    summary = module.build_mode_summary(
        module.RECENT_ONLY_MODE,
        [
            {
                "generation_success": True,
                "latency_seconds": 2.0,
                "counter_summary": {
                    "freed_after_sketch_blocks_total": 3,
                    "invariants_clean": False,
                    "sketch_build_succeeded": 0,
                },
            }
        ],
    )
    assert summary["invariants_clean"] is False
    assert "invariants_not_clean" in summary["warnings"]


def test_summarize_quality_counters_includes_sketch_topk_fields() -> None:
    module = _load_module()
    summary = module.summarize_quality_counters(
        {
            "sketch_backend": "random_projection",
            "sketch_topk_old_blocks_considered": 9,
            "sketch_topk_extra_blocks_kept": 2,
            "sketch_topk_missing_scores": 1,
            "last_sketch_topk_keep_ids_sample": (21, 22),
        }
    )
    assert summary["sketch_topk_old_blocks_considered"] == 9
    assert summary["sketch_topk_extra_blocks_kept"] == 2
    assert summary["sketch_topk_missing_scores"] == 1
    assert summary["last_sketch_topk_keep_ids_sample"] == [21, 22]


def test_build_overall_summary_schema_basics() -> None:
    module = _load_module()
    overall = module.build_overall_summary(
        [
            {
                "mode": module.BASELINE_MODE,
                "summary": {
                    "success_count": 4,
                    "average_latency_seconds": 1.0,
                    "total_freed_after_sketch_blocks": 0,
                    "invariants_clean": True,
                    "random_projection_sketch_success_count": 0,
                },
            },
            {
                "mode": module.RECENT_ONLY_MODE,
                "summary": {
                    "success_count": 4,
                    "average_latency_seconds": 1.2,
                    "total_freed_after_sketch_blocks": 10,
                    "invariants_clean": True,
                    "random_projection_sketch_success_count": 0,
                },
            },
            {
                "mode": module.RANDOM_PROJECTION_MODE,
                "summary": {
                    "success_count": 4,
                    "average_latency_seconds": 1.3,
                    "total_freed_after_sketch_blocks": 12,
                    "invariants_clean": True,
                    "random_projection_sketch_success_count": 3,
                },
            },
            {
                "mode": module.SKETCH_TOPK_MODE,
                "summary": {
                    "success_count": 4,
                    "average_latency_seconds": 1.4,
                    "total_freed_after_sketch_blocks": 11,
                    "invariants_clean": True,
                    "random_projection_sketch_success_count": 4,
                },
            },
        ]
    )
    assert overall["per_mode_success_count"][module.BASELINE_MODE] == 4
    assert (
        overall["per_mode_total_freed_after_sketch_blocks"][
            module.RANDOM_PROJECTION_MODE
        ]
        == 12
    )
    assert overall["random_projection_sketch_success_count"] == 3
    assert overall["warnings"] == []
