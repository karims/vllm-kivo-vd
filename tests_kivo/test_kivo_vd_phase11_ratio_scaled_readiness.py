# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "check_phase11_ratio_scaled_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase11_ratio_scaled_readiness",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    ratio_name: str = "balanced",
    layer_map: str = "0:35,5:27,8:27,11:35",
    policy: str = "query_key_block_score",
    target: int = 960,
    actual: float = 917.0,
    max_new_tokens: int = 32,
    exact: float = 1.0,
    token_match: float = 1.0,
    edit_distance: float = 0.0,
    kl: float = 0.001344,
    selected_ratio: float = 0.527726,
) -> dict:
    return {
        "status": "succeeded",
        "ratio_policy_name": ratio_name,
        "ratio_policy_spec": "0:0.6,5:0.45,8:0.45,11:0.6",
        "derived_layer_budget_map": layer_map,
        "target_token_length": target,
        "average_actual_prompt_tokens": actual,
        "estimated_context_blocks": 58,
        "policy": policy,
        "max_new_tokens": max_new_tokens,
        "exact_sequence_match_rate": exact,
        "average_token_match_rate": token_match,
        "average_normalized_edit_distance": edit_distance,
        "average_per_step_kl_divergence": kl,
        "average_per_step_top1_match_rate": token_match,
        "average_selected_block_ratio_across_patched_layers": selected_ratio,
        "estimated_active_block_reduction_ratio": 1.0 - selected_ratio,
    }


def _write_summary(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({
            "policy_ratio_length_token": rows,
            "caveats": {
                "outside_vllm": True,
                "no_vllm_integration": True,
                "measured_runtime_reduction": False,
            },
        }),
        encoding="utf-8",
    )
    return path


def test_ready_for_clean_ratio_scaled_query_key_run(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = _write_summary(tmp_path, [_row()])

    report = module.build_readiness_report([path])

    assert report["phase11_7_ready"] is True
    assert report["phase12_ready"] is False
    assert report["best_deployable_tradeoff"]["ratio_policy_name"] == (
        "balanced"
    )


def test_not_ready_when_no_query_key_run_passes(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_summary(
        tmp_path,
        [
            _row(
                exact=0.5,
                token_match=0.5,
                edit_distance=0.5,
                kl=7.0,
            ),
            _row(policy="oracle_topk"),
        ],
    )

    report = module.build_readiness_report([path])

    assert report["phase11_7_ready"] is False
    assert report["checks"]["passing_query_key_run_exists"] is False


def test_best_tradeoff_prefers_reduction_and_excludes_oracle(
    tmp_path: Path,
) -> None:
    module = _load_module()
    rows = [
        _row(),
        _row(
            ratio_name="safer",
            layer_map="0:41,5:32,8:32,11:41",
            max_new_tokens=16,
            kl=0.000408,
            selected_ratio=0.626644,
        ),
        _row(
            ratio_name="oracle",
            policy="oracle_topk",
            selected_ratio=0.2,
        ),
    ]
    path = _write_summary(tmp_path, rows)

    report = module.build_readiness_report([path])

    assert report["best_deployable_tradeoff"]["ratio_policy_name"] == (
        "balanced"
    )
    assert report["best_deployable_tradeoff"][
        "estimated_active_block_reduction_ratio"
    ] == pytest.approx(0.472274)


def test_safest_config_prefers_lowest_kl(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_summary(
        tmp_path,
        [
            _row(),
            _row(
                ratio_name="safer",
                layer_map="0:41,5:32,8:32,11:41",
                max_new_tokens=16,
                kl=0.000408,
                selected_ratio=0.626644,
            ),
        ],
    )

    report = module.build_readiness_report([path])

    assert report["safest_passing_deployable_config"][
        "ratio_policy_name"
    ] == "safer"


def test_warnings_capture_aggressive_and_balanced_failures(
    tmp_path: Path,
) -> None:
    module = _load_module()
    failed_aggressive = _row(
        ratio_name="aggressive",
        layer_map="0:23,5:19,8:19,11:23",
        target=768,
        actual=734.0,
        exact=0.5,
        token_match=0.5,
        edit_distance=0.5,
        kl=8.0,
        selected_ratio=0.45,
    )
    aggressive_oracle = _row(
        ratio_name="aggressive",
        layer_map="0:23,5:19,8:19,11:23",
        policy="oracle_topk",
        target=768,
        actual=734.0,
        selected_ratio=0.45,
    )
    failed_balanced = _row(
        target=768,
        actual=734.0,
        exact=0.5,
        token_match=0.5,
        edit_distance=0.5,
        kl=7.979269,
        selected_ratio=0.523401,
    )
    balanced_oracle = _row(
        policy="oracle_topk",
        target=768,
        actual=734.0,
        selected_ratio=0.523401,
    )
    safer = _row(
        ratio_name="safer",
        layer_map="0:41,5:32,8:32,11:41",
        max_new_tokens=16,
        kl=0.000408,
        selected_ratio=0.626644,
    )
    path = _write_summary(
        tmp_path,
        [
            failed_aggressive,
            aggressive_oracle,
            failed_balanced,
            balanced_oracle,
            safer,
        ],
    )

    report = module.build_readiness_report([path])
    warnings = " ".join(report["warnings"])

    assert "aggressive query-key ratio policy failed" in warnings
    assert "balanced query-key ratio policy failed at target 768" in warnings
    assert "selector margin risk" in warnings
    assert "safer currently looks like the reliable" in warnings


def test_jsonl_input_phase12_and_caveats(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    report = module.build_readiness_report([path])

    assert report["phase11_7_ready"] is True
    assert report["phase12_ready"] is False
    assert report["input_summaries"][0]["caveats_available"] is False
    assert report["caveats"]["outside_vllm"] is True
    assert report["caveats"]["no_vllm_integration"] is True
    assert report["caveats"]["measured_runtime_reduction"] is False

    markdown = module.render_markdown(report)
    assert "No vLLM integration or active routing" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "Generation quality preservation is not claimed" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "check_phase11_ratio_scaled_readiness.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--inputs",
        "--output-json",
        "--output-md",
        "--min-exact-match-rate",
        "--min-token-match-rate",
        "--max-normalized-edit-distance",
        "--max-average-kl",
        "--max-selected-ratio-for-tradeoff",
    ):
        assert flag in process.stdout
