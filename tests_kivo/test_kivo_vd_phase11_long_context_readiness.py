# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "check_phase11_long_context_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase11_long_context_readiness",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    policy: str = "query_key_block_score",
    layer_map: str = "0:32,5:24,8:24,11:32",
    target: int = 960,
    actual: float = 917.0,
    exact: float = 1.0,
    token_match: float = 1.0,
    edit_distance: float = 0.0,
    kl: float = 0.002257,
    selected_ratio: float = 0.480713,
) -> dict:
    return {
        "status": "succeeded",
        "policy": policy,
        "layer_budget_map": layer_map,
        "target_token_length": target,
        "average_actual_prompt_token_length": actual,
        "max_new_tokens": 16,
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
            "policy_length_map_tokens": rows,
            "caveats": {
                "outside_vllm": True,
                "no_vllm_integration": True,
                "measured_runtime_reduction": False,
            },
        }),
        encoding="utf-8",
    )
    return path


def test_ready_for_clean_long_context_query_key_run(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = _write_summary(tmp_path, [_row()])

    report = module.build_readiness_report([path])

    assert report["phase11_6_ready"] is True
    assert report["phase12_ready"] is False
    assert report["best_tradeoff_config"]["layer_budget_map"] == (
        "0:32,5:24,8:24,11:32"
    )


def test_not_ready_without_long_context_run(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_summary(
        tmp_path,
        [_row(target=512, actual=500.0)],
    )

    report = module.build_readiness_report([path])

    assert report["phase11_6_ready"] is False
    assert report["checks"]["long_context_query_key_run_exists"] is False


def test_not_ready_when_exact_match_fails(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_summary(
        tmp_path,
        [_row(exact=0.5, token_match=0.5, edit_distance=0.3, kl=7.0)],
    )

    report = module.build_readiness_report([path])

    assert report["phase11_6_ready"] is False
    assert report["checks"]["exact_match_threshold_met"] is False


def test_best_tradeoff_differs_from_lowest_kl(tmp_path: Path) -> None:
    module = _load_module()
    tradeoff = _row(
        layer_map="0:32,5:24,8:24,11:32",
        kl=0.002257,
        selected_ratio=0.480713,
    )
    safest = _row(
        layer_map="0:48,5:40,8:40,11:48",
        kl=0.000122,
        selected_ratio=0.755406,
    )
    path = _write_summary(tmp_path, [tradeoff, safest])

    report = module.build_readiness_report([path])

    assert report["safest_passing_config"]["layer_budget_map"] == (
        "0:48,5:40,8:40,11:48"
    )
    assert report["best_tradeoff_config"]["layer_budget_map"] == (
        "0:32,5:24,8:24,11:32"
    )


def test_warnings_capture_aggressive_budget_and_selector_margin(
    tmp_path: Path,
) -> None:
    module = _load_module()
    failed_query = _row(
        layer_map="0:12,5:8,8:8,11:12",
        target=768,
        actual=734.0,
        exact=0.0,
        token_match=0.125,
        edit_distance=0.796875,
        kl=11.861486,
        selected_ratio=0.211399,
    )
    failed_oracle = _row(
        policy="oracle_topk",
        layer_map="0:12,5:8,8:8,11:12",
        target=768,
        actual=734.0,
        exact=0.5,
        token_match=0.5,
        edit_distance=0.34375,
        kl=8.179362,
        selected_ratio=0.211399,
    )
    selector_failure = _row(
        layer_map="0:24,5:24,8:24,11:24",
        target=768,
        actual=734.0,
        exact=0.5,
        token_match=0.5,
        edit_distance=0.34375,
        kl=7.25,
        selected_ratio=0.5,
    )
    selector_oracle = _row(
        policy="oracle_topk",
        layer_map="0:24,5:24,8:24,11:24",
        target=768,
        actual=734.0,
        selected_ratio=0.5,
    )
    path = _write_summary(
        tmp_path,
        [failed_query, failed_oracle, selector_failure, selector_oracle, _row()],
    )

    report = module.build_readiness_report([path])
    warnings = " ".join(report["warnings"])

    assert "short-context map" in warnings
    assert "0.18-0.21" in warnings
    assert "budget/risk issue" in warnings
    assert "selector margin risk" in warnings


def test_jsonl_input_and_output_caveats_are_supported(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    report = module.build_readiness_report([path])

    assert report["phase11_6_ready"] is True
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
        / "check_phase11_long_context_readiness.py"
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
