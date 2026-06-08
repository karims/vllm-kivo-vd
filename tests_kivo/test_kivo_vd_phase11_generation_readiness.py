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
        / "check_phase11_generation_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase11_generation_readiness",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_result(
    tmp_path: Path,
    *,
    layer: int,
    budget: int,
    policy: str,
    exact: float = 1.0,
    token_match: float = 1.0,
    kl: float = 0.001,
) -> Path:
    path = tmp_path / f"{policy}_layer{layer}_budget{budget}.json"
    payload = {
        "config": {
            "layer_index": layer,
            "candidate_budget_blocks": budget,
            "selection_policy": policy,
            "max_new_tokens": 16,
        },
        "aggregate": {
            "exact_sequence_match_rate": exact,
            "average_token_match_rate": token_match,
            "average_prefix_match_length": 16.0,
            "average_normalized_edit_distance": 0.0,
            "average_per_step_kl_divergence": kl,
            "average_per_step_top1_match_rate": token_match,
            "average_selected_block_ratio": 0.5,
        },
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "generation_quality_probe_only": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_clean_budget16_results_pass(tmp_path: Path) -> None:
    module = _load_module()
    paths = [
        _write_result(tmp_path, layer=0, budget=16, policy=policy)
        for policy in ("query_key_block_score", "oracle_topk")
    ]

    report = module.build_readiness_report(paths)

    assert report["phase11_3_ready"] is True
    assert report["adaptive_budget_map"]["0"][
        "minimum_clean_observed_budget"
    ] == 16
    assert report["adaptive_budget_map"]["0"][
        "safer_recommended_budget"
    ] == 16


def test_budget8_layer0_divergence_creates_warning(tmp_path: Path) -> None:
    module = _load_module()
    paths = [
        _write_result(
            tmp_path,
            layer=0,
            budget=8,
            policy=policy,
            exact=0.8,
            token_match=0.9,
            kl=0.49,
        )
        for policy in ("query_key_block_score", "oracle_topk")
    ]

    report = module.build_readiness_report(paths)

    assert report["phase11_3_ready"] is False
    assert report["any_layer_budget_divergence"] is True
    assert report["adaptive_budget_map"]["0"]["divergent_budgets"] == [8]
    assert any("layer 0 budget 8" in item for item in report["warnings"])


def test_budget12_layer0_recovery_updates_recommendation(
    tmp_path: Path,
) -> None:
    module = _load_module()
    paths = []
    for policy in ("query_key_block_score", "oracle_topk"):
        paths.append(_write_result(
            tmp_path,
            layer=0,
            budget=8,
            policy=policy,
            exact=0.8,
            token_match=0.9,
            kl=0.49,
        ))
        paths.append(_write_result(
            tmp_path,
            layer=0,
            budget=12,
            policy=policy,
            exact=1.0,
            token_match=1.0,
            kl=0.001,
        ))

    report = module.build_readiness_report(paths)

    assert report["phase11_3_ready"] is True
    assert report["any_layer_budget_divergence"] is True
    assert report["adaptive_budget_map"]["0"][
        "minimum_clean_observed_budget"
    ] == 12
    assert report["adaptive_budget_map"]["0"][
        "safer_recommended_budget"
    ] == 16


def test_markdown_preserves_caveats(tmp_path: Path) -> None:
    module = _load_module()
    paths = [
        _write_result(tmp_path, layer=0, budget=16, policy=policy)
        for policy in ("query_key_block_score", "oracle_topk")
    ]
    report = module.build_readiness_report(paths)

    markdown = module.render_markdown(report)

    assert "outside vLLM" in markdown
    assert "No vLLM integration" in markdown
    assert "No active routing" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No latency improvement" in markdown
    assert "Generation quality preservation is not claimed" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "check_phase11_generation_readiness.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in ("--inputs", "--output-json", "--output-md"):
        assert flag in process.stdout
