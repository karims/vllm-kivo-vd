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
        / "check_phase11_multilayer_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_phase11_multilayer_readiness",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_result(
    tmp_path: Path,
    *,
    name: str,
    policy: str,
    layer_map: dict[str, int],
    exact: float = 1.0,
    token_match: float = 1.0,
    edit_distance: float = 0.0,
    kl: float = 0.001,
) -> Path:
    path = tmp_path / f"{name}.json"
    payload = {
        "config": {
            "selection_policy": policy,
            "max_new_tokens": 32,
        },
        "layer_budget_map": layer_map,
        "aggregate": {
            "exact_sequence_match_rate": exact,
            "average_token_match_rate": token_match,
            "average_normalized_edit_distance": edit_distance,
            "average_per_step_kl_divergence": kl,
            "average_per_step_top1_match_rate": token_match,
            "average_selected_block_ratio_across_patched_layers": 0.39,
        },
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "no_measured_runtime_reduction": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ready_when_adaptive_query_key_run_is_clean(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = _write_result(
        tmp_path,
        name="adaptive-query",
        policy="query_key_block_score",
        layer_map={"0": 12, "5": 8, "8": 8, "11": 12},
    )

    report = module.build_readiness_report([path])

    assert report["phase11_4_ready"] is True
    assert all(report["checks"].values())


def test_not_ready_without_adaptive_query_key_run(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_result(
        tmp_path,
        name="adaptive-oracle",
        policy="oracle_topk",
        layer_map={"0": 12, "5": 8, "8": 8, "11": 12},
    )

    report = module.build_readiness_report([path])

    assert report["phase11_4_ready"] is False
    assert report["checks"]["adaptive_query_key_run_exists"] is False


def test_not_ready_when_exact_match_below_threshold(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = _write_result(
        tmp_path,
        name="adaptive-query-failed",
        policy="query_key_block_score",
        layer_map={"0": 12, "5": 8, "8": 8, "11": 12},
        exact=0.8,
        token_match=0.9,
        edit_distance=0.1,
    )

    report = module.build_readiness_report([path])

    assert report["phase11_4_ready"] is False
    assert report["checks"]["exact_match_threshold_met"] is False


def test_warns_on_naive_query_key_failure_with_oracle_pass(
    tmp_path: Path,
) -> None:
    module = _load_module()
    naive_map = {"5": 8, "8": 8, "11": 8}
    paths = [
        _write_result(
            tmp_path,
            name="naive-query",
            policy="query_key_block_score",
            layer_map=naive_map,
            exact=0.8,
            token_match=0.8125,
            edit_distance=0.1625,
            kl=1.2,
        ),
        _write_result(
            tmp_path,
            name="naive-oracle",
            policy="oracle_topk",
            layer_map=naive_map,
        ),
        _write_result(
            tmp_path,
            name="adaptive-query",
            policy="query_key_block_score",
            layer_map={"0": 12, "5": 8, "8": 8, "11": 12},
        ),
    ]

    report = module.build_readiness_report(paths)

    assert report["phase11_4_ready"] is True
    assert any("naive map" in warning for warning in report["warnings"])
    assert any(
        "selector/accumulation" in warning
        for warning in report["warnings"]
    )


def test_recommended_map_and_caveats_are_preserved(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = _write_result(
        tmp_path,
        name="adaptive-query",
        policy="query_key_block_score",
        layer_map={"0": 12, "5": 8, "8": 8, "11": 12},
    )
    report = module.build_readiness_report([path])

    assert report["recommended_adaptive_layer_budget_map"] == {
        "0": 12,
        "5": 8,
        "8": 8,
        "11": 12,
    }
    assert report["caveats"]["outside_vllm"] is True
    assert report["caveats"]["no_vllm_integration"] is True
    assert report["caveats"]["measured_runtime_reduction"] is False

    markdown = module.render_markdown(report)
    assert "No vLLM integration" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "Generation quality preservation is not claimed" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "check_phase11_multilayer_readiness.py"
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
    ):
        assert flag in process.stdout
