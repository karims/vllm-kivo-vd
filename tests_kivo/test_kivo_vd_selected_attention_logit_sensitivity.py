# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_selected_attention_logit_sensitivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_selected_attention_logit_sensitivity",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topk_overlap_counts_shared_ids() -> None:
    module = _load_module()
    baseline = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    patched = torch.tensor([5.0, 1.0, 4.0, 3.0, 2.0])

    assert module.topk_overlap(baseline, patched, 3) == 2
    assert module.topk_overlap(baseline, patched, 10) == 5


def test_kl_divergence_is_stable_and_zero_for_equal_logits() -> None:
    module = _load_module()
    logits = torch.tensor([1000.0, 999.0, -1000.0])
    shifted = torch.tensor([999.0, 1000.0, -1000.0])

    assert module.kl_divergence_from_logits(logits, logits) == pytest.approx(
        0.0
    )
    divergence = module.kl_divergence_from_logits(logits, shifted)
    assert divergence > 0
    assert torch.isfinite(torch.tensor(divergence))


def test_logits_comparison_metric_schema() -> None:
    module = _load_module()
    baseline = torch.tensor([3.0, 2.0, 1.0, 0.0])
    patched = torch.tensor([2.5, 2.2, 1.0, 0.0])

    metrics = module.compare_logits(baseline, patched)

    for field in (
        "logits_cosine_similarity",
        "logits_relative_l2_error",
        "kl_divergence",
        "top1_token_match",
        "top5_overlap_count",
        "top10_overlap_count",
        "baseline_top_token_id",
        "patched_top_token_id",
        "baseline_top_token_probability",
        "patched_top_token_probability",
        "baseline_top_token_probability_after_patch",
        "baseline_top_token_probability_delta",
    ):
        assert field in metrics


def test_report_has_required_sections_and_caveats() -> None:
    module = _load_module()
    row = {
        "logits_cosine_similarity": 0.99,
        "logits_relative_l2_error": 0.05,
        "kl_divergence": 0.001,
        "top1_token_match": True,
        "top5_overlap_count": 5,
        "top10_overlap_count": 10,
        "attention_output_cosine_similarity": 0.98,
        "attention_output_relative_l2_error": 0.10,
    }

    report = module.build_report(
        config={"model": "gpt2"},
        rows=[row],
    )

    assert set(report) == {"config", "aggregate", "per_prompt", "caveats"}
    assert report["aggregate"]["num_prompts"] == 1
    assert report["per_prompt"] == [row]
    assert report["caveats"]["outside_vllm"] is True
    assert report["caveats"]["no_vllm_integration"] is True
    assert report["caveats"]["single_layer_patch_only"] is True


def test_tiny_random_gpt2_patch_smoke() -> None:
    transformers = pytest.importorskip("transformers")
    module = _load_module()
    helpers = module._load_selected_attention_helpers()
    config = transformers.GPT2Config(
        vocab_size=97,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    input_ids = torch.randint(0, 97, (1, 12))

    patched, info = module.patched_next_token_logits(
        model=model,
        input_ids=input_ids,
        layer_idx=0,
        block_size=4,
        candidate_budget_blocks=2,
        selection_policy="query_key_block_score",
        sketch_dim=8,
        block_score_reduction="max",
        seed=0,
        helpers=helpers,
    )

    assert patched.shape == (1, 97)
    assert len(info["selected_block_ids"]) == 2


def test_markdown_contains_required_caveats() -> None:
    module = _load_module()
    report = {
        "config": {"model": "gpt2"},
        "aggregate": {"num_prompts": 1},
        "per_prompt": [{
            "prompt_index": 0,
            "token_length": 32,
            "selected_block_count": 2,
            "attention_output_cosine_similarity": 0.98,
            "attention_output_relative_l2_error": 0.10,
            "logits_cosine_similarity": 0.99,
            "logits_relative_l2_error": 0.05,
            "kl_divergence": 0.001,
            "top1_token_match": True,
            "top5_overlap_count": 5,
            "top10_overlap_count": 10,
            "baseline_top_token": "A",
            "patched_top_token": "A",
        }],
        "caveats": {},
    }

    markdown = module.render_markdown(report)

    assert "outside vLLM" in markdown
    assert "No vLLM integration" in markdown
    assert "Only one layer" in markdown
    assert "No measured runtime memory reduction" in markdown
    assert "No latency claim" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_selected_attention_logit_sensitivity.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--prompt",
        "--prompts-file",
        "--layer-idx",
        "--block-size",
        "--candidate-budget-blocks",
        "--selection-policy",
        "--sketch-dim",
        "--block-score-reduction",
        "--max-length",
        "--dtype",
        "--device",
        "--seed",
        "--output-json",
        "--output-md",
    ):
        assert flag in process.stdout
