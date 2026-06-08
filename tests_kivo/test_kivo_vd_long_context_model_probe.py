# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_long_context_model_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_long_context_model_probe",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_list_parsing() -> None:
    module = _load_module()

    assert module.parse_models("gpt2", None) == ["gpt2"]
    assert module.parse_models(
        "unused",
        "EleutherAI/pythia-160m,facebook/opt-125m,"
        "EleutherAI/pythia-160m",
    ) == ["EleutherAI/pythia-160m", "facebook/opt-125m"]


def test_target_length_and_ratio_policy_parsing() -> None:
    module = _load_module()

    assert module.parse_target_lengths("1024,1536") == [1024, 1536]
    name, ratios = module.parse_ratio_policy(
        "safer=0:0.70,5:0.55,8:0.55,11:0.70"
    )
    assert name == "safer"
    assert ratios == {0: 0.70, 5: 0.55, 8: 0.55, 11: 0.70}


def test_compatibility_metadata_for_fake_configs() -> None:
    module = _load_module()
    pythia = SimpleNamespace(
        model_type="gpt_neox",
        architectures=["GPTNeoXForCausalLM"],
        max_position_embeddings=2048,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
    )
    gpt2 = SimpleNamespace(
        model_type="gpt2",
        architectures=["GPT2LMHeadModel"],
        n_positions=1024,
        n_layer=12,
        n_embd=768,
        n_head=12,
    )

    pythia_metadata = module.compatibility_metadata(
        config=pythia,
        tokenizer=None,
        model=None,
    )
    gpt2_metadata = module.compatibility_metadata(
        config=gpt2,
        tokenizer=None,
        model=None,
    )

    assert pythia_metadata["architecture_family"] == "gpt_neox"
    assert pythia_metadata["max_context_estimate"] == 2048
    assert pythia_metadata["supported_for_selected_attention_eval"] is False
    assert "rotary-position" in pythia_metadata["reason_if_unsupported"]
    assert gpt2_metadata["architecture_family"] == "gpt2"
    assert gpt2_metadata["supported_for_selected_attention_eval"] is True
    assert gpt2_metadata["adapter_name"] == "gpt2_phase11_adapter"


def test_unsupported_model_row_has_reason() -> None:
    module = _load_module()
    opt = SimpleNamespace(
        model_type="opt",
        architectures=["OPTForCausalLM"],
        max_position_embeddings=2048,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
    )

    metadata = module.compatibility_metadata(
        config=opt,
        tokenizer=None,
        model=None,
    )

    assert metadata["supported_for_selected_attention_eval"] is False
    assert metadata["adapter_name"] == "unsupported"
    assert "OPT needs an adapter" in metadata["reason_if_unsupported"]


def test_summary_generation_from_fake_metadata() -> None:
    module = _load_module()
    supported = {
        "model": "gpt2",
        "model_class": "GPT2LMHeadModel",
        "architecture_family": "gpt2",
        "can_load_model": True,
        "can_tokenize_long_prompt": True,
        "max_context_estimate": 1024,
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "supported_for_selected_attention_eval": True,
        "adapter_name": "gpt2_phase11_adapter",
        "status": "supported",
        "reason_if_unsupported": None,
    }
    unsupported = {
        "model": "EleutherAI/pythia-160m",
        "model_class": "GPTNeoXForCausalLM",
        "architecture_family": "gpt_neox",
        "can_load_model": True,
        "can_tokenize_long_prompt": True,
        "max_context_estimate": 2048,
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "supported_for_selected_attention_eval": False,
        "adapter_name": "unsupported",
        "status": "unsupported",
        "reason_if_unsupported": "GPTNeoX adapter required",
    }

    summary = module.build_summary(
        [supported, unsupported],
        config={"models": ["gpt2", "EleutherAI/pythia-160m"]},
    )

    assert summary["counts"]["supported"] == 1
    assert summary["counts"]["unsupported"] == 1
    assert summary["recommended_next_model"] == "gpt2"
    assert summary["recommended_adapter_work"] == ["GPTNeoX adapter required"]


def test_dry_run_writes_planned_rows_without_model_download(
    tmp_path: Path,
) -> None:
    module = _load_module()
    args = module._parse_args([
        "--models",
        "EleutherAI/pythia-160m,facebook/opt-125m",
        "--target-token-lengths",
        "1024",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    ])

    result = module.run_probe(args)

    assert result["summary"]["counts"] == {
        "total": 2,
        "supported": 0,
        "unsupported": 0,
        "failed": 0,
        "planned": 2,
    }
    rows = [
        json.loads(line)
        for line in Path(result["rows_path"]).read_text().splitlines()
    ]
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["architecture_family"] for row in rows} == {
        "gpt_neox",
        "opt",
    }
    assert result["summary"]["recommended_next_model"] == (
        "EleutherAI/pythia-160m"
    )
    assert result["summary"]["ready_for_selected_attention_smoke"] is False


def test_markdown_has_supported_and_unsupported_tables() -> None:
    module = _load_module()
    rows = [
        {
            "model": "gpt2",
            "model_class": "GPT2LMHeadModel",
            "architecture_family": "gpt2",
            "can_load_model": True,
            "can_tokenize_long_prompt": True,
            "max_context_estimate": 1024,
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "supported_for_selected_attention_eval": True,
            "adapter_name": "gpt2_phase11_adapter",
            "status": "supported",
            "reason_if_unsupported": None,
        },
        {
            "model": "facebook/opt-125m",
            "model_class": "OPTForCausalLM",
            "architecture_family": "opt",
            "can_load_model": True,
            "can_tokenize_long_prompt": True,
            "max_context_estimate": 2048,
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "supported_for_selected_attention_eval": False,
            "adapter_name": "unsupported",
            "status": "unsupported",
            "reason_if_unsupported": "OPT adapter required",
        },
    ]
    summary = module.build_summary(rows, config={"models": []})

    markdown = module.render_markdown(summary)

    assert "Supported Models" in markdown
    assert "Unsupported Models" in markdown
    assert "GPT2LMHeadModel" in markdown
    assert "OPT adapter required" in markdown
    assert "No vLLM integration or active routing" in markdown
    assert "No measured runtime memory reduction" in markdown


def test_cli_help_includes_expected_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "kivo_vd"
        / "run_long_context_model_probe.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--model",
        "--models",
        "--target-token-lengths",
        "--num-prompts-per-length",
        "--max-new-tokens",
        "--block-size",
        "--ratio-policy",
        "--device",
        "--dtype",
        "--max-length",
        "--dry-run",
        "--output-dir",
        "--continue-on-error",
    ):
        assert flag in process.stdout
