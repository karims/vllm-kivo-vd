#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Probe longer-context HF models for offline selected-attention evaluation."""

import argparse
import gc
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "EleutherAI/pythia-160m"
KNOWN_CANDIDATES = (
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "facebook/opt-125m",
    "facebook/opt-350m",
)


def _load_script(filename: str, module_name: str) -> Any:
    module_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load helper script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ratio_module() -> Any:
    return _load_script(
        "run_ratio_scaled_long_context_sweep.py",
        "run_ratio_scaled_long_context_sweep",
    )


def _load_long_context_module() -> Any:
    return _load_script(
        "run_long_context_adaptive_generation_sweep.py",
        "run_long_context_adaptive_generation_sweep",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe longer-context HuggingFace causal LMs for compatibility "
            "with Kivo-VD standalone selected-attention evaluation."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--models")
    parser.add_argument("--target-token-lengths", default="1024,1536")
    parser.add_argument("--num-prompts-per-length", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--ratio-policy",
        default="safer=0:0.70,5:0.55,8:0.55,11:0.70",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="outputs/kivo_vd/phase11_7_long_context_model_probe",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def parse_models(model: str, models: str | None) -> list[str]:
    value = models if models is not None else model
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("at least one model must be provided")
    return list(dict.fromkeys(result))


def parse_target_lengths(value: str) -> list[int]:
    long_context = _load_long_context_module()
    return long_context.parse_target_token_lengths(value)


def parse_ratio_policy(value: str) -> tuple[str, dict[int, float]]:
    ratio_module = _load_ratio_module()
    policies = ratio_module.parse_ratio_policies(value)
    if len(policies) != 1:
        raise ValueError("--ratio-policy must contain exactly one policy")
    return next(iter(policies.items()))


def _positive_int(config: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def architecture_family(config: Any) -> str:
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(
        str(item).lower()
        for item in (getattr(config, "architectures", None) or [])
    )
    combined = f"{model_type} {architectures}"
    if "gpt_neox" in combined or "gptneox" in combined:
        return "gpt_neox"
    if "opt" in combined:
        return "opt"
    if "gpt2" in combined:
        return "gpt2"
    return "unsupported"


def max_context_estimate(config: Any, tokenizer: Any | None) -> int | None:
    config_value = _positive_int(
        config,
        ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"),
    )
    if config_value is not None:
        return config_value
    tokenizer_value = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_value, int)
        and tokenizer_value > 0
        and tokenizer_value < 1_000_000
    ):
        return tokenizer_value
    return None


class UnsupportedModelAdapter:
    name = "unsupported"
    supported = False

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def run_smoke(self, **_kwargs: Any) -> dict[str, Any] | None:
        return None


class Gpt2AttentionPatchAdapter:
    name = "gpt2_phase11_adapter"
    supported = True
    reason = None

    def run_smoke(
        self,
        *,
        args: argparse.Namespace,
        model_name: str,
        target_length: int,
        max_context: int,
        ratio_policy: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        ratio_module = _load_ratio_module()
        max_length = args.max_length or max_context
        effective_target = min(
            target_length,
            max_length - args.max_new_tokens,
        )
        if effective_target <= 0:
            raise ValueError("model context is too small for smoke generation")
        smoke_args = ratio_module._parse_args([
            "--model",
            model_name,
            "--target-token-lengths",
            str(effective_target),
            "--num-prompts-per-length",
            str(args.num_prompts_per_length),
            "--ratio-policies",
            ratio_policy,
            "--policies",
            "query_key_block_score,oracle_topk",
            "--block-size",
            str(args.block_size),
            "--max-new-tokens-values",
            str(args.max_new_tokens),
            "--max-length",
            str(max_length),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--output-dir",
            str(output_dir),
        ])
        result = ratio_module.run_sweep(smoke_args)
        rows_path = Path(result["rows_path"])
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        query_row = next(
            (
                row
                for row in rows
                if row.get("policy") == "query_key_block_score"
                and row.get("status") == "succeeded"
            ),
            None,
        )
        if query_row is None:
            return {
                "smoke_status": "failed",
                "smoke_output_dir": str(output_dir),
            }
        return {
            "smoke_status": "succeeded",
            "smoke_output_dir": str(output_dir),
            "exact_sequence_match_rate": query_row[
                "exact_sequence_match_rate"
            ],
            "average_token_match_rate": query_row[
                "average_token_match_rate"
            ],
            "average_per_step_kl_divergence": query_row[
                "average_per_step_kl_divergence"
            ],
            "average_selected_block_ratio_across_patched_layers": query_row[
                "average_selected_block_ratio_across_patched_layers"
            ],
            "estimated_active_block_reduction_ratio": query_row[
                "estimated_active_block_reduction_ratio"
            ],
        }


def select_adapter(family: str) -> Any:
    if family == "gpt2":
        return Gpt2AttentionPatchAdapter()
    if family == "gpt_neox":
        return UnsupportedModelAdapter(
            "GPTNeoX/Pythia needs an adapter for query_key_value projection, "
            "rotary-position handling, attention output projection, and "
            "parallel residual semantics."
        )
    if family == "opt":
        return UnsupportedModelAdapter(
            "OPT needs an adapter for decoder-layer pre/post normalization, "
            "separate q/k/v projections, attention output projection, and "
            "residual/MLP ordering."
        )
    return UnsupportedModelAdapter(
        "No selected-attention patch adapter is registered for this "
        "architecture."
    )


def compatibility_metadata(
    *,
    config: Any,
    tokenizer: Any | None,
    model: Any | None,
) -> dict[str, Any]:
    family = architecture_family(config)
    adapter = select_adapter(family)
    model_class = (
        type(model).__name__
        if model is not None
        else (
            (getattr(config, "architectures", None) or [None])[0]
        )
    )
    return {
        "model_class": model_class,
        "architecture_family": family,
        "max_context_estimate": max_context_estimate(config, tokenizer),
        "num_layers": _positive_int(
            config,
            ("num_hidden_layers", "n_layer", "num_layers"),
        ),
        "hidden_size": _positive_int(
            config,
            ("hidden_size", "n_embd", "d_model"),
        ),
        "num_attention_heads": _positive_int(
            config,
            ("num_attention_heads", "n_head", "attention_heads"),
        ),
        "supported_for_selected_attention_eval": adapter.supported,
        "adapter_name": adapter.name,
        "reason_if_unsupported": adapter.reason,
    }


def _tokenize_probe(
    *,
    tokenizer: Any,
    target_lengths: list[int],
    max_context: int | None,
    max_new_tokens: int,
    num_prompts: int,
) -> tuple[bool, list[int], list[str]]:
    long_context = _load_long_context_module()
    actual_lengths = []
    warnings = []
    for target in target_lengths:
        if max_context is not None and target + max_new_tokens > max_context:
            warnings.append(
                f"target {target} plus generation exceeds context "
                f"estimate {max_context}"
            )
            continue
        limit = (
            max_context - max_new_tokens
            if max_context is not None
            else target
        )
        prompts = long_context.generate_synthetic_prompts(
            tokenizer=tokenizer,
            target_token_length=target,
            num_prompts=num_prompts,
            max_prompt_tokens=limit,
            seed=0,
        )
        actual_lengths.extend(
            int(row["actual_prompt_token_length"]) for row in prompts
        )
    return bool(actual_lengths), actual_lengths, warnings


def _release_model(model: Any | None) -> None:
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _probe_model(
    *,
    model_name: str,
    args: argparse.Namespace,
    target_lengths: list[int],
    ratio_policy_text: str,
    output_dir: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "status": "probing",
        "can_load_model": False,
        "can_tokenize_long_prompt": False,
        "target_token_lengths": target_lengths,
        "warnings": [],
    }
    model = None
    try:
        try:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for a real model probe"
            ) from exc

        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        row.update(
            compatibility_metadata(
                config=config,
                tokenizer=tokenizer,
                model=None,
            )
        )
        context_limit = row["max_context_estimate"]
        if args.max_length is not None:
            context_limit = (
                min(context_limit, args.max_length)
                if context_limit is not None
                else args.max_length
            )
        can_tokenize, actual_lengths, warnings = _tokenize_probe(
            tokenizer=tokenizer,
            target_lengths=target_lengths,
            max_context=context_limit,
            max_new_tokens=args.max_new_tokens,
            num_prompts=args.num_prompts_per_length,
        )
        row["can_tokenize_long_prompt"] = can_tokenize
        row["actual_prompt_token_lengths"] = actual_lengths
        row["warnings"].extend(warnings)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        row["can_load_model"] = True
        model_metadata = compatibility_metadata(
            config=config,
            tokenizer=tokenizer,
            model=model,
        )
        row["model_class"] = model_metadata["model_class"]
        adapter = select_adapter(row["architecture_family"])
        _release_model(model)
        model = None
        if adapter.supported and can_tokenize:
            smoke_dir = output_dir / "smoke" / model_name.replace("/", "__")
            smoke = adapter.run_smoke(
                args=args,
                model_name=model_name,
                target_length=target_lengths[0],
                max_context=row["max_context_estimate"],
                ratio_policy=ratio_policy_text,
                output_dir=smoke_dir,
            )
            if smoke:
                row.update(smoke)
        row["status"] = (
            "supported"
            if row["supported_for_selected_attention_eval"]
            else "unsupported"
        )
        return row
    except Exception as exc:
        row["status"] = "failed"
        row["reason_if_unsupported"] = str(exc)
        return row
    finally:
        _release_model(model)


def _planned_row(model_name: str, target_lengths: list[int]) -> dict[str, Any]:
    lower = model_name.lower()
    if "pythia" in lower:
        family = "gpt_neox"
    elif "/opt-" in lower:
        family = "opt"
    elif "gpt2" in lower:
        family = "gpt2"
    else:
        family = "unknown"
    return {
        "model": model_name,
        "model_class": None,
        "architecture_family": family,
        "can_load_model": None,
        "can_tokenize_long_prompt": None,
        "max_context_estimate": None,
        "num_layers": None,
        "hidden_size": None,
        "num_attention_heads": None,
        "supported_for_selected_attention_eval": None,
        "adapter_name": None,
        "status": "planned",
        "reason_if_unsupported": (
            "dry-run does not inspect or download model artifacts"
        ),
        "target_token_lengths": target_lengths,
        "warnings": [],
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    supported = [
        row
        for row in rows
        if row.get("supported_for_selected_attention_eval") is True
    ]
    unsupported = [
        row
        for row in rows
        if row.get("supported_for_selected_attention_eval") is False
    ]
    failed = [row for row in rows if row.get("status") == "failed"]
    planned = [row for row in rows if row.get("status") == "planned"]
    recommended_model = (
        supported[0]["model"]
        if supported
        else next(
            (
                row["model"]
                for row in unsupported + planned
                if row["architecture_family"] == "gpt_neox"
            ),
            None,
        )
    )
    adapter_work = list(dict.fromkeys(
        str(row["reason_if_unsupported"])
        for row in unsupported
        if row.get("reason_if_unsupported")
    ))
    return {
        "status": (
            "planned"
            if planned and len(planned) == len(rows)
            else ("complete" if not failed else "partial")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "counts": {
            "total": len(rows),
            "supported": len(supported),
            "unsupported": len(unsupported),
            "failed": len(failed),
            "planned": len(planned),
        },
        "supported_models": supported,
        "unsupported_models": unsupported,
        "failed_models": failed,
        "planned_models": planned,
        "recommended_next_model": recommended_model,
        "recommended_adapter_work": adapter_work,
        "ready_for_selected_attention_smoke": bool(supported),
        "caveats": {
            "outside_vllm": True,
            "no_vllm_integration": True,
            "greedy_generation_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "latency_claim": False,
            "generation_quality_preservation_claim": False,
        },
        "recommended_next_step": (
            "Implement and validate one architecture-specific adapter before "
            "running a longer-context selected-attention sweep."
            if not supported
            else "Run a small supported-model smoke test, then expand prompt "
            "coverage only if the adapter result is clean."
        ),
    }


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(
    lines: list[str],
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")


def _model_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    _table(
        lines,
        [
            "model",
            "class",
            "family",
            "context",
            "layers",
            "hidden",
            "heads",
            "tokenizes",
            "adapter",
            "status",
            "reason",
        ],
        [
            [
                row.get("model"),
                row.get("model_class"),
                row.get("architecture_family"),
                row.get("max_context_estimate"),
                row.get("num_layers"),
                row.get("hidden_size"),
                row.get("num_attention_heads"),
                row.get("can_tokenize_long_prompt"),
                row.get("adapter_name"),
                row.get("status"),
                row.get("reason_if_unsupported"),
            ]
            for row in rows
        ],
    )


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 11.7 Long-Context Model Probe",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        "## High-Level Counts",
        "",
    ]
    _table(
        lines,
        ["field", "value"],
        [[key, value] for key, value in summary["counts"].items()],
    )
    lines.extend(["", "## Supported Models", ""])
    _model_table(lines, summary["supported_models"])
    lines.extend(["", "## Unsupported Models", ""])
    _model_table(lines, summary["unsupported_models"])
    lines.extend(["", "## Failed Or Planned Models", ""])
    _model_table(
        lines,
        summary["failed_models"] + summary["planned_models"],
    )
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"- Recommended next model: `{summary['recommended_next_model']}`",
        "- Ready for selected-attention smoke: "
        f"`{_format(summary['ready_for_selected_attention_smoke'])}`",
        "",
        "### Adapter Work",
        "",
    ])
    lines.extend(
        [f"- {item}" for item in summary["recommended_adapter_work"]]
        or ["- none"]
    )
    lines.extend([
        "",
        "## Caveats",
        "",
        "- This probe runs outside vLLM.",
        "- No vLLM integration or active routing is implemented.",
        "- Unsupported architectures are reported, not patched speculatively.",
        "- No measured runtime memory reduction is claimed.",
        "- No latency claim is made.",
        "- Generation quality preservation is not claimed.",
        "",
        "## Next Step",
        "",
        summary["recommended_next_step"],
    ])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    models = parse_models(args.model, args.models)
    target_lengths = parse_target_lengths(args.target_token_lengths)
    ratio_name, ratios = parse_ratio_policy(args.ratio_policy)
    if args.num_prompts_per_length <= 0:
        raise ValueError("--num-prompts-per-length must be positive")
    if args.max_new_tokens <= 0 or args.block_size <= 0:
        raise ValueError("token and block sizes must be positive")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio_module = _load_ratio_module()
    ratio_text = (
        f"{ratio_name}={ratio_module.format_ratio_policy(ratios)}"
    )
    rows = (
        [_planned_row(model, target_lengths) for model in models]
        if args.dry_run
        else []
    )
    if not args.dry_run:
        for model_name in models:
            row = _probe_model(
                model_name=model_name,
                args=args,
                target_lengths=target_lengths,
                ratio_policy_text=ratio_text,
                output_dir=output_dir,
            )
            rows.append(row)
            if row["status"] == "failed" and not args.continue_on_error:
                break
    config = {
        "models": models,
        "target_token_lengths": target_lengths,
        "num_prompts_per_length": args.num_prompts_per_length,
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "ratio_policy": ratio_text,
        "device": args.device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "dry_run": args.dry_run,
        "known_candidates": list(KNOWN_CANDIDATES),
    }
    summary = build_summary(rows, config=config)
    rows_path = output_dir / "long_context_model_probe_runs.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_json = output_dir / "long_context_model_probe_summary.json"
    summary_md = output_dir / "long_context_model_probe_summary.md"
    _write_json(summary_json, summary)
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return {
        "summary": summary,
        "rows_path": str(rows_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        result = run_probe(_parse_args(argv))
        print(json.dumps({
            "status": result["summary"]["status"],
            "counts": result["summary"]["counts"],
            "recommended_next_model": result["summary"][
                "recommended_next_model"
            ],
            "rows_path": result["rows_path"],
            "summary_json": result["summary_json"],
            "summary_md": result["summary_md"],
        }, separators=(",", ":")))
        return 0 if result["summary"]["counts"]["failed"] == 0 else 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
