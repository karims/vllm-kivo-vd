#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run an offline structured-sketch parameter sweep over HF Q/K tensors."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


STRUCTURED_SKETCH_TYPES = {
    "bidiagonal_sign",
    "bidiagonal_sign_subsample",
    "tridiagonal_sign",
}


def _load_module(module_name: str, relative_path: str) -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def _parse_float_csv(value: str) -> list[float]:
    return [float(part) for part in _parse_csv(value)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline parameter sweep for structured Kivo-VD sketches."
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--sketch-types",
        default="bidiagonal_sign_subsample,bidiagonal_sign,tridiagonal_sign",
        help=(
            "Comma-separated structured sketch types: bidiagonal_sign, "
            "bidiagonal_sign_subsample, tridiagonal_sign."
        ),
    )
    parser.add_argument("--sketch-dims", default="16,24,32,48")
    parser.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--coordinate-strategies",
        default="uniform,stride,low,high,alternating",
    )
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument("--heads", default="0,1,2,3")
    parser.add_argument("--query-positions", default="sweep")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--topk-blocks", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--truncate-side", choices=["left", "right"], default="right")
    parser.add_argument(
        "--extraction-mode",
        choices=["auto", "gpt2_fused_c_attn", "separate_qk_proj"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default=(
            "outputs/kivo_vd/runs/phase6_2_structured_param_sweep/"
            "structured_param_sweep.jsonl"
        ),
    )
    parser.add_argument(
        "--include-ranked-blocks",
        action="store_true",
        help="Include full approximate block rankings in each row.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    hf_eval = _load_module(
        "run_hf_qk_sketch_eval", "scripts/kivo_vd/run_hf_qk_sketch_eval.py"
    )
    head_sweep = _load_module(
        "run_hf_qk_head_sweep", "scripts/kivo_vd/run_hf_qk_head_sweep.py"
    )
    math = hf_eval._load_sketch_math_module()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "This optional sweep requires torch and transformers. Install them "
            "in the benchmark environment first."
        ) from exc

    sketch_types = _parse_csv(args.sketch_types)
    invalid = [s for s in sketch_types if s not in STRUCTURED_SKETCH_TYPES]
    if invalid:
        raise ValueError(
            f"Unsupported structured sketch types {invalid}; expected one of "
            f"{sorted(STRUCTURED_SKETCH_TYPES)}"
        )
    sketch_dims = _parse_int_csv(args.sketch_dims)
    alphas = _parse_float_csv(args.alphas)
    coordinate_strategies = _parse_csv(args.coordinate_strategies)
    allowed_strategies = {"uniform", "stride", "low", "high", "alternating"}
    invalid_strategies = [s for s in coordinate_strategies if s not in allowed_strategies]
    if invalid_strategies:
        raise ValueError(
            f"Unsupported coordinate strategies {invalid_strategies}; expected "
            f"one of {sorted(allowed_strategies)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(args.device)
    model.eval()

    prompt = args.prompt if args.prompt is not None else hf_eval._default_prompt()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    original_prompt_num_tokens = int(input_ids.shape[1])

    model_context = hf_eval._resolve_model_max_context_tokens(model, tokenizer)
    if model_context is not None and args.max_tokens is not None:
        effective_max_tokens = min(model_context, args.max_tokens)
    else:
        effective_max_tokens = model_context if model_context is not None else args.max_tokens

    input_ids, truncated = hf_eval._truncate_input_ids(
        input_ids=input_ids,
        max_tokens=effective_max_tokens,
        truncate_side=args.truncate_side,
    )
    prompt_num_tokens = int(input_ids.shape[1])

    layers_modules = hf_eval._get_transformer_layers(model)
    first_layer_idx = 0 if args.layers == "all" else int(args.layers.split(",")[0])
    first_attn = hf_eval._get_layer_attention(layers_modules[first_layer_idx])
    num_heads = hf_eval._get_num_query_heads(model, first_attn)
    layers = head_sweep._parse_index_list(args.layers, len(layers_modules), "layer")
    heads = head_sweep._parse_index_list(args.heads, num_heads, "head")
    query_positions = head_sweep._parse_query_positions(
        args.query_positions, prompt_num_tokens, hf_eval
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = 0
    with output_path.open("w", encoding="utf-8") as f:
        for sketch_type in sketch_types:
            for sketch_dim in sketch_dims:
                for alpha in alphas:
                    for coordinate_strategy in coordinate_strategies:
                        for layer in layers:
                            for head in heads:
                                for query_position in query_positions:
                                    metrics = hf_eval._evaluate_at_query_position(
                                        math=math,
                                        model=model,
                                        input_ids=input_ids,
                                        layer=layer,
                                        head=head,
                                        query_position=query_position,
                                        sketch_type=sketch_type,
                                        sketch_dim=sketch_dim,
                                        seed=args.seed,
                                        block_size=args.block_size,
                                        topk_blocks=args.topk_blocks,
                                        include_ranked_blocks=args.include_ranked_blocks,
                                        extraction_mode=args.extraction_mode,
                                        structured_alpha=alpha,
                                        structured_coordinate_strategy=(
                                            coordinate_strategy
                                        ),
                                    )
                                    row = {
                                        "model_name": args.model_name,
                                        "original_prompt_num_tokens": (
                                            original_prompt_num_tokens
                                        ),
                                        "prompt_num_tokens": prompt_num_tokens,
                                        "truncated": bool(truncated),
                                        "max_context_tokens": (
                                            int(effective_max_tokens)
                                            if effective_max_tokens is not None
                                            else None
                                        ),
                                        "layer": layer,
                                        "head": head,
                                        "query_positions_spec": args.query_positions,
                                        "extraction_mode_requested": (
                                            args.extraction_mode
                                        ),
                                        "sketch_type": sketch_type,
                                        "sketch_dim": sketch_dim,
                                        "structured_alpha": alpha,
                                        "structured_coordinate_strategy": (
                                            coordinate_strategy
                                        ),
                                        "block_size": args.block_size,
                                        "topk_blocks": args.topk_blocks,
                                        "seed": args.seed,
                                    }
                                    row.update(metrics)
                                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                                    num_rows += 1

    print(json.dumps({"output": str(output_path), "num_rows": num_rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
