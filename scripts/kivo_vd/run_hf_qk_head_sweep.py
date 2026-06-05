#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_hf_eval_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "kivo_vd" / "run_hf_qk_sketch_eval.py"
    spec = importlib.util.spec_from_file_location("run_hf_qk_sketch_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_index_list(spec: str, max_count: int, label: str) -> list[int]:
    if spec == "all":
        return list(range(max_count))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= max_count:
            raise ValueError(f"{label} index {idx} out of range [0, {max_count - 1}]")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError(f"{label} list is empty")
    return out


def _parse_query_positions(spec: str, seq_len: int, hf_eval: Any) -> list[int]:
    if spec == "sweep":
        return hf_eval._sweep_query_positions(seq_len)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        pos = hf_eval._resolve_query_position(part, seq_len)
        if pos not in out:
            out.append(pos)
    if not out:
        raise ValueError("query positions list is empty")
    return out


def _parse_sketch_types(sketch_type: str, sketch_types: str | None) -> list[str]:
    if sketch_types is None:
        return [sketch_type]
    allowed = {
        "random_projection",
        "count_sketch",
        "srht",
        "bidiagonal_sign",
        "bidiagonal_sign_subsample",
        "tridiagonal_sign",
    }
    out: list[str] = []
    for part in sketch_types.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in allowed:
            raise ValueError(
                f"Invalid sketch type {part!r}; expected one of {sorted(allowed)}"
            )
        if part not in out:
            out.append(part)
    if not out:
        raise ValueError("sketch-types list is empty")
    return out


def _parse_sketch_dims(sketch_dim: int, sketch_dims: str | None) -> list[int]:
    if sketch_dims is None:
        return [int(sketch_dim)]
    out: list[int] = []
    for part in sketch_dims.split(","):
        part = part.strip()
        if not part:
            continue
        dim = int(part)
        if dim <= 0:
            raise ValueError(f"Invalid sketch dim {dim}; expected positive integer")
        if dim not in out:
            out.append(dim)
    if not out:
        raise ValueError("sketch-dims list is empty")
    return out


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _resolve_query_head_dim(model: Any, attn: Any, num_query_heads: int) -> int:
    for obj, name in (
        (getattr(model, "config", None), "hidden_size"),
        (getattr(model, "config", None), "n_embd"),
    ):
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0 and value % num_query_heads == 0:
            return value // num_query_heads

    q_proj = getattr(attn, "q_proj", None)
    out_features = getattr(q_proj, "out_features", None)
    if isinstance(out_features, int) and out_features > 0:
        return out_features // num_query_heads

    c_attn = getattr(attn, "c_attn", None)
    nf = getattr(c_attn, "nf", None)
    if isinstance(nf, int) and nf > 0:
        return nf // num_query_heads

    raise ValueError("Unable to determine per-head Q/K dimension.")


def _srht_sketch_dim_is_valid(head_dim: int, sketch_dim: int) -> bool:
    return sketch_dim <= _next_power_of_two(head_dim)


def _parse_args(hf_eval: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional HF Q/K layer/head sweep.")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--prompt", default=hf_eval._default_prompt())
    parser.add_argument(
        "--sketch-type",
        choices=[
            "random_projection",
            "count_sketch",
            "srht",
            "bidiagonal_sign",
            "bidiagonal_sign_subsample",
            "tridiagonal_sign",
        ],
        default="random_projection",
    )
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--sketch-types", default=None)
    parser.add_argument("--sketch-dims", default=None)
    parser.add_argument("--structured-alpha", type=float, default=None)
    parser.add_argument(
        "--structured-coordinate-strategy",
        choices=["uniform", "stride", "low", "high", "alternating"],
        default="uniform",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--topk-blocks", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--truncate-side", choices=["left", "right"], default="right")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--query-positions", default="sweep")
    parser.add_argument(
        "--extraction-mode",
        choices=["auto", "gpt2_fused_c_attn", "separate_qk_proj"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", default="outputs/kivo_vd/hf_qk_head_sweep.jsonl"
    )
    parser.add_argument(
        "--include-ranked-blocks",
        action="store_true",
        help="Include full approximate block rankings for policy simulation.",
    )
    return parser.parse_args()


def _aggregate(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows to summarize.")
        return

    metric_keys = [
        "block_topk_recall",
        "block_recall_at_2x_budget",
        "block_recall_at_4x_budget",
        "block_mrr",
        "block_score_correlation",
    ]

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            f"avg_{k}": sum(float(r[k]) for r in group_rows) / len(group_rows)
            for k in metric_keys
        }

    overall = summarize(rows)
    print("SUMMARY overall")
    print(json.dumps({"count": len(rows), **overall}, separators=(",", ":")))

    grouped_sd: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_sd[(row["sketch_type"], int(row["sketch_dim"]))].append(row)

    print("SUMMARY by sketch_type/sketch_dim")
    for key in sorted(grouped_sd.keys()):
        group_rows = grouped_sd[key]
        stats = summarize(group_rows)
        payload = {
            "sketch_type": key[0],
            "sketch_dim": key[1],
            "count": len(group_rows),
            **stats,
        }
        print(json.dumps(payload, separators=(",", ":")))

    grouped_sdl: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["sketch_type"], int(row["sketch_dim"]), int(row["layer"]))
        grouped_sdl[key].append(row)

    print("SUMMARY by sketch_type/sketch_dim/layer")
    for key in sorted(grouped_sdl.keys()):
        group_rows = grouped_sdl[key]
        stats = summarize(group_rows)
        payload = {
            "sketch_type": key[0],
            "sketch_dim": key[1],
            "layer": key[2],
            "count": len(group_rows),
            **stats,
        }
        print(json.dumps(payload, separators=(",", ":")))


def main() -> int:
    hf_eval = _load_hf_eval_module()
    args = _parse_args(hf_eval)
    math = hf_eval._load_sketch_math_module()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "This optional script requires torch and transformers. "
            "Install them in your environment first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(args.device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    original_prompt_num_tokens = int(input_ids.shape[1])

    model_context = hf_eval._resolve_model_max_context_tokens(model, tokenizer)
    if model_context is not None and args.max_tokens is not None:
        effective_max_tokens = min(model_context, args.max_tokens)
    else:
        effective_max_tokens = (
            model_context if model_context is not None else args.max_tokens
        )

    input_ids, truncated = hf_eval._truncate_input_ids(
        input_ids=input_ids,
        max_tokens=effective_max_tokens,
        truncate_side=args.truncate_side,
    )
    prompt_num_tokens = int(input_ids.shape[1])

    layers_modules = hf_eval._get_transformer_layers(model)
    num_layers = len(layers_modules)
    first_layer_idx = 0 if args.layers == "all" else int(args.layers.split(",")[0])
    first_attn = hf_eval._get_layer_attention(layers_modules[first_layer_idx])
    num_heads = hf_eval._get_num_query_heads(model, first_attn)
    head_dim = _resolve_query_head_dim(model, first_attn, num_heads)
    layers = _parse_index_list(args.layers, num_layers, "layer")
    heads = _parse_index_list(args.heads, num_heads, "head")
    query_positions = _parse_query_positions(
        args.query_positions, prompt_num_tokens, hf_eval
    )
    sketch_types = _parse_sketch_types(args.sketch_type, args.sketch_types)
    sketch_dims = _parse_sketch_dims(args.sketch_dim, args.sketch_dims)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for sketch_type in sketch_types:
            for sketch_dim in sketch_dims:
                if sketch_type == "srht" and not _srht_sketch_dim_is_valid(
                    head_dim, sketch_dim
                ):
                    continue
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
                                structured_alpha=args.structured_alpha,
                                structured_coordinate_strategy=(
                                    args.structured_coordinate_strategy
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
                                "extraction_mode_requested": args.extraction_mode,
                                "sketch_type": sketch_type,
                                "sketch_dim": sketch_dim,
                                "block_size": args.block_size,
                                "topk_blocks": args.topk_blocks,
                                "seed": args.seed,
                                "structured_alpha": args.structured_alpha,
                                "structured_coordinate_strategy": (
                                    args.structured_coordinate_strategy
                                ),
                            }
                            row.update(metrics)
                            f.write(json.dumps(row, separators=(",", ":")) + "\n")
                            rows.append(row)

    _aggregate(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
