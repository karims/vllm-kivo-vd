#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny live decode smoke for the sketch-gated demotion/free path.

This runner is intended for a real GPU pod environment.
Local macOS validation is limited to unit tests and argument/config checks.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (  # noqa: E402
    _build_llm_kwargs,
    _load_exported_counters,
    build_prompts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Sk-1.3c tiny live sketch smoke test."
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help=(
            "Resolve Hugging Face repo IDs from the local cache only and "
            "avoid network-backed model lookup when possible."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-repeats", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--sketch-seed", type=int, default=123)
    parser.add_argument(
        "--runtime-policy",
        default="recent_only",
        choices=("recent_only", "countsketch_online", "sketch_topk"),
    )
    parser.add_argument("--keep-recent-blocks", type=int, default=2)
    parser.add_argument("--max-full-blocks", type=int, default=2)
    parser.add_argument("--sketch-topk", type=int, default=2)
    parser.add_argument(
        "--counter-export-file",
        default=None,
        help=(
            "Optional explicit path for KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE. "
            "Defaults to <output>.counters.json."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _hf_cache_repo_dir(model_name: str) -> Path:
    escaped = model_name.replace("/", "--")
    return HF_CACHE_ROOT / f"models--{escaped}"


def discover_cached_model_paths(
    model_name: str | None = None,
    *,
    limit: int = 20,
) -> list[str]:
    base_dirs: list[Path]
    if model_name is not None:
        candidate = _hf_cache_repo_dir(model_name)
        base_dirs = [candidate] if candidate.exists() else []
    else:
        if not HF_CACHE_ROOT.exists():
            return []
        base_dirs = sorted(HF_CACHE_ROOT.glob("models--*"))

    results: list[str] = []
    for repo_dir in base_dirs:
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            results.append(str(snapshot))
            if len(results) >= limit:
                return results
    return results


def resolve_model_reference(
    model: str,
    *,
    local_files_only: bool,
) -> tuple[str, bool, list[str]]:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path), True, []
    if not local_files_only:
        return model, False, discover_cached_model_paths(model)

    cached_candidates = discover_cached_model_paths(model)
    if cached_candidates:
        return cached_candidates[0], True, cached_candidates
    raise FileNotFoundError(
        f"No cached local snapshot found for model {model!r} under "
        f"{HF_CACHE_ROOT}"
    )


def _smoke_env(args: argparse.Namespace) -> dict[str, str]:
    counter_export_file = args.counter_export_file or str(
        Path(args.output).with_suffix(".counters.json")
    )
    env = {
        "KIVO_KV_SKETCH_ENABLE": "1",
        "KIVO_KV_SKETCH_BACKEND": "random_projection",
        "KIVO_KV_SKETCH_DIM": str(args.sketch_dim),
        "KIVO_KV_SKETCH_SEED": str(args.sketch_seed),
        "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE": "1",
        "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION": "apply_block_table_only",
        "KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY": args.runtime_policy,
        "KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS": str(
            args.keep_recent_blocks
        ),
        "KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS": str(
            args.max_full_blocks
        ),
        "KIVO_KV_RUNTIME_BLOCK_TABLE_SKETCH_TOPK": str(args.sketch_topk),
        "KIVO_KV_DEMOTION_TRANSPORT_ENABLE": "1",
        "KIVO_KV_DEMOTION_TRANSPORT_ACTION": "apply_core_mark_demoted",
        "KIVO_KV_CORE_DEMOTION_ENABLE": "1",
        "KIVO_KV_CORE_DEMOTION_ACTION": "mark_demoted_only",
        "KIVO_KV_OWNERSHIP_REMOVE_ENABLE": "1",
        "KIVO_KV_OWNERSHIP_REMOVE_ACTION": "remove_marked_demoted_only",
        "KIVO_KV_FREE_TO_POOL_ENABLE": "1",
        "KIVO_KV_FREE_TO_POOL_ACTION": "free_removed_demoted_only",
        "KIVO_KV_DEMOTION_COUNTERS_ENABLE": "1",
        "KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE": counter_export_file,
    }
    if args.local_files_only:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


@contextlib.contextmanager
def patched_environ(values: dict[str, str]) -> Iterator[None]:
    original: dict[str, str | None] = {
        key: os.environ.get(key) for key in values
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in original.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def summarize_sketch_counters(
    counters: dict[str, Any] | None,
    *,
    expected_backend: str = "random_projection",
) -> dict[str, Any]:
    counters = counters or {}
    sketch_backend = counters.get("sketch_backend")
    sketch_attempted = int(counters.get("sketch_build_attempted", 0) or 0)
    sketch_succeeded = int(counters.get("sketch_build_succeeded", 0) or 0)
    sketch_failed = int(counters.get("sketch_build_failed", 0) or 0)
    sketched_blocks_total = int(counters.get("sketched_blocks_total", 0) or 0)
    freed_after_sketch = int(
        counters.get("freed_after_sketch_blocks_total", 0) or 0
    )
    missing_prevented_demotion = int(
        counters.get("sketch_missing_prevented_demotion", 0) or 0
    )
    missing_prevented_free = int(
        counters.get("sketch_missing_prevented_free", 0) or 0
    )
    invariant_failures = {
        "ownership_remove_invariant_failed": int(
            counters.get("ownership_remove_invariant_failed", 0) or 0
        ),
        "free_to_pool_double_free_prevented": int(
            counters.get("free_to_pool_double_free_prevented", 0) or 0
        ),
        "block_pool_free_accounting_rejected": int(
            counters.get("block_pool_free_accounting_rejected", 0) or 0
        ),
    }
    all_clean = all(value == 0 for value in invariant_failures.values())
    warnings: list[str] = []
    if sketch_attempted > 0 and sketch_succeeded == 0:
        warnings.append(
            "sketch_attempted_without_success; possible kv_cache shape extraction issue"
        )
    if sketch_backend not in (None, expected_backend):
        warnings.append(
            f"unexpected_sketch_backend:{sketch_backend}"
        )
    return {
        "sketch_backend": sketch_backend,
        "sketch_build_attempted": sketch_attempted,
        "sketch_build_succeeded": sketch_succeeded,
        "sketch_build_failed": sketch_failed,
        "sketched_blocks_total": sketched_blocks_total,
        "sketch_missing_prevented_demotion": missing_prevented_demotion,
        "sketch_missing_prevented_free": missing_prevented_free,
        "freed_after_sketch_blocks_total": freed_after_sketch,
        "ownership_remove_succeeded": int(
            counters.get("ownership_remove_succeeded", 0) or 0
        ),
        "free_to_pool_succeeded": int(
            counters.get("free_to_pool_succeeded", 0) or 0
        ),
        "free_to_pool_calls": int(counters.get("free_to_pool_calls", 0) or 0),
        "sketch_bytes_total": int(counters.get("sketch_bytes_total", 0) or 0),
        "invariant_counters": invariant_failures,
        "invariants_clean": all_clean,
        "warnings": warnings,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    export_file = args.counter_export_file or str(
        Path(args.output).with_suffix(".counters.json")
    )
    export_path = Path(export_file)
    if export_path.exists():
        export_path.unlink()

    prompts = build_prompts(repeats=args.prompt_repeats, num_prompts=args.num_prompts)
    llm = None
    env_values = _smoke_env(args)
    resolved_model = args.model
    model_is_local = False
    cached_model_candidates: list[str] = []
    try:
        resolved_model, model_is_local, cached_model_candidates = (
            resolve_model_reference(
                args.model,
                local_files_only=args.local_files_only,
            )
        )
        with patched_environ(env_values):
            from vllm import LLM, SamplingParams
            from vllm.v1.core.kivo_demotion_counters import (
                get_kivo_demotion_counters_snapshot,
                reset_kivo_demotion_counters,
            )

            reset_kivo_demotion_counters()
            llm_kwargs = _build_llm_kwargs(args)
            llm_kwargs["model"] = resolved_model
            llm = LLM(**llm_kwargs)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            prompt_token_lengths = [
                len(getattr(output, "prompt_token_ids", None) or [])
                if getattr(output, "prompt_token_ids", None) is not None
                else None
                for output in outputs
            ]
            parent_counters = get_kivo_demotion_counters_snapshot()
            exported_counters, file_found, export_pid = _load_exported_counters(
                export_file
            )
            counters = exported_counters if exported_counters is not None else parent_counters
            return {
                "generation_success": True,
                "model": args.model,
                "resolved_model": resolved_model,
                "model_is_local": model_is_local,
                "local_files_only": args.local_files_only,
                "cached_model_candidates": cached_model_candidates,
                "prompt_count": len(prompts),
                "prompt_token_lengths": prompt_token_lengths,
                "env_flags": env_values,
                "counter_export_file_found": file_found,
                "counter_export_pid": export_pid,
                "counter_export_file": export_file,
                "counters": counters,
                "counter_summary": summarize_sketch_counters(counters),
                "error": None,
            }
    except Exception as exc:
        exported_counters, file_found, export_pid = _load_exported_counters(export_file)
        counters = exported_counters if exported_counters is not None else {}
        return {
            "generation_success": False,
            "model": args.model,
            "resolved_model": resolved_model,
            "model_is_local": model_is_local,
            "local_files_only": args.local_files_only,
            "cached_model_candidates": cached_model_candidates,
            "prompt_count": len(prompts),
            "prompt_token_lengths": [None] * len(prompts),
            "env_flags": env_values,
            "counter_export_file_found": file_found,
            "counter_export_pid": export_pid,
            "counter_export_file": export_file,
            "counters": counters,
            "counter_summary": summarize_sketch_counters(counters),
            "model_resolution_debug": {
                "hf_cache_root": str(HF_CACHE_ROOT),
                "requested_model": args.model,
                "resolved_model": resolved_model,
                "model_is_local": model_is_local,
                "local_files_only": args.local_files_only,
                "cached_model_candidates": cached_model_candidates,
                "export_file_exists": export_path.exists(),
                "export_file_size": export_path.stat().st_size
                if export_path.exists()
                else 0,
            },
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if llm is not None:
            del llm
        gc.collect()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_smoke(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["generation_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
