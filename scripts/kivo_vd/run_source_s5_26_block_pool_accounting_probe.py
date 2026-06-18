#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S5.26 block-pool accounting probe."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SENTENCE = (
    "This is a cached context sentence about retrieval, memory, attention, "
    "reuse, and demotion accounting."
)
VARIED_SENTENCE_BANK = (
    "A short note about cache reuse and prompt diversity.",
    "An observation about block pools, ownership, and reuse safety.",
    "A reminder that counters can move without proving GPU memory reduction.",
    "A compact benchmark line about prompt variety and pool accounting.",
)


class TokenizerLike(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]: ...


def build_long_context_prompt(*, repeats: int = 200) -> str:
    long_context = " ".join([DEFAULT_SENTENCE] * repeats)
    return (
        long_context
        + "\nQuestion: summarize the repeated context in one sentence."
    )


def build_varied_prompt(*, repeats: int, prompt_index: int) -> str:
    bank_sentence = VARIED_SENTENCE_BANK[prompt_index % len(VARIED_SENTENCE_BANK)]
    context = " ".join(
        [
            f"{bank_sentence} Variant {prompt_index}.",
            *([DEFAULT_SENTENCE] * max(1, repeats)),
            f"Unique suffix token set {prompt_index} / {prompt_index * 7 + 3}.",
        ]
    )
    return (
        context
        + f"\nQuestion: summarize prompt {prompt_index} and mention its suffix."
    )


def _count_tokens(tokenizer: TokenizerLike, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _load_tokenizer(model_name: str) -> TokenizerLike | None:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    return AutoTokenizer.from_pretrained(model_name)


def _target_tokens_for_index(
    *,
    target_prompt_tokens: int,
    prompt_index: int,
    prompt_mode: str,
    max_model_len: int,
) -> int:
    if prompt_mode != "varied-length":
        return min(target_prompt_tokens, max_model_len)
    multipliers = (0.75, 0.9, 1.0, 1.15)
    target = int(round(target_prompt_tokens * multipliers[prompt_index % len(multipliers)]))
    return min(max(1, target), max_model_len)


def build_targeted_prompt(
    tokenizer: TokenizerLike,
    *,
    target_prompt_tokens: int,
    prompt_index: int,
    prompt_mode: str,
    max_model_len: int,
) -> str:
    target_tokens = _target_tokens_for_index(
        target_prompt_tokens=target_prompt_tokens,
        prompt_index=prompt_index,
        prompt_mode=prompt_mode,
        max_model_len=max_model_len,
    )
    if prompt_mode == "repeated":
        prompt = build_long_context_prompt(repeats=1)
        filler = DEFAULT_SENTENCE
    else:
        prompt = build_varied_prompt(repeats=1, prompt_index=prompt_index)
        filler = VARIED_SENTENCE_BANK[prompt_index % len(VARIED_SENTENCE_BANK)]
    while _count_tokens(tokenizer, prompt) < target_tokens:
        prompt += f" {filler}"
    return prompt


def build_prompts(
    *,
    repeats: int,
    num_prompts: int,
    prompt_mode: str = "repeated",
    target_prompt_tokens: int | None = None,
    tokenizer: TokenizerLike | None = None,
    max_model_len: int = 1024,
) -> list[str]:
    num_prompts = max(1, num_prompts)
    if target_prompt_tokens is not None and tokenizer is not None:
        return [
            build_targeted_prompt(
                tokenizer,
                target_prompt_tokens=target_prompt_tokens,
                prompt_index=idx,
                prompt_mode=prompt_mode,
                max_model_len=max_model_len,
            )
            for idx in range(num_prompts)
        ]
    if prompt_mode == "repeated":
        prompt = build_long_context_prompt(repeats=repeats)
        return [prompt for _ in range(num_prompts)]
    if prompt_mode == "varied":
        return [
            build_varied_prompt(repeats=repeats, prompt_index=idx)
            for idx in range(num_prompts)
        ]
    if prompt_mode == "varied-length":
        prompts: list[str] = []
        for idx in range(num_prompts):
            offset = (idx % 5) - 2
            varied_repeats = max(1, repeats + offset * max(1, repeats // 8 or 1))
            prompts.append(
                build_varied_prompt(
                    repeats=varied_repeats,
                    prompt_index=idx,
                )
            )
        return prompts
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an S5.26 vLLM probe for block-pool accounting."
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument(
        "--max-output-tokens",
        "--max-tokens",
        dest="max_output_tokens",
        type=int,
        default=16,
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-repeats", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--target-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=("repeated", "varied", "varied-length"),
        default="repeated",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable prefix caching if this local vLLM LLM surface supports it.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _snapshot_cuda_memory() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "cuda_memory_snapshot_observed": False,
            "cuda_memory_snapshot_blocker": "torch_unavailable",
        }

    if not torch.cuda.is_available():
        return {
            "cuda_memory_snapshot_observed": False,
            "cuda_memory_snapshot_blocker": "cuda_unavailable",
        }

    try:
        return {
            "cuda_memory_snapshot_observed": True,
            "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
            "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
        }
    except Exception:
        return {
            "cuda_memory_snapshot_observed": False,
            "cuda_memory_snapshot_blocker": "cuda_snapshot_failed",
        }


def _load_exported_counters(
    path: str | None,
) -> tuple[dict[str, Any] | None, bool, int | None]:
    if not path:
        return None, False, None
    export_path = Path(path)
    if not export_path.exists():
        return None, False, None
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    exported_counters = payload.get("counters")
    if not isinstance(exported_counters, dict):
        exported_counters = None
    pid = payload.get("pid")
    return exported_counters, True, pid if isinstance(pid, int) else None


def _supports_enable_prefix_caching(llm_cls: type) -> bool:
    try:
        signature = inspect.signature(llm_cls.__init__)
    except (TypeError, ValueError):
        return False
    return "enable_prefix_caching" in signature.parameters


def _build_llm_kwargs(args: argparse.Namespace, llm_cls: type) -> tuple[dict[str, Any], bool]:
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "enforce_eager": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.device != "auto":
        llm_kwargs["device"] = args.device
    prefix_caching_disabled = False
    if args.disable_prefix_caching and _supports_enable_prefix_caching(llm_cls):
        llm_kwargs["enable_prefix_caching"] = False
        prefix_caching_disabled = True
    return llm_kwargs, prefix_caching_disabled


def _average_prompt_tokens(prompt_token_lengths: list[int | None]) -> float | None:
    valid = [length for length in prompt_token_lengths if isinstance(length, int)]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _sum_prompt_tokens(prompt_token_lengths: list[int | None]) -> int | None:
    valid = [length for length in prompt_token_lengths if isinstance(length, int)]
    if not valid:
        return None
    return sum(valid)


def _max_prompt_tokens(prompt_token_lengths: list[int | None]) -> int | None:
    valid = [length for length in prompt_token_lengths if isinstance(length, int)]
    if not valid:
        return None
    return max(valid)


def _build_counter_views(counters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cumulative_keys = (
        "worker_envelopes_built",
        "scheduler_envelopes_received",
        "core_commands_attempted",
        "core_commands_accepted",
        "core_commands_rejected",
        "manager_mark_demoted_attempted",
        "manager_mark_demoted_succeeded",
        "manager_mark_demoted_rejected",
        "demoted_blocks_marked",
        "ownership_remove_attempted",
        "ownership_remove_succeeded",
        "ownership_remove_rejected",
        "ownership_remove_invariant_checked",
        "ownership_remove_invariant_failed",
        "ownership_removed_subset_of_marked",
        "ownership_removed_subset_of_owned",
        "ownership_removed_absent_after",
        "ownership_remaining_nonempty",
        "ownership_removed_reintroduced",
        "ownership_demoted_bookkeeping_cleared",
        "ownership_removed_blocks",
        "ownership_removed_blocks_total",
        "req_to_blocks_removed",
        "free_to_pool_attempted",
        "free_to_pool_succeeded",
        "free_to_pool_rejected",
        "free_to_pool_blocks",
        "free_to_pool_calls",
        "free_to_pool_double_free_prevented",
        "block_pool_free_accounting_observed",
        "block_pool_free_accounting_increased",
        "block_pool_free_accounting_rejected",
    )
    last_keys = (
        "last_visible_before_count",
        "last_visible_after_count",
        "last_candidate_demote_count",
        "ownership_remaining_blocks_last",
        "last_owned_before_remove_count",
        "last_owned_after_remove_count",
        "last_marked_demoted_before_remove_count",
        "last_removed_after_absent",
        "last_removed_block_ids_sample",
        "last_remaining_block_ids_sample",
        "last_freed_block_ids_sample",
        "last_free_rejected_block_ids_sample",
        "block_pool_free_capacity_before",
        "block_pool_free_capacity_after",
        "block_pool_free_capacity_delta",
        "block_pool_num_free_blocks_before",
        "block_pool_num_free_blocks_after",
        "block_pool_num_free_blocks_delta",
        "block_pool_free_accounting_blocker_reasons",
    )
    cumulative = {key: counters.get(key) for key in cumulative_keys}
    last_snapshot = {key: counters.get(key) for key in last_keys}
    return cumulative, last_snapshot


def _build_compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_success": bool(summary.get("generation_success")),
        "prompt_count": int(summary.get("prompt_count", 0) or 0),
        "avg_prompt_tokens": summary.get("avg_prompt_tokens"),
        "max_prompt_tokens": summary.get("max_prompt_tokens"),
        "requested_max_output_tokens": summary.get("requested_max_output_tokens"),
        "max_num_seqs": summary.get("max_num_seqs"),
        "estimated_max_active_total_tokens": summary.get(
            "estimated_max_active_total_tokens"
        ),
        "gpu_kv_cache_size_tokens": summary.get("gpu_kv_cache_size_tokens"),
        "estimated_pressure_ratio": summary.get("estimated_pressure_ratio"),
        "transport_observed": bool(summary.get("transport_observed")),
        "ownership_remove_observed": bool(summary.get("ownership_remove_observed")),
        "free_to_pool_observed": bool(
            int(summary.get("free_to_pool_succeeded", 0) or 0) > 0
            or int(summary.get("free_to_pool_calls", 0) or 0) > 0
        ),
        "block_pool_accounting_observed": bool(
            int(summary.get("block_pool_free_accounting_observed", 0) or 0) > 0
        ),
        "req_to_blocks_removed_total": int(
            summary.get("req_to_blocks_removed_total", 0) or 0
        ),
        "free_to_pool_blocks_total_or_last": (
            summary.get("free_to_pool_blocks_total")
            if summary.get("free_to_pool_blocks_total") is not None
            else summary.get("free_to_pool_blocks_last")
        ),
        "free_to_pool_calls_total": int(
            summary.get("free_to_pool_calls_total", 0) or 0
        ),
        "removed_but_not_freed_block_ids_count": int(
            summary.get("removed_but_not_freed_block_ids_count", 0) or 0
        ),
        "free_attempt_duplicate_block_ids_count": int(
            summary.get("free_attempt_duplicate_block_ids_count", 0) or 0
        ),
        "invariant_failed": int(
            summary.get("ownership_remove_invariant_failed", 0) or 0
        ),
        "wall_time_seconds": summary.get("wall_time_seconds"),
    }


def build_summary(
    *,
    args: argparse.Namespace,
    generation_success: bool,
    prompt_char_lengths: list[int],
    prompt_token_lengths: list[int | None],
    parent_counters: dict[str, Any],
    exported_counters: dict[str, Any] | None,
    counter_export_file_found: bool,
    counter_export_pid: int | None,
    prefix_caching_disabled: bool,
    prefix_caching_disable_supported: bool,
    cuda_before: dict[str, Any],
    cuda_after: dict[str, Any],
    wall_time_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    from scripts.kivo_vd.run_source_s5_19_demotable_transport_probe import (
        build_summary as build_transport_summary,
    )

    base_summary = build_transport_summary(
        generation_success=generation_success,
        prompt_count=len(prompt_char_lengths),
        prompt_char_lengths=prompt_char_lengths,
        prompt_token_lengths=prompt_token_lengths,
        parent_counters=parent_counters,
        exported_counters=exported_counters,
        counter_export_file_found=counter_export_file_found,
        counter_export_pid=counter_export_pid,
        error=error,
    )
    counters = base_summary.get("counters")
    if not isinstance(counters, dict):
        counters = {}
    cumulative_counters, last_snapshot_counters = _build_counter_views(counters)
    avg_prompt_tokens = _average_prompt_tokens(prompt_token_lengths)
    total_prompt_tokens = _sum_prompt_tokens(prompt_token_lengths)
    max_prompt_tokens = _max_prompt_tokens(prompt_token_lengths)
    requested_max_output_tokens = int(
        getattr(args, "max_output_tokens", getattr(args, "max_tokens", 16))
    )
    max_num_seqs = int(getattr(args, "max_num_seqs", 1))
    estimated_max_active_prompt_tokens = (
        avg_prompt_tokens * max_num_seqs
        if avg_prompt_tokens is not None
        else None
    )
    estimated_max_active_total_tokens = (
        (avg_prompt_tokens + requested_max_output_tokens) * max_num_seqs
        if avg_prompt_tokens is not None
        else None
    )
    gpu_kv_cache_size_tokens = None
    estimated_pressure_ratio = (
        estimated_max_active_total_tokens / gpu_kv_cache_size_tokens
        if (
            estimated_max_active_total_tokens is not None
            and gpu_kv_cache_size_tokens not in (None, 0)
        )
        else None
    )
    req_to_blocks_removed_total = int(counters.get("req_to_blocks_removed", 0) or 0)
    ownership_removed_blocks_total = int(
        counters.get("ownership_removed_blocks_total", 0)
        or counters.get("ownership_removed_blocks", 0)
        or 0
    )
    free_to_pool_blocks_total = int(counters.get("free_to_pool_blocks", 0) or 0)
    last_removed_sample = tuple(
        last_snapshot_counters.get("last_removed_block_ids_sample") or ()
    )[:20]
    last_freed_sample = tuple(
        last_snapshot_counters.get("last_freed_block_ids_sample") or ()
    )[:20]
    last_rejected_sample = tuple(
        last_snapshot_counters.get("last_free_rejected_block_ids_sample") or ()
    )[:20]
    removed_but_not_freed_sample = tuple(
        block_id for block_id in last_removed_sample if block_id not in set(last_freed_sample)
    )[:20]
    blocker_reasons = dict(counters.get("blocker_reasons", {}) or {})
    free_to_pool_rejected_reason_counts = {
        "already_freed_or_duplicate": int(
            blocker_reasons.get("double_free_prevented", 0) or 0
        ),
        "not_removed": int(
            blocker_reasons.get("no_removed_demoted_blocks", 0) or 0
        ),
        "not_marked_demoted": int(
            blocker_reasons.get("not_marked_demoted", 0) or 0
        ),
        "still_owned": int(
            (blocker_reasons.get("removed_block_still_owned_by_request", 0) or 0)
            + (blocker_reasons.get("removed_block_still_owned_by_other_request", 0) or 0)
        ),
    }
    total_known_blockers = sum(
        count
        for reason, count in blocker_reasons.items()
        if reason in {
            "double_free_prevented",
            "no_removed_demoted_blocks",
            "not_marked_demoted",
            "removed_block_still_owned_by_request",
            "removed_block_still_owned_by_other_request",
        }
    )
    free_to_pool_rejected_reason_counts["unknown"] = max(
        sum(blocker_reasons.values()) - total_known_blockers,
        0,
    )

    summary = dict(base_summary)
    summary["phase"] = "S5.26"
    summary["goal"] = "observe_block_pool_accounting_after_gated_free_to_pool"
    summary["prompt_mode"] = args.prompt_mode
    summary["prompt_repeats"] = args.prompt_repeats
    summary["num_prompts"] = max(1, args.num_prompts)
    summary["target_prompt_tokens"] = args.target_prompt_tokens
    summary["requested_max_output_tokens"] = requested_max_output_tokens
    summary["max_num_seqs"] = max_num_seqs
    summary["total_prompt_tokens"] = total_prompt_tokens
    summary["avg_prompt_tokens"] = avg_prompt_tokens
    summary["max_prompt_tokens"] = max_prompt_tokens
    summary["estimated_max_active_prompt_tokens"] = (
        estimated_max_active_prompt_tokens
    )
    summary["estimated_max_active_total_tokens"] = (
        estimated_max_active_total_tokens
    )
    summary["gpu_kv_cache_size_tokens"] = gpu_kv_cache_size_tokens
    summary["estimated_pressure_ratio"] = estimated_pressure_ratio
    summary["prefix_caching_disable_requested"] = bool(
        getattr(args, "disable_prefix_caching", False)
    )
    summary["prefix_caching_disable_supported"] = prefix_caching_disable_supported
    summary["prefix_caching_disabled"] = prefix_caching_disabled
    summary["ownership_remove_observed"] = (
        int(summary.get("ownership_remove_succeeded", 0) or 0) > 0
    )
    summary["free_to_pool_observed"] = (
        int(summary.get("free_to_pool_succeeded", 0) or 0) > 0
        or int(summary.get("free_to_pool_calls", 0) or 0) > 0
    )
    summary["live_kv_free_enabled"] = summary["free_to_pool_observed"] or (
        int(summary.get("free_to_pool_attempted", 0) or 0) > 0
    )
    summary["block_pool_accounting_observed"] = (
        int(summary.get("block_pool_free_accounting_observed", 0) or 0) > 0
    )
    summary["first_generation_success"] = bool(generation_success)
    summary["second_generation_success"] = None
    summary["reuse_probe_enabled"] = False
    summary["reuse_probe_success"] = None
    summary["wall_time_seconds"] = wall_time_seconds
    summary["cuda_memory_allocated_before"] = cuda_before.get("cuda_memory_allocated")
    summary["cuda_memory_reserved_before"] = cuda_before.get("cuda_memory_reserved")
    summary["cuda_memory_allocated_after"] = cuda_after.get("cuda_memory_allocated")
    summary["cuda_memory_reserved_after"] = cuda_after.get("cuda_memory_reserved")
    summary["cuda_memory_snapshot_observed"] = bool(
        cuda_before.get("cuda_memory_snapshot_observed")
        and cuda_after.get("cuda_memory_snapshot_observed")
    )
    summary["cuda_memory_snapshot_blocker"] = (
        cuda_before.get("cuda_memory_snapshot_blocker")
        or cuda_after.get("cuda_memory_snapshot_blocker")
    )
    summary["req_to_blocks_removed_total"] = req_to_blocks_removed_total
    summary["ownership_removed_blocks_total"] = ownership_removed_blocks_total
    summary["free_to_pool_blocks_total"] = free_to_pool_blocks_total
    summary["free_to_pool_blocks_last"] = (
        len(last_snapshot_counters.get("last_freed_block_ids_sample") or [])
        if last_snapshot_counters.get("last_freed_block_ids_sample") is not None
        else None
    )
    summary["free_to_pool_blocks_total_or_last"] = (
        free_to_pool_blocks_total
        if free_to_pool_blocks_total > 0
        else summary["free_to_pool_blocks_last"]
    )
    summary["free_to_pool_calls_total"] = int(
        counters.get("free_to_pool_calls", 0) or 0
    )
    summary["block_pool_num_free_blocks_before"] = int(
        counters.get("block_pool_num_free_blocks_before", 0) or 0
    )
    summary["block_pool_num_free_blocks_after"] = int(
        counters.get("block_pool_num_free_blocks_after", 0) or 0
    )
    summary["block_pool_num_free_blocks_delta"] = int(
        counters.get("block_pool_num_free_blocks_delta", 0) or 0
    )
    summary["unique_removed_block_ids_count"] = req_to_blocks_removed_total
    summary["unique_free_to_pool_block_ids_count"] = free_to_pool_blocks_total
    summary["removed_but_not_freed_block_ids_count"] = max(
        req_to_blocks_removed_total - free_to_pool_blocks_total,
        0,
    )
    summary["free_attempt_duplicate_block_ids_count"] = int(
        counters.get("free_to_pool_double_free_prevented", 0) or 0
    )
    summary["free_to_pool_rejected_reason_counts"] = free_to_pool_rejected_reason_counts
    summary["removed_but_not_freed_block_ids_sample"] = list(removed_but_not_freed_sample)
    summary["duplicate_free_block_ids_sample"] = list(last_rejected_sample)
    summary["invariant_failed"] = int(
        summary.get("ownership_remove_invariant_failed", 0) or 0
    )
    summary["cumulative_counters"] = cumulative_counters
    summary["last_snapshot_counters"] = last_snapshot_counters
    summary["summary"] = _build_compact_summary(summary)
    summary["memory_claim_allowed"] = False
    summary["quality_claim_allowed"] = False
    summary["performance_claim_allowed"] = False
    return summary


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.v1.core.kivo_demotion_counters import (
        get_kivo_demotion_counters_snapshot,
        reset_kivo_demotion_counters,
    )

    prompts = build_prompts(
        repeats=args.prompt_repeats,
        num_prompts=args.num_prompts,
        prompt_mode=args.prompt_mode,
        target_prompt_tokens=args.target_prompt_tokens,
        tokenizer=_load_tokenizer(args.model) if args.target_prompt_tokens else None,
        max_model_len=args.max_model_len,
    )
    prompt_char_lengths = [len(prompt) for prompt in prompts]
    export_file = os.getenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE")
    if export_file:
        export_path = Path(export_file)
        if export_path.exists():
            export_path.unlink()
    llm = None
    reset_kivo_demotion_counters()
    cuda_before = _snapshot_cuda_memory()
    wall_start = time.perf_counter()
    prefix_caching_disabled = False
    prefix_caching_disable_supported = False
    try:
        prefix_caching_disable_supported = _supports_enable_prefix_caching(LLM)
        llm_kwargs, prefix_caching_disabled = _build_llm_kwargs(args, LLM)
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_output_tokens,
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
        cuda_after = _snapshot_cuda_memory()
        wall_time_seconds = time.perf_counter() - wall_start
        return build_summary(
            args=args,
            generation_success=True,
            prompt_char_lengths=prompt_char_lengths,
            prompt_token_lengths=prompt_token_lengths,
            parent_counters=parent_counters,
            exported_counters=exported_counters,
            counter_export_file_found=file_found,
            counter_export_pid=export_pid,
            prefix_caching_disabled=prefix_caching_disabled,
            prefix_caching_disable_supported=prefix_caching_disable_supported,
            cuda_before=cuda_before,
            cuda_after=cuda_after,
            wall_time_seconds=wall_time_seconds,
        )
    except Exception as exc:
        parent_counters = get_kivo_demotion_counters_snapshot()
        exported_counters, file_found, export_pid = _load_exported_counters(
            export_file
        )
        cuda_after = _snapshot_cuda_memory()
        wall_time_seconds = time.perf_counter() - wall_start
        return build_summary(
            args=args,
            generation_success=False,
            prompt_char_lengths=prompt_char_lengths,
            prompt_token_lengths=[None] * len(prompts),
            parent_counters=parent_counters,
            exported_counters=exported_counters,
            counter_export_file_found=file_found,
            counter_export_pid=export_pid,
            prefix_caching_disabled=prefix_caching_disabled,
            prefix_caching_disable_supported=prefix_caching_disable_supported,
            cuda_before=cuda_before,
            cuda_after=cuda_after,
            wall_time_seconds=wall_time_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_generation(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("generation_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
