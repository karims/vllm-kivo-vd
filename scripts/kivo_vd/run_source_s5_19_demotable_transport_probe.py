#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a long-prompt probe to observe Kivo demotion transport counters."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SENTENCE = (
    "This is a cached context sentence about retrieval, memory, and attention."
)


def build_long_context_prompt(*, repeats: int = 200) -> str:
    long_context = " ".join([DEFAULT_SENTENCE] * repeats)
    return (
        long_context
        + "\nQuestion: summarize the repeated context in one sentence."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a long-prompt vLLM probe for demotable Kivo transport."
    )
    parser.add_argument("--model", default="facebook/opt-125m")
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
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def build_prompts(*, repeats: int, num_prompts: int) -> list[str]:
    prompt = build_long_context_prompt(repeats=repeats)
    return [prompt for _ in range(max(1, num_prompts))]


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
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
    return llm_kwargs


def build_summary(
    *,
    generation_success: bool,
    prompt_count: int,
    prompt_char_lengths: list[int],
    prompt_token_lengths: list[int | None],
    parent_counters: dict[str, Any],
    exported_counters: dict[str, Any] | None,
    counter_export_file_found: bool,
    counter_export_pid: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    counters = exported_counters if exported_counters is not None else parent_counters
    export_blocker_reasons = {
        "no_apply_summary": int(
            counters.get("demotion_command_export_skipped_no_apply_summary", 0) or 0
        ),
        "apply_not_successful": int(
            counters.get(
                "demotion_command_export_skipped_apply_not_successful", 0
            )
            or 0
        ),
        "missing_request_id": int(
            counters.get("demotion_command_export_skipped_no_request_id", 0) or 0
        ),
        "missing_visible_before_blocks": int(
            counters.get("demotion_command_export_skipped_no_visible_before", 0) or 0
        ),
        "missing_visible_after_blocks": int(
            counters.get("demotion_command_export_skipped_no_visible_after", 0) or 0
        ),
        "empty_candidate_demote_ids": int(
            counters.get(
                "demotion_command_export_skipped_no_candidate_demote_ids", 0
            )
            or 0
        ),
        "empty_after_filter": int(
            counters.get("demotion_command_export_skipped_empty_after_filter", 0)
            or 0
        ),
    }
    worker_envelope_observed = int(counters.get("worker_envelopes_built", 0) or 0) > 0
    scheduler_envelope_observed = (
        int(counters.get("scheduler_envelopes_received", 0) or 0) > 0
    )
    core_command_observed = int(counters.get("core_commands_attempted", 0) or 0) > 0
    manager_mark_demoted_observed = (
        int(counters.get("manager_mark_demoted_attempted", 0) or 0) > 0
    )
    transport_observed = (
        worker_envelope_observed
        or scheduler_envelope_observed
        or core_command_observed
        or manager_mark_demoted_observed
    )
    return {
        "phase": "S5.19",
        "generation_success": generation_success,
        "prompt_count": prompt_count,
        "prompt_char_lengths": prompt_char_lengths,
        "prompt_token_lengths": prompt_token_lengths,
        "parent_counters": parent_counters,
        "exported_counters": exported_counters,
        "counter_export_file_found": counter_export_file_found,
        "counter_export_pid": counter_export_pid,
        "counters": counters,
        "block_table_apply_attempted": int(
            counters.get("block_table_apply_attempted", 0) or 0
        ),
        "block_table_apply_succeeded": int(
            counters.get("block_table_apply_succeeded", 0) or 0
        ),
        "block_table_apply_rejected": int(
            counters.get("block_table_apply_rejected", 0) or 0
        ),
        "filtered_row_plan_attempted": int(
            counters.get("filtered_row_plan_attempted", 0) or 0
        ),
        "filtered_row_plan_succeeded": int(
            counters.get("filtered_row_plan_succeeded", 0) or 0
        ),
        "filtered_row_plan_rejected": int(
            counters.get("filtered_row_plan_rejected", 0) or 0
        ),
        "filtered_row_apply_noop": int(
            counters.get("filtered_row_apply_noop", 0) or 0
        ),
        "filtered_row_changed_count": int(
            counters.get("filtered_row_changed_count", 0) or 0
        ),
        "filtered_row_candidate_drop_count": int(
            counters.get("filtered_row_candidate_drop_count", 0) or 0
        ),
        "demotion_command_export_path_entered": int(
            counters.get("demotion_command_export_path_entered", 0) or 0
        ),
        "demotion_command_export_attempted": int(
            counters.get("demotion_command_export_attempted", 0) or 0
        ),
        "demotion_command_export_succeeded": int(
            counters.get("demotion_command_export_succeeded", 0) or 0
        ),
        "demotion_command_export_blocker_reasons": export_blocker_reasons,
        "visible_before_count": int(
            counters.get("last_visible_before_count", 0) or 0
        ),
        "visible_after_count": int(counters.get("last_visible_after_count", 0) or 0),
        "candidate_demote_count": int(
            counters.get("last_candidate_demote_count", 0) or 0
        ),
        "last_worker_row_raw_count": int(
            counters.get("last_worker_row_raw_count", 0) or 0
        ),
        "last_worker_row_nonzero_count": int(
            counters.get("last_worker_row_nonzero_count", 0) or 0
        ),
        "last_worker_row_unique_count": int(
            counters.get("last_worker_row_unique_count", 0) or 0
        ),
        "last_worker_row_trailing_zero_count": int(
            counters.get("last_worker_row_trailing_zero_count", 0) or 0
        ),
        "last_filtered_keep_count": int(
            counters.get("last_filtered_keep_count", 0) or 0
        ),
        "last_filtered_drop_count": int(
            counters.get("last_filtered_drop_count", 0) or 0
        ),
        "last_filtered_drop_ids_sample": list(
            counters.get("last_filtered_drop_ids_sample", ()) or ()
        ),
        "last_filtered_keep_ids_sample": list(
            counters.get("last_filtered_keep_ids_sample", ()) or ()
        ),
        "filtered_row_changed": counters.get("last_filtered_row_changed"),
        "keep_recent_blocks": int(counters.get("last_keep_recent_blocks", 0) or 0),
        "policy": counters.get("last_policy"),
        "transport_observed": transport_observed,
        "worker_envelope_observed": worker_envelope_observed,
        "scheduler_envelope_observed": scheduler_envelope_observed,
        "core_command_observed": core_command_observed,
        "manager_mark_demoted_observed": manager_mark_demoted_observed,
        "demoted_blocks_marked": int(counters.get("demoted_blocks_marked", 0) or 0),
        "ownership_remove_attempted": int(
            counters.get("ownership_remove_attempted", 0) or 0
        ),
        "ownership_remove_succeeded": int(
            counters.get("ownership_remove_succeeded", 0) or 0
        ),
        "ownership_remove_rejected": int(
            counters.get("ownership_remove_rejected", 0) or 0
        ),
        "ownership_removed_blocks": int(
            counters.get("ownership_removed_blocks", 0) or 0
        ),
        "ownership_remove_invariant_checked": int(
            counters.get("ownership_remove_invariant_checked", 0) or 0
        ),
        "ownership_remove_invariant_failed": int(
            counters.get("ownership_remove_invariant_failed", 0) or 0
        ),
        "ownership_removed_subset_of_marked": int(
            counters.get("ownership_removed_subset_of_marked", 0) or 0
        ),
        "ownership_removed_subset_of_owned": int(
            counters.get("ownership_removed_subset_of_owned", 0) or 0
        ),
        "ownership_removed_absent_after": int(
            counters.get("ownership_removed_absent_after", 0) or 0
        ),
        "ownership_remaining_nonempty": int(
            counters.get("ownership_remaining_nonempty", 0) or 0
        ),
        "ownership_removed_reintroduced": int(
            counters.get("ownership_removed_reintroduced", 0) or 0
        ),
        "ownership_demoted_bookkeeping_cleared": int(
            counters.get("ownership_demoted_bookkeeping_cleared", 0) or 0
        ),
        "ownership_removed_blocks_total": int(
            counters.get("ownership_removed_blocks_total", 0) or 0
        ),
        "free_to_pool_attempted": int(
            counters.get("free_to_pool_attempted", 0) or 0
        ),
        "free_to_pool_succeeded": int(
            counters.get("free_to_pool_succeeded", 0) or 0
        ),
        "free_to_pool_rejected": int(
            counters.get("free_to_pool_rejected", 0) or 0
        ),
        "free_to_pool_blocks": int(counters.get("free_to_pool_blocks", 0) or 0),
        "free_to_pool_double_free_prevented": int(
            counters.get("free_to_pool_double_free_prevented", 0) or 0
        ),
        "block_pool_free_capacity_before": int(
            counters.get("block_pool_free_capacity_before", 0) or 0
        ),
        "block_pool_free_capacity_after": int(
            counters.get("block_pool_free_capacity_after", 0) or 0
        ),
        "block_pool_free_capacity_delta": int(
            counters.get("block_pool_free_capacity_delta", 0) or 0
        ),
        "block_pool_num_free_blocks_before": int(
            counters.get("block_pool_num_free_blocks_before", 0) or 0
        ),
        "block_pool_num_free_blocks_after": int(
            counters.get("block_pool_num_free_blocks_after", 0) or 0
        ),
        "block_pool_num_free_blocks_delta": int(
            counters.get("block_pool_num_free_blocks_delta", 0) or 0
        ),
        "block_pool_free_accounting_observed": int(
            counters.get("block_pool_free_accounting_observed", 0) or 0
        ),
        "block_pool_free_accounting_increased": int(
            counters.get("block_pool_free_accounting_increased", 0) or 0
        ),
        "block_pool_free_accounting_rejected": int(
            counters.get("block_pool_free_accounting_rejected", 0) or 0
        ),
        "block_pool_free_accounting_blocker_reasons": dict(
            counters.get("block_pool_free_accounting_blocker_reasons", {}) or {}
        ),
        "ownership_remaining_blocks_last": int(
            counters.get("ownership_remaining_blocks_last", 0) or 0
        ),
        "last_owned_before_remove_count": int(
            counters.get("last_owned_before_remove_count", 0) or 0
        ),
        "last_owned_after_remove_count": int(
            counters.get("last_owned_after_remove_count", 0) or 0
        ),
        "last_marked_demoted_before_remove_count": int(
            counters.get("last_marked_demoted_before_remove_count", 0) or 0
        ),
        "last_removed_after_absent": counters.get("last_removed_after_absent"),
        "last_removed_block_ids_sample": list(
            counters.get("last_removed_block_ids_sample", ()) or ()
        ),
        "last_remaining_block_ids_sample": list(
            counters.get("last_remaining_block_ids_sample", ()) or ()
        ),
        "last_freed_block_ids_sample": list(
            counters.get("last_freed_block_ids_sample", ()) or ()
        ),
        "last_free_rejected_block_ids_sample": list(
            counters.get("last_free_rejected_block_ids_sample", ()) or ()
        ),
        "req_to_blocks_removed": int(counters.get("req_to_blocks_removed", 0) or 0),
        "free_to_pool_calls": int(counters.get("free_to_pool_calls", 0) or 0),
        "memory_claim_allowed": False,
        "free_to_pool_claim_allowed": False,
        "ownership_removal_claim_allowed": False,
        "error": error,
    }


def _load_exported_counters(path: str | None) -> tuple[dict[str, Any] | None, bool, int | None]:
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


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.v1.core.kivo_demotion_counters import (
        get_kivo_demotion_counters_snapshot,
        reset_kivo_demotion_counters,
    )

    prompts = build_prompts(repeats=args.prompt_repeats, num_prompts=args.num_prompts)
    prompt_char_lengths = [len(prompt) for prompt in prompts]
    export_file = os.getenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE")
    if export_file:
        export_path = Path(export_file)
        if export_path.exists():
            export_path.unlink()
    llm = None
    reset_kivo_demotion_counters()
    try:
        llm = LLM(**_build_llm_kwargs(args))
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
        return build_summary(
            generation_success=True,
            prompt_count=len(prompts),
            prompt_char_lengths=prompt_char_lengths,
            prompt_token_lengths=prompt_token_lengths,
            parent_counters=parent_counters,
            exported_counters=exported_counters,
            counter_export_file_found=file_found,
            counter_export_pid=export_pid,
        )
    except Exception as exc:
        parent_counters = get_kivo_demotion_counters_snapshot()
        exported_counters, file_found, export_pid = _load_exported_counters(
            export_file
        )
        return build_summary(
            generation_success=False,
            prompt_count=len(prompts),
            prompt_char_lengths=prompt_char_lengths,
            prompt_token_lengths=[None] * len(prompts),
            parent_counters=parent_counters,
            exported_counters=exported_counters,
            counter_export_file_found=file_found,
            counter_export_pid=export_pid,
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
    return 0 if summary["generation_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
