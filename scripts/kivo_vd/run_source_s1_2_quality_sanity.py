#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Phase S1.2/S1.3 source-level valid-slot mutation checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.kivo_vd import run_source_s1_gpt2_probe as source_s1

DEFAULT_PROMPTS = [
    "Kivo source quality sanity prompt one.",
    "The quick brown fox",
    "In a distant future,",
    "Machine learning systems",
    "A small experiment",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase S1.2/S1.3 source-level valid-slot mutation checks."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to evaluate. May be repeated. Defaults to a small list.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help=(
            "Active mutation policy to test. May be repeated. Defaults to "
            "mask_oldest_valid_slot, mask_middle_valid_slot, "
            "mask_last_valid_slot."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        default="outputs/kivo_vd/runs/source_s1_2_baseline",
    )
    parser.add_argument(
        "--active-dir",
        default="outputs/kivo_vd/runs/source_s1_2_active",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/source_s1_2_quality_sanity.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/source_s1_2_quality_sanity.md",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _resolve_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    return [str(item) for item in prompts if str(item).strip()]


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return source_s1.load_records(path)


def _run_generation(args: argparse.Namespace) -> dict[str, Any]:
    """Run a single greedy generation using the source-level probe path."""

    return source_s1._run_generation(args)


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [item for item in records if item.get("mutation_attempted") is True]
    applied = [item for item in records if item.get("mutation_applied") is True]
    old_new_differ = [item for item in records if item.get("old_new_differ") is True]
    active_routing = [item for item in records if item.get("active_routing") is True]
    runtime_behavior_changed = [
        item for item in records if item.get("runtime_behavior_changed") is True
    ]
    valid_slot_counts = [
        int(item["valid_slot_count"])
        for item in records
        if isinstance(item.get("valid_slot_count"), int)
    ]
    blocker_reasons = sorted(
        {
            str(item.get("mutation_blocker_reason"))
            for item in records
            if item.get("mutation_blocker_reason")
        }
    )
    return {
        "records_written": len(records),
        "mutation_attempted_count": len(attempted),
        "mutation_applied_count": len(applied),
        "active_routing_count": len(active_routing),
        "runtime_behavior_changed_count": len(runtime_behavior_changed),
        "old_new_differ_count": len(old_new_differ),
        "max_valid_slot_count": max(valid_slot_counts) if valid_slot_counts else None,
        "min_valid_slot_count": min(valid_slot_counts) if valid_slot_counts else None,
        "blocker_reasons": blocker_reasons,
        "measured_runtime_reduction": False,
    }


def _build_probe_args(
    args: argparse.Namespace,
    prompt: str,
    prompt_index: int,
) -> SimpleNamespace:
    baseline_dir = Path(args.baseline_dir)
    active_dir = Path(args.active_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    active_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        baseline_obs_jsonl=str(
            baseline_dir / f"prompt_{prompt_index:02d}_baseline.jsonl"
        ),
        observation_obs_jsonl=str(
            baseline_dir / f"prompt_{prompt_index:02d}_observation.jsonl"
        ),
        active_obs_jsonl=str(
            active_dir / f"prompt_{prompt_index:02d}_active.jsonl"
        ),
        output_json=str(
            active_dir / f"prompt_{prompt_index:02d}_quality_sanity.json"
        ),
        output_md=str(active_dir / f"prompt_{prompt_index:02d}_quality_sanity.md"),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        continue_on_error=args.continue_on_error,
    )


def build_report(
    args: argparse.Namespace,
    *,
    probe_runner=source_s1.build_report,
) -> dict[str, Any]:
    prompts = _resolve_prompts(args)
    prompt_results = []
    stop_early = False
    for prompt_index, prompt in enumerate(prompts):
        probe_args = _build_probe_args(args, prompt, prompt_index)
        probe_report = probe_runner(probe_args)
        active_records = _load_records(probe_args.active_obs_jsonl)
        active_summary = _summarize_records(active_records)
        prompt_report = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "baseline_status": probe_report["baseline_status"],
            "active_status": probe_report["active_status"],
            "baseline_output": probe_report["baseline_output"],
            "active_output": probe_report["active_output"],
            "output_changed": probe_report["output_changed"],
            "active_error": probe_report["active_error"],
            "mutation_attempted_count": active_summary["mutation_attempted_count"],
            "mutation_applied_count": active_summary["mutation_applied_count"],
            "active_routing_count": active_summary["active_routing_count"],
            "runtime_behavior_changed_count": active_summary[
                "runtime_behavior_changed_count"
            ],
            "max_valid_slot_count": active_summary["max_valid_slot_count"],
            "min_valid_slot_count": active_summary["min_valid_slot_count"],
            "old_new_differ_count": active_summary["old_new_differ_count"],
            "blocker_reasons": active_summary["blocker_reasons"],
            "baseline_records_written": probe_report["baseline_records_written"],
            "active_records_written": active_summary["records_written"],
            "measured_runtime_reduction": False,
            "baseline_obs_jsonl": probe_args.baseline_obs_jsonl,
            "active_obs_jsonl": probe_args.active_obs_jsonl,
        }
        prompt_results.append(prompt_report)
        if (
            not args.continue_on_error
            and (
                probe_report["baseline_status"] != "succeeded"
                or probe_report["active_status"] != "succeeded"
            )
        ):
            stop_early = True
            break

    total_prompts = len(prompt_results)
    baseline_success_count = sum(
        item["baseline_status"] == "succeeded" for item in prompt_results
    )
    active_success_count = sum(
        item["active_status"] == "succeeded" for item in prompt_results
    )
    mutation_applied_prompt_count = sum(
        item["mutation_applied_count"] > 0 for item in prompt_results
    )
    output_changed_count = sum(item["output_changed"] for item in prompt_results)
    output_unchanged_count = total_prompts - output_changed_count
    total_mutation_applied_records = sum(
        item["mutation_applied_count"] for item in prompt_results
    )
    total_active_records = sum(
        item["active_records_written"] for item in prompt_results
    )
    quality_sanity_passed = bool(
        total_prompts > 0
        and baseline_success_count == total_prompts
        and active_success_count == total_prompts
        and total_mutation_applied_records > 0
    )
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "prompts": prompts,
        "total_prompts": total_prompts,
        "baseline_success_count": baseline_success_count,
        "active_success_count": active_success_count,
        "mutation_applied_prompt_count": mutation_applied_prompt_count,
        "output_changed_count": output_changed_count,
        "output_unchanged_count": output_unchanged_count,
        "total_mutation_applied_records": total_mutation_applied_records,
        "total_active_records": total_active_records,
        "measured_runtime_reduction": False,
        "quality_sanity_passed": quality_sanity_passed,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "stop_early": stop_early,
        "prompt_results": prompt_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S1.2 Source-Level Valid-Slot Mutation Quality Sanity",
        "",
        "## Summary",
        "",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        f"- Active success count: `{report['active_success_count']}`",
        (
            "- Mutation-applied prompt count: "
            f"`{report['mutation_applied_prompt_count']}`"
        ),
        f"- Output changed count: `{report['output_changed_count']}`",
        f"- Output unchanged count: `{report['output_unchanged_count']}`",
        (
            "- Total mutation-applied records: "
            f"`{report['total_mutation_applied_records']}`"
        ),
        f"- Total active records: `{report['total_active_records']}`",
        "- Measured runtime reduction: `false`",
        (
            "- Quality sanity passed: "
            f"`{report['quality_sanity_passed']}`"
        ),
        (
            "- Selected-attention claim allowed: "
            f"`{report['selected_attention_claim_allowed']}`"
        ),
        (
            "- Performance claim allowed: "
            f"`{report['performance_claim_allowed']}`"
        ),
        "",
        "## Prompt Results",
        "",
    ]
    for item in report["prompt_results"]:
        lines.extend([
            f"### Prompt {item['prompt_index']}",
            "",
            f"- Prompt: `{item['prompt']}`",
            f"- Baseline status: `{item['baseline_status']}`",
            f"- Active status: `{item['active_status']}`",
            f"- Output changed: `{item['output_changed']}`",
            f"- Mutation applied count: `{item['mutation_applied_count']}`",
            f"- Active routing count: `{item['active_routing_count']}`",
            (
                "- Runtime behavior changed count: "
                f"`{item['runtime_behavior_changed_count']}`"
            ),
            f"- Max valid slot count: `{item['max_valid_slot_count']}`",
            f"- Min valid slot count: `{item['min_valid_slot_count']}`",
            f"- Old/new differ count: `{item['old_new_differ_count']}`",
            f"- Blocker reasons: `{item['blocker_reasons']}`",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _safe_policy_name(policy: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in policy)


def _resolve_policies(args: argparse.Namespace) -> list[str]:
    policies = args.policy if getattr(args, "policy", None) else None
    if policies:
        return [str(item) for item in policies if str(item).strip()]
    return [
        "mask_oldest_valid_slot",
        "mask_middle_valid_slot",
        "mask_last_valid_slot",
    ]


def _policy_priority(policy: str) -> int:
    priorities = {
        "mask_oldest_valid_slot": 0,
        "mask_middle_valid_slot": 1,
        "mask_last_valid_slot": 2,
        "noop_valid_slot_shadow": 3,
        "mask_last_slot": 4,
    }
    return priorities.get(policy, 99)


def _clear_source_env() -> None:
    for key in [
        "KIVO_SOURCE_ENABLE",
        "KIVO_SOURCE_OBS_PATH",
        "KIVO_SOURCE_ACTIVE",
        "KIVO_SOURCE_POLICY",
        "KIVO_SOURCE_MAX_MUTATIONS",
        "KIVO_SOURCE_FAIL_CLOSED",
    ]:
        os.environ.pop(key, None)


def _set_active_env(observation_path: str, policy: str) -> None:
    os.environ["KIVO_SOURCE_ENABLE"] = "1"
    os.environ["KIVO_SOURCE_OBS_PATH"] = str(observation_path)
    os.environ["KIVO_SOURCE_FAIL_CLOSED"] = "1"
    os.environ["KIVO_SOURCE_ACTIVE"] = "1"
    os.environ["KIVO_SOURCE_POLICY"] = policy
    os.environ["KIVO_SOURCE_MAX_MUTATIONS"] = "1"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _write(path, json.dumps(payload, indent=2) + "\n")


def _load_prompt_records(path: str | Path) -> list[dict[str, Any]]:
    return _load_records(path)


def _policy_prompt_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_records(records)
    summary["blocker_reasons"] = sorted(
        {str(item.get("mutation_blocker_reason")) for item in records if item.get("mutation_blocker_reason")}
    )
    return summary


def _summarize_policy_results(
    prompt_results: list[dict[str, Any]],
    policies: list[str],
) -> dict[str, dict[str, Any]]:
    per_policy: dict[str, dict[str, Any]] = {}
    for policy in policies:
        active_results = [
            item["per_policy"][policy] for item in prompt_results
            if policy in item.get("per_policy", {})
        ]
        output_changed_count = sum(item["output_changed"] for item in active_results)
        mutation_applied_prompt_count = sum(
            item["mutation_applied_count"] > 0 for item in active_results
        )
        per_policy[policy] = {
            "active_success_count": sum(
                item["active_status"] == "succeeded" for item in active_results
            ),
            "mutation_applied_prompt_count": mutation_applied_prompt_count,
            "output_changed_count": output_changed_count,
            "output_unchanged_count": len(active_results) - output_changed_count,
            "total_mutation_applied_records": sum(
                item["mutation_applied_count"] for item in active_results
            ),
            "total_active_records": sum(
                item["active_records_written"] for item in active_results
            ),
            "old_new_differ_count": sum(
                item["old_new_differ_count"] for item in active_results
            ),
            "blocker_reasons": sorted(
                {
                    reason
                    for item in active_results
                    for reason in item["blocker_reasons"]
                }
            ),
            "max_valid_slot_count": max(
                (
                    item["max_valid_slot_count"]
                    for item in active_results
                    if item["max_valid_slot_count"] is not None
                ),
                default=None,
            ),
            "min_valid_slot_count": min(
                (
                    item["min_valid_slot_count"]
                    for item in active_results
                    if item["min_valid_slot_count"] is not None
                ),
                default=None,
            ),
            "measured_runtime_reduction": False,
            "prompt_results": active_results,
        }
    return per_policy


def _select_best_drift_policy(per_policy: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        {
            "policy": policy,
            "output_changed_count": summary["output_changed_count"],
            "mutation_applied_prompt_count": summary["mutation_applied_prompt_count"],
            "active_success_count": summary["active_success_count"],
        }
        for policy, summary in per_policy.items()
        if summary["mutation_applied_prompt_count"] > 0
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["output_changed_count"],
            _policy_priority(item["policy"]),
            item["policy"],
        )
    )
    return candidates[0]


def build_drift_report(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _resolve_prompts(args)
    policies = _resolve_policies(args)
    baseline_dir = Path(args.baseline_dir)
    active_dir = Path(args.active_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    active_dir.mkdir(parents=True, exist_ok=True)

    prompt_results: list[dict[str, Any]] = []
    stop_early = False
    _clear_source_env()
    for prompt_index, prompt in enumerate(prompts):
        probe_args = _build_probe_args(args, prompt, prompt_index)

        _clear_source_env()
        baseline = _run_generation(probe_args)
        _write_json(
            baseline_dir / f"prompt_{prompt_index:02d}_baseline.json",
            baseline,
        )

        prompt_result: dict[str, Any] = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "baseline_status": baseline["status"],
            "baseline_output": baseline["output_text"],
            "baseline_error": baseline["error"],
            "measured_runtime_reduction": False,
            "per_policy": {},
        }

        if baseline["status"] != "succeeded" and not args.continue_on_error:
            prompt_results.append(prompt_result)
            stop_early = True
            break

        for policy in policies:
            active_obs_path = active_dir / (
                f"prompt_{prompt_index:02d}_{_safe_policy_name(policy)}_active.jsonl"
            )
            if active_obs_path.exists():
                active_obs_path.unlink()
            _set_active_env(str(active_obs_path), policy)
            active = _run_generation(probe_args)
            active_records = _load_prompt_records(active_obs_path)
            policy_summary = _policy_prompt_summary(active_records)
            policy_result = {
                "policy": policy,
                "active_status": active["status"],
                "active_output": active["output_text"],
                "active_error": active["error"],
                "output_changed": bool(
                    baseline["status"] == "succeeded"
                    and active["status"] == "succeeded"
                    and baseline["output_text"] != active["output_text"]
                ),
                "mutation_attempted_count": policy_summary["mutation_attempted_count"],
                "mutation_applied_count": policy_summary["mutation_applied_count"],
                "active_routing_count": policy_summary["active_routing_count"],
                "runtime_behavior_changed_count": policy_summary[
                    "runtime_behavior_changed_count"
                ],
                "old_new_differ_count": policy_summary["old_new_differ_count"],
                "max_valid_slot_count": policy_summary["max_valid_slot_count"],
                "min_valid_slot_count": policy_summary["min_valid_slot_count"],
                "blocker_reasons": policy_summary["blocker_reasons"],
                "active_records_written": policy_summary["records_written"],
                "measured_runtime_reduction": False,
                "active_obs_jsonl": str(active_obs_path),
            }
            prompt_result["per_policy"][policy] = policy_result
            _write_json(
                active_dir / (
                    f"prompt_{prompt_index:02d}_{_safe_policy_name(policy)}_active.json"
                ),
                {"generation": active, "summary": policy_summary},
            )
            if (
                not args.continue_on_error
                and active["status"] != "succeeded"
            ):
                stop_early = True
                break
        prompt_results.append(prompt_result)
        if stop_early:
            break

    baseline_success_count = sum(
        item["baseline_status"] == "succeeded" for item in prompt_results
    )
    per_policy = _summarize_policy_results(prompt_results, policies)
    best_drift_policy = _select_best_drift_policy(per_policy)
    quality_sanity_passed = bool(
        prompt_results
        and baseline_success_count == len(prompt_results)
        and any(
            summary["active_success_count"] == len(prompt_results)
            and summary["mutation_applied_prompt_count"] > 0
            for summary in per_policy.values()
        )
    )
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "prompts": prompts,
        "policies": policies,
        "total_prompts": len(prompt_results),
        "baseline_success_count": baseline_success_count,
        "per_policy": per_policy,
        "best_drift_policy": best_drift_policy,
        "measured_runtime_reduction": False,
        "quality_sanity_passed": quality_sanity_passed,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "stop_early": stop_early,
        "prompt_results": prompt_results,
    }


def render_drift_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase S1.3 Source-Level Policy Drift",
        "",
        "## Summary",
        "",
        f"- Total prompts: `{report['total_prompts']}`",
        f"- Policies tested: `{report['policies']}`",
        f"- Baseline success count: `{report['baseline_success_count']}`",
        f"- Quality sanity passed: `{report['quality_sanity_passed']}`",
        (
            "- Selected-attention claim allowed: "
            f"`{report['selected_attention_claim_allowed']}`"
        ),
        (
            "- Performance claim allowed: "
            f"`{report['performance_claim_allowed']}`"
        ),
        "- Measured runtime reduction: `false`",
        "",
        "## Best Drift Policy",
        "",
        f"- Best drift policy: `{report['best_drift_policy']}`",
        "",
        "## Per-Policy Summary",
        "",
    ]
    for policy, summary in report["per_policy"].items():
        lines.extend([
            f"### {policy}",
            "",
            f"- Active success count: `{summary['active_success_count']}`",
            (
                "- Mutation-applied prompt count: "
                f"`{summary['mutation_applied_prompt_count']}`"
            ),
            f"- Output changed count: `{summary['output_changed_count']}`",
            f"- Output unchanged count: `{summary['output_unchanged_count']}`",
            (
                "- Total mutation-applied records: "
                f"`{summary['total_mutation_applied_records']}`"
            ),
            f"- Total active records: `{summary['total_active_records']}`",
            f"- Old/new differ count: `{summary['old_new_differ_count']}`",
            f"- Max valid slot count: `{summary['max_valid_slot_count']}`",
            f"- Min valid slot count: `{summary['min_valid_slot_count']}`",
            f"- Blocker reasons: `{summary['blocker_reasons']}`",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_drift_report(args)
    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_drift_markdown(report))
    print(
        json.dumps(
            {
                "total_prompts": report["total_prompts"],
                "baseline_success_count": report["baseline_success_count"],
                "policies": report["policies"],
                "best_drift_policy": report["best_drift_policy"],
                "quality_sanity_passed": report["quality_sanity_passed"],
                "selected_attention_claim_allowed": report[
                    "selected_attention_claim_allowed"
                ],
                "performance_claim_allowed": report["performance_claim_allowed"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            separators=(",", ":"),
        )
    )
    failed = not report["quality_sanity_passed"]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
