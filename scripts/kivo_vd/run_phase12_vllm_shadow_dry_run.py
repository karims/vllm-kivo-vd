#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate a real vLLM environment and the Phase 12 shadow event path."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any, Callable

from phase12_vllm_runtime_touchpoint import (
    observe_phase12_decode_shadow_metadata,
)
from validate_phase12_shadow_event import load_events, validate_events

DEFAULT_PROMPT = "Kivo Phase 12 shadow dry run."
SHADOW_LAYERS = (0, 5, 8, 11)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real vLLM environment and Phase 12 shadow dry-run."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--enable-shadow", action="store_true")
    parser.add_argument(
        "--shadow-output-jsonl",
        default="outputs/kivo_vd/phase12_vllm_shadow_dry_run_events.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/phase12_vllm_shadow_dry_run.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/phase12_vllm_shadow_dry_run.md",
    )
    parser.add_argument("--skip-vllm-generation", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--prefer-installed-vllm",
        action="store_true",
        help=(
            "Remove repo-root import entries before loading vLLM so an "
            "installed wheel is preferred over an unbuilt source tree."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def sanitize_sys_path_for_installed_vllm(
    repo_root: Path,
) -> dict[str, Any]:
    """Remove only sys.path entries that resolve to the repository root."""

    resolved_repo_root = repo_root.resolve()
    removed_paths: list[str] = []
    kept_paths: list[str] = []
    for entry in sys.path:
        display_entry = entry or ""
        try:
            resolved_entry = (
                Path.cwd().resolve()
                if entry == ""
                else Path(entry).expanduser().resolve()
            )
        except (OSError, RuntimeError):
            kept_paths.append(entry)
            continue
        if resolved_entry == resolved_repo_root:
            removed_paths.append(display_entry)
        else:
            kept_paths.append(entry)
    sys.path[:] = kept_paths
    return {
        "prefer_installed_vllm": True,
        "sys_path_sanitized": True,
        "removed_paths": removed_paths,
        "kept_paths_preview": kept_paths[:12],
    }


def _default_import_path_report() -> dict[str, Any]:
    return {
        "prefer_installed_vllm": False,
        "sys_path_sanitized": False,
        "removed_paths": [],
        "kept_paths_preview": sys.path[:12],
    }


def _classify_vllm_source(
    source: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    if not source:
        return {
            "vllm_import_source": source,
            "vllm_source_is_repo_local": False,
            "vllm_source_is_site_packages": False,
        }
    try:
        resolved_source = Path(source).expanduser().resolve()
        resolved_repo_root = repo_root.resolve()
        is_repo_local = resolved_source.is_relative_to(resolved_repo_root)
        is_site_packages = any(
            part in {"site-packages", "dist-packages"}
            for part in resolved_source.parts
        )
    except (OSError, RuntimeError):
        is_repo_local = False
        is_site_packages = False
    return {
        "vllm_import_source": source,
        "vllm_source_is_repo_local": is_repo_local,
        "vllm_source_is_site_packages": is_site_packages,
    }


def _import_status(
    module_name: str,
    import_module: Callable[[str], Any],
) -> tuple[dict[str, Any], Any | None]:
    try:
        module = import_module(module_name)
        return {
            "ok": True,
            "module": module_name,
            "version": getattr(module, "__version__", None),
            "file": str(getattr(module, "__file__", "") or ""),
            "error_type": None,
            "error": None,
        }, module
    except Exception as exc:
        return {
            "ok": False,
            "module": module_name,
            "version": None,
            "file": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, None


def collect_environment_report(
    import_module: Callable[[str], Any] = importlib.import_module,
    *,
    repo_root: Path = REPO_ROOT,
    import_path_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch_status, torch_module = _import_status("torch", import_module)
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        cuda_available = bool(
            cuda is not None
            and hasattr(cuda, "is_available")
            and cuda.is_available()
        )
        torch_status.update({
            "cuda_version": getattr(
                getattr(torch_module, "version", None),
                "cuda",
                None,
            ),
            "cuda_available": cuda_available,
            "gpu_name": (
                str(cuda.get_device_name(0))
                if cuda_available and hasattr(cuda, "get_device_name")
                else None
            ),
        })
    else:
        torch_status.update({
            "cuda_version": None,
            "cuda_available": False,
            "gpu_name": None,
        })

    vllm_status, _ = _import_status("vllm", import_module)
    extension_statuses = {}
    for module_name in (
        "vllm._C",
        "vllm._C_stable_libtorch",
        "vllm.vllm_flash_attn",
    ):
        status, _ = _import_status(module_name, import_module)
        extension_statuses[module_name] = status

    path_report = import_path_report or _default_import_path_report()
    source_report = _classify_vllm_source(
        vllm_status["file"],
        repo_root,
    )
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_status,
        "vllm": vllm_status,
        "extensions": extension_statuses,
        "prefer_installed_vllm": path_report[
            "prefer_installed_vllm"
        ],
        "sys_path_sanitized": path_report["sys_path_sanitized"],
        "removed_sys_path_entries": path_report["removed_paths"],
        "kept_sys_path_preview": path_report["kept_paths_preview"],
        **source_report,
    }


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "seed": args.seed,
        "enforce_eager": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
    }


def run_vllm_generation(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(**_build_llm_kwargs(args))
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        use_tqdm=False,
    )
    first = outputs[0] if outputs else None
    prompt_token_ids = getattr(first, "prompt_token_ids", None)
    candidates = getattr(first, "outputs", None) if first is not None else None
    text = str(candidates[0].text) if candidates else ""
    return {
        "status": "succeeded",
        "output_text": text,
        "prompt_token_length": (
            len(prompt_token_ids) if prompt_token_ids is not None else None
        ),
        "error_type": None,
        "error": None,
    }


def _estimate_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt.split()))


def emit_shadow_events(
    *,
    output_jsonl: str,
    context_token_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_path = Path(output_jsonl)
    if output_path.exists():
        output_path.unlink()
    block_size = 16
    total_blocks = max(1, math.ceil(context_token_count / block_size))
    env = {
        "KIVO_PHASE12_SHADOW_ENABLED": "1",
        "KIVO_PHASE12_SHADOW_OUTPUT": output_jsonl,
        "KIVO_PHASE12_RATIO_POLICY": (
            "balanced=0:0.60,5:0.45,8:0.45,11:0.60"
        ),
        "KIVO_PHASE12_BLOCK_SIZE": str(block_size),
        "KIVO_PHASE12_PREVIEW_ONLY": "0",
    }
    rng = random.Random(seed)
    results = []
    for layer_idx in SHADOW_LAYERS:
        scores = {
            block_id: rng.uniform(-1.0, 1.0)
            for block_id in range(total_blocks)
        }
        results.append(observe_phase12_decode_shadow_metadata(
            request_id="phase12-vllm-shadow-dry-run",
            sequence_id="sequence-0",
            layer_idx=layer_idx,
            step_idx=0,
            context_token_count=context_token_count,
            total_context_blocks=total_blocks,
            block_ids=list(range(total_blocks)),
            scores=scores,
            metadata={
                "source": "runtime_adjacent_post_generation",
                "synthetic_scores": True,
            },
            env=env,
        ))
    validation = validate_events(load_events(output_path))
    return results, validation


def _status_from_exception(exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "output_text": None,
        "prompt_token_length": None,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def build_report(
    args: argparse.Namespace,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    generation_fn: Callable[[argparse.Namespace], dict[str, Any]] = (
        run_vllm_generation
    ),
    sanitize_fn: Callable[[Path], dict[str, Any]] = (
        sanitize_sys_path_for_installed_vllm
    ),
) -> dict[str, Any]:
    import_path_report = (
        sanitize_fn(REPO_ROOT)
        if args.prefer_installed_vllm
        else _default_import_path_report()
    )
    environment = collect_environment_report(
        import_module,
        import_path_report=import_path_report,
    )
    generation = {
        "status": "skipped",
        "output_text": None,
        "prompt_token_length": None,
        "error_type": None,
        "error": None,
    }
    if not args.skip_vllm_generation:
        try:
            generation = generation_fn(args)
        except Exception as exc:
            generation = _status_from_exception(exc)
            if not args.continue_on_error:
                raise

    shadow = {
        "enabled": bool(args.enable_shadow),
        "status": "disabled",
        "output_jsonl": None,
        "events_requested": 0,
        "events_written": 0,
        "validation": None,
        "error_type": None,
        "error": None,
    }
    if args.enable_shadow:
        context_tokens = (
            generation["prompt_token_length"]
            or _estimate_prompt_tokens(args.prompt)
        )
        try:
            results, validation = emit_shadow_events(
                output_jsonl=args.shadow_output_jsonl,
                context_token_count=context_tokens,
                seed=args.seed,
            )
            shadow.update({
                "status": (
                    "succeeded"
                    if validation["validation_passed"]
                    else "validation_failed"
                ),
                "output_jsonl": args.shadow_output_jsonl,
                "events_requested": len(SHADOW_LAYERS),
                "events_written": sum(
                    result["event_written"] for result in results
                ),
                "validation": validation,
            })
        except Exception as exc:
            shadow.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if not args.continue_on_error:
                raise

    generation_ready = generation["status"] == "succeeded"
    shadow_ready = (
        shadow["status"] == "succeeded"
        if args.enable_shadow
        else False
    )
    environment_ready = bool(
        environment["torch"]["ok"]
        and environment["torch"]["cuda_available"]
        and environment["vllm"]["ok"]
        and environment["extensions"]["vllm._C"]["ok"]
        and (
            not args.prefer_installed_vllm
            or (
                environment["vllm_source_is_site_packages"]
                and not environment["vllm_source_is_repo_local"]
            )
        )
    )
    installed_source_mismatch = bool(
        args.prefer_installed_vllm
        and (
            environment["vllm_source_is_repo_local"]
            or not environment["vllm_source_is_site_packages"]
        )
    )
    return {
        "config": {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "enable_shadow": bool(args.enable_shadow),
            "skip_vllm_generation": bool(args.skip_vllm_generation),
            "prefer_installed_vllm": bool(args.prefer_installed_vllm),
        },
        "environment": environment,
        "generation": generation,
        "shadow": shadow,
        "readiness": {
            "phase12_6_runtime_hook_ready": (
                environment_ready and generation_ready and shadow_ready
            ),
            "environment_ready": environment_ready,
            "generation_ready": generation_ready,
            "shadow_events_ready": shadow_ready,
            "recommendation": (
                "Consider one reviewed opt-in runtime hook in Phase 12.6."
                if environment_ready and generation_ready and shadow_ready
                else (
                    "Installed vLLM was requested, but import provenance "
                    "was not a non-repo site-packages path."
                    if installed_source_mismatch
                    else (
                        "Resolve environment/generation/event validation "
                        "gaps before an actual runtime hook."
                    )
                )
            ),
        },
        "dry_run_only": True,
        "shadow_only": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "no_attention_kernel_change": True,
        "no_kv_cache_mutation": True,
        "no_scheduler_change": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    environment = report["environment"]
    generation = report["generation"]
    shadow = report["shadow"]
    readiness = report["readiness"]
    shadow_validation_passed = bool(
        shadow["validation"]
        and shadow["validation"]["validation_passed"]
    )
    extensions = environment["extensions"]
    core_extension_ok = extensions["vllm._C"]["ok"]
    stable_libtorch_ok = extensions["vllm._C_stable_libtorch"]["ok"]
    flash_attention_ok = extensions["vllm.vllm_flash_attn"]["ok"]
    lines = [
        "# Kivo-VD Phase 12.5 vLLM Shadow Dry-Run",
        "",
        "## Environment",
        "",
        f"- Python: `{environment['python_version']}`",
        f"- Platform: `{environment['platform']}`",
        f"- Torch import: `{str(environment['torch']['ok']).lower()}`",
        f"- Torch version: `{environment['torch']['version']}`",
        f"- Torch CUDA version: `{environment['torch']['cuda_version']}`",
        (
            "- CUDA available: "
            f"`{str(environment['torch']['cuda_available']).lower()}`"
        ),
        f"- GPU: `{environment['torch']['gpu_name']}`",
        f"- vLLM import: `{str(environment['vllm']['ok']).lower()}`",
        f"- vLLM version: `{environment['vllm']['version']}`",
        f"- vLLM path: `{environment['vllm']['file']}`",
        (
            "- Prefer installed vLLM: "
            f"`{str(environment['prefer_installed_vllm']).lower()}`"
        ),
        (
            "- `sys.path` sanitized: "
            f"`{str(environment['sys_path_sanitized']).lower()}`"
        ),
        (
            "- Removed `sys.path` entries: "
            f"`{environment['removed_sys_path_entries']}`"
        ),
        (
            "- vLLM import source: "
            f"`{environment['vllm_import_source']}`"
        ),
        (
            "- vLLM source is repo-local: "
            f"`{str(environment['vllm_source_is_repo_local']).lower()}`"
        ),
        (
            "- vLLM source is site-packages: "
            f"`{str(environment['vllm_source_is_site_packages']).lower()}`"
        ),
        (
            "- `vllm._C` import: "
            f"`{str(core_extension_ok).lower()}`"
        ),
        (
            "- `vllm._C_stable_libtorch` import: "
            f"`{str(stable_libtorch_ok).lower()}`"
        ),
        (
            "- `vllm.vllm_flash_attn` import: "
            f"`{str(flash_attention_ok).lower()}`"
        ),
        "",
        "## Generation Smoke",
        "",
        f"- Status: `{generation['status']}`",
        f"- Prompt tokens: `{generation['prompt_token_length']}`",
        f"- Output text: `{generation['output_text']}`",
        f"- Error: `{generation['error']}`",
        "",
        "## Shadow Events",
        "",
        f"- Enabled: `{str(shadow['enabled']).lower()}`",
        f"- Status: `{shadow['status']}`",
        f"- Events written: `{shadow['events_written']}`",
        f"- Output: `{shadow['output_jsonl']}`",
        (
            "- Validation passed: "
            f"`{str(shadow_validation_passed).lower()}`"
        ),
        "",
        "## Readiness",
        "",
        (
            "- Phase 12.6 runtime-hook ready: "
            f"`{str(readiness['phase12_6_runtime_hook_ready']).lower()}`"
        ),
        f"- Recommendation: {readiness['recommendation']}",
        "",
        "## Caveats",
        "",
        "- This is a dry-run and environment validation workflow.",
        "- Shadow scores are synthetic and emitted after generation.",
        "- No automatic vLLM runtime hook is installed.",
        "- Attention, KV cache, block tables, and scheduling are unchanged.",
        "- No measured memory, latency, or quality claim is made.",
    ]
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(args)
    except Exception as exc:
        report = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "dry_run_only": True,
            "shadow_only": True,
            "active_routing": False,
            "measured_runtime_reduction": False,
            "no_attention_kernel_change": True,
            "no_kv_cache_mutation": True,
            "no_scheduler_change": True,
        }
        _write(args.output_json, json.dumps(report, indent=2) + "\n")
        _write(
            args.output_md,
            "# Kivo-VD Phase 12.5 vLLM Shadow Dry-Run\n\n"
            f"- Failed: `{report['error_type']}: {report['error']}`\n",
        )
        print(json.dumps(report, separators=(",", ":")))
        return 1

    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "generation_status": report["generation"]["status"],
        "shadow_status": report["shadow"]["status"],
        "phase12_6_runtime_hook_ready": report["readiness"][
            "phase12_6_runtime_hook_ready"
        ],
        "output_json": args.output_json,
        "output_md": args.output_md,
        "dry_run_only": True,
        "active_routing": False,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
