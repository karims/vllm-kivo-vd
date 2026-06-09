#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Probe vLLM general-plugin discovery with a marker-only Kivo plugin."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe installed vLLM loading the kivo_shadow plugin."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="Kivo Phase 12 plugin probe.")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--marker-path",
        default="outputs/kivo_vd/runs/phase12_6_plugin_marker.json",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/kivo_vd/runs/phase12_6_plugin_probe.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/kivo_vd/runs/phase12_6_plugin_probe.md",
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _set_probe_environment(marker_path: str) -> None:
    os.environ["VLLM_PLUGINS"] = "kivo_shadow"
    os.environ["KIVO_SHADOW_PLUGIN_MARKER"] = marker_path


def _read_marker(path: str | Path) -> dict[str, Any] | None:
    marker_path = Path(path)
    if not marker_path.exists():
        return None
    value = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("plugin marker must contain a JSON object")
    return value


def _load_vllm_and_plugins(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    vllm = import_module("vllm")
    plugins = import_module("vllm.plugins")
    plugins.load_general_plugins()
    return {
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_file": str(getattr(vllm, "__file__", "") or ""),
    }


def _run_generation(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        seed=args.seed,
        enforce_eager=True,
        gpu_memory_utilization=0.05,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        use_tqdm=False,
    )
    candidates = getattr(outputs[0], "outputs", None) if outputs else None
    return {
        "status": "succeeded",
        "output_text": str(candidates[0].text) if candidates else "",
        "error_type": None,
        "error": None,
    }


def build_probe_report(
    args: argparse.Namespace,
    *,
    load_fn: Callable[[], dict[str, Any]] | None = None,
    generation_fn: Callable[[argparse.Namespace], dict[str, Any]] = (
        _run_generation
    ),
) -> dict[str, Any]:
    marker_path = Path(args.marker_path)
    if marker_path.exists():
        marker_path.unlink()
    _set_probe_environment(str(marker_path))

    load_status = {
        "status": "not_started",
        "error_type": None,
        "error": None,
        "vllm_version": None,
        "vllm_file": None,
    }
    try:
        load_status.update((load_fn or _load_vllm_and_plugins)())
        load_status["status"] = "succeeded"
    except Exception as exc:
        load_status.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        if not args.continue_on_error:
            raise

    generation = {
        "status": "skipped",
        "output_text": None,
        "error_type": None,
        "error": None,
    }
    if not args.skip_generation and load_status["status"] == "succeeded":
        try:
            generation = generation_fn(args)
        except Exception as exc:
            generation = {
                "status": "failed",
                "output_text": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if not args.continue_on_error:
                raise

    try:
        marker = _read_marker(marker_path)
        marker_error = None
    except Exception as exc:
        marker = None
        marker_error = f"{type(exc).__name__}: {exc}"
    marker_written = marker is not None
    plugin_loaded = bool(
        marker
        and marker.get("loaded") is True
        and marker.get("plugin_name") == "kivo_shadow"
    )
    candidate = bool(
        load_status["status"] == "succeeded"
        and plugin_loaded
        and (
            args.skip_generation
            or generation["status"] == "succeeded"
        )
    )
    return {
        "plugin_name": "kivo_shadow",
        "plugin_marker_written": marker_written,
        "plugin_loaded": plugin_loaded,
        "marker_path": str(marker_path),
        "marker": marker,
        "marker_error": marker_error,
        "vllm_file": load_status["vllm_file"],
        "vllm_version": load_status["vllm_version"],
        "plugin_load_status": load_status["status"],
        "plugin_load_error": load_status["error"],
        "generation_status": generation["status"],
        "generation_output_text": generation["output_text"],
        "generation_error": generation["error"],
        "phase12_6b_plugin_shadow_hook_candidate": candidate,
        "dry_run_only": True,
        "shadow_only": True,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "runtime_monkeypatch_applied": False,
        "scheduler_behavior_changed": False,
        "attention_behavior_changed": False,
        "kv_cache_mutated": False,
        "block_table_mutated": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Kivo-VD Phase 12.6A Plugin Feasibility Probe",
        "",
        "## Plugin Discovery",
        "",
        f"- Plugin load status: `{report['plugin_load_status']}`",
        f"- Marker written: `{str(report['plugin_marker_written']).lower()}`",
        f"- Plugin loaded: `{str(report['plugin_loaded']).lower()}`",
        f"- Marker path: `{report['marker_path']}`",
        f"- vLLM version: `{report['vllm_version']}`",
        f"- vLLM file: `{report['vllm_file']}`",
        f"- Load error: `{report['plugin_load_error']}`",
        "",
        "## Generation",
        "",
        f"- Status: `{report['generation_status']}`",
        f"- Output text: `{report['generation_output_text']}`",
        f"- Error: `{report['generation_error']}`",
        "",
        "## Feasibility",
        "",
        (
            "- Phase 12.6B plugin shadow-hook candidate: "
            f"`{str(report['phase12_6b_plugin_shadow_hook_candidate']).lower()}`"
        ),
        "",
        "## Caveats",
        "",
        "- This probe confirms plugin discovery and invocation only.",
        "- It does not prove access to block tables or decode metadata.",
        "- It applies no runtime monkeypatch.",
        "- Scheduler, attention, KV cache, and model output are unchanged.",
        "- No memory, latency, or quality claim is made.",
    ]
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_probe_report(args)
    except Exception as exc:
        report = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "phase12_6b_plugin_shadow_hook_candidate": False,
            "dry_run_only": True,
            "active_routing": False,
            "runtime_monkeypatch_applied": False,
        }
        _write(args.output_json, json.dumps(report, indent=2) + "\n")
        _write(
            args.output_md,
            "# Kivo-VD Phase 12.6A Plugin Feasibility Probe\n\n"
            f"- Failed: `{report['error_type']}: {report['error']}`\n",
        )
        print(json.dumps(report, separators=(",", ":")))
        return 1

    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    _write(args.output_md, render_markdown(report))
    print(json.dumps({
        "plugin_loaded": report["plugin_loaded"],
        "plugin_marker_written": report["plugin_marker_written"],
        "generation_status": report["generation_status"],
        "phase12_6b_plugin_shadow_hook_candidate": report[
            "phase12_6b_plugin_shadow_hook_candidate"
        ],
        "output_json": args.output_json,
        "output_md": args.output_md,
        "active_routing": False,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
