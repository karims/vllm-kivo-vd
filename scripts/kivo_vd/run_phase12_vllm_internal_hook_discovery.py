#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Inspect installed-wheel vLLM for passive internal hook candidates."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and rank installed-wheel vLLM hook candidates without "
            "installing a patch."
        )
    )
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_6c_internal_hook_discovery.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "outputs/kivo_vd/runs/"
            "phase12_6c_internal_hook_discovery.md"
        ),
    )
    parser.add_argument(
        "--include-source-previews",
        action="store_true",
    )
    parser.add_argument("--max-doc-preview-chars", type=int, default=240)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _load_discovery_api() -> tuple[Any, Any]:
    try:
        module = importlib.import_module(
            "kivo_vllm_shadow_plugin.internal_discovery"
        )
    except ImportError:
        plugin_root = (
            Path(__file__).resolve().parents[2]
            / "plugins"
            / "kivo_vllm_shadow_plugin"
        )
        if not plugin_root.exists():
            raise
        sys.path.insert(0, str(plugin_root))
        module = importlib.import_module(
            "kivo_vllm_shadow_plugin.internal_discovery"
        )
    return module.discover_internal_hooks, module.classify_vllm_source_path


def _safe_import_summary(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "version": getattr(module, "__version__", None),
            "file": str(getattr(module, "__file__", "") or ""),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "file": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_environment(
    *,
    import_module: Any = importlib.import_module,
    classify_path: Any = None,
) -> dict[str, Any]:
    if classify_path is None:
        _, classify_path = _load_discovery_api()
    vllm = import_module("vllm")
    vllm_file = str(getattr(vllm, "__file__", "") or "")
    provenance = classify_path(vllm_file)

    try:
        torch = import_module("torch")
        torch_summary = {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_version": getattr(
                getattr(torch, "version", None),
                "cuda",
                None,
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "error": None,
        }
    except Exception as exc:
        torch_summary = {
            "available": False,
            "version": None,
            "cuda_version": None,
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    extensions = {
        name: _safe_import_summary(name)
        for name in (
            "vllm._C",
            "vllm._C_stable_libtorch",
            "vllm.vllm_flash_attn",
        )
    }
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_summary,
        "vllm_version": getattr(vllm, "__version__", None),
        **provenance,
        "compiled_extensions": extensions,
    }


def build_report(
    *,
    include_source_previews: bool = False,
    max_doc_preview_chars: int = 240,
    discover_fn: Any = None,
    environment_fn: Any = collect_environment,
) -> dict[str, Any]:
    if max_doc_preview_chars < 0:
        raise ValueError("max_doc_preview_chars must be non-negative")
    if discover_fn is None:
        discover_fn, _ = _load_discovery_api()
    environment = environment_fn()
    discovery = discover_fn(
        include_source_previews=include_source_previews,
        max_doc_preview_chars=max_doc_preview_chars,
    )
    candidates = discovery["candidates"]
    nonempty = bool(candidates)
    installed_wheel = bool(environment["installed_wheel_path"])
    return {
        "phase": "12.6C",
        "status": (
            "succeeded" if nonempty and installed_wheel else "needs_attention"
        ),
        "environment": environment,
        **discovery,
        "candidate_list_nonempty": nonempty,
        "installed_wheel_vllm": installed_wheel,
        "phase12_6d_candidate_review_ready": bool(
            nonempty
            and installed_wheel
            and discovery["summary"]["callable_candidate_count"] > 0
        ),
        "caveats": [
            "Discovery records signatures and provenance only.",
            "High usefulness does not imply that a method is safe to patch.",
            "Scheduler, attention, KV, block-table, and slot hooks are high risk.",
            "No internal wrapper or monkeypatch is installed.",
            "No memory, latency, quality, or active-routing claim is made.",
        ],
    }


def _format_bool(value: Any) -> str:
    return str(bool(value)).lower()


def render_markdown(report: dict[str, Any]) -> str:
    environment = report["environment"]
    summary = report["summary"]
    lines = [
        "# Kivo-VD Phase 12.6C Internal Hook Discovery",
        "",
        "## Environment",
        "",
        f"- Status: `{report['status']}`",
        f"- Python: `{environment['python_version']}`",
        f"- PyTorch: `{environment['torch']['version']}`",
        f"- PyTorch CUDA: `{environment['torch']['cuda_version']}`",
        f"- vLLM: `{environment['vllm_version']}`",
        f"- vLLM file: `{environment['vllm_file']}`",
        (
            "- Installed-wheel path: "
            f"`{_format_bool(environment['installed_wheel_path'])}`"
        ),
        "",
        "## Candidate Summary",
        "",
        f"- Candidates inspected: `{summary['candidate_count']}`",
        (
            "- Callable candidates: "
            f"`{summary['callable_candidate_count']}`"
        ),
        f"- Missing modules: `{summary['missing_module_count']}`",
        "- Patch installed: `false`",
        "- Runtime behavior changed: `false`",
        "- Active routing: `false`",
        "- Measured runtime reduction: `false`",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | Candidate | Available | Risk | Usefulness | Category |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for candidate in report["candidates"]:
        lines.append(
            "| {rank} | `{name}` | {available} | {risk} | {usefulness} | "
            "{category} |".format(
                rank=candidate["rank"],
                name=candidate["qualified_name"],
                available=_format_bool(candidate["callable"]),
                risk=candidate["risk_level"],
                usefulness=candidate["usefulness_level"],
                category=candidate["category"],
            )
        )

    lines.extend([
        "",
        "## Missing Modules",
        "",
    ])
    if report["missing_modules"]:
        lines.extend([
            "| Module | Import error |",
            "| --- | --- |",
        ])
        for item in report["missing_modules"]:
            lines.append(
                f"| `{item['module_path']}` | `{item['error']}` |"
            )
    else:
        lines.append("No catalog modules were missing.")

    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    if report["recommendations"]:
        for item in report["recommendations"]:
            lines.append(
                f"- `{item['qualified_name']}`: {item['recommendation']}"
            )
    else:
        lines.append(
            "- No low/medium-risk high-usefulness callable was identified."
        )
    lines.extend([
        "",
        "High-risk scheduler, model execution, attention, KV-cache, block-table,",
        "and slot-mapping methods are inventory targets only. They are not safe",
        "to patch in Phase 12.6C.",
        "",
        "## Caveats",
        "",
    ])
    lines.extend(f"- {item}" for item in report["caveats"])
    lines.extend([
        "",
        "## Next Decision",
        "",
        (
            "- Phase 12.6D candidate review ready: "
            f"`{_format_bool(report['phase12_6d_candidate_review_ready'])}`"
        ),
        "",
        "Any next phase must review one copied-metadata observation surface",
        "separately. Discovery alone does not authorize an internal hook.",
    ])
    return "\n".join(lines) + "\n"


def _write(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(
            include_source_previews=args.include_source_previews,
            max_doc_preview_chars=args.max_doc_preview_chars,
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "phase": "12.6C",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "active_routing": False,
            "measured_runtime_reduction": False,
            "runtime_behavior_changed": False,
            "patch_installed": False,
            "discovery_only": True,
            "phase12_6d_candidate_review_ready": False,
        }
        exit_code = 0 if args.continue_on_error else 1

    _write(args.output_json, json.dumps(report, indent=2) + "\n")
    if report["status"] == "failed":
        markdown = (
            "# Kivo-VD Phase 12.6C Internal Hook Discovery\n\n"
            f"- Failed: `{report['error_type']}: {report['error']}`\n"
            "- Patch installed: `false`\n"
            "- Runtime behavior changed: `false`\n"
        )
    else:
        markdown = render_markdown(report)
    _write(args.output_md, markdown)
    print(json.dumps({
        "status": report["status"],
        "candidate_list_nonempty": report.get(
            "candidate_list_nonempty",
            False,
        ),
        "patch_installed": False,
        "runtime_behavior_changed": False,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "phase12_6d_candidate_review_ready": report[
            "phase12_6d_candidate_review_ready"
        ],
        "output_json": args.output_json,
        "output_md": args.output_md,
    }, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
