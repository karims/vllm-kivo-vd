# SPDX-License-Identifier: Apache-2.0

"""Read-only discovery of possible installed-wheel vLLM hook surfaces."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

ImportModule = Callable[[str], Any]


@dataclass(frozen=True)
class CandidateSpec:
    module_path: str
    class_name: str | None
    method_name: str
    category: str
    risk_level: str
    usefulness_level: str
    reason: str


CANDIDATE_SPECS = (
    CandidateSpec(
        "vllm.entrypoints.llm",
        "LLM",
        "generate",
        "generate_boundary",
        "low",
        "medium",
        "Public boundary already validated by Phase 12.6B.",
    ),
    CandidateSpec(
        "vllm.v1.engine.llm_engine",
        "LLMEngine",
        "step",
        "engine_step",
        "medium",
        "high",
        "Engine output boundary may expose request outputs, but drives execution.",
    ),
    CandidateSpec(
        "vllm.v1.engine.core",
        "EngineCore",
        "step",
        "engine_step",
        "high",
        "high",
        "Core step drives scheduling and model execution.",
    ),
    CandidateSpec(
        "vllm.v1.core.scheduler",
        "Scheduler",
        "schedule",
        "scheduler_step",
        "high",
        "high",
        "Legacy scheduler path requested for version discovery.",
    ),
    CandidateSpec(
        "vllm.v1.core.sched.scheduler",
        "Scheduler",
        "schedule",
        "scheduler_step",
        "high",
        "high",
        "Scheduling decisions and block allocation occur on this path.",
    ),
    CandidateSpec(
        "vllm.v1.core.sched.scheduler",
        "Scheduler",
        "update_from_output",
        "scheduler_output",
        "high",
        "high",
        "Consumes model-runner output and mutates request state.",
    ),
    CandidateSpec(
        "vllm.v1.worker.gpu_model_runner",
        "GPUModelRunner",
        "execute_model",
        "model_execution",
        "high",
        "high",
        "Model execution boundary has useful batch metadata but is latency critical.",
    ),
    CandidateSpec(
        "vllm.v1.worker.gpu_model_runner",
        "GPUModelRunner",
        "_build_attention_metadata",
        "attention_metadata",
        "high",
        "high",
        "Private attention metadata construction is version-sensitive.",
    ),
    CandidateSpec(
        "vllm.v1.worker.gpu_model_runner",
        "GPUModelRunner",
        "_get_slot_mappings",
        "slot_mapping",
        "high",
        "high",
        "Private slot mapping directly affects KV placement.",
    ),
    CandidateSpec(
        "vllm.v1.core.kv_cache_manager",
        "KVCacheManager",
        "allocate_slots",
        "kv_cache_allocation",
        "high",
        "high",
        "Allocation path mutates KV block ownership.",
    ),
    CandidateSpec(
        "vllm.v1.core.kv_cache_manager",
        "KVCacheManager",
        "get_block_ids",
        "kv_cache_metadata",
        "medium",
        "high",
        "Read-oriented block ID accessor may support passive copied metadata.",
    ),
    CandidateSpec(
        "vllm.v1.core.kv_cache_manager",
        "KVCacheManager",
        "free",
        "kv_cache_free",
        "high",
        "medium",
        "Free path mutates KV ownership and must not be wrapped casually.",
    ),
    CandidateSpec(
        "vllm.v1.core.kv_cache_utils",
        "FreeKVCacheBlockQueue",
        "get_all_free_blocks",
        "kv_cache_metadata",
        "high",
        "medium",
        "Exposes live free-block objects and may have concurrency hazards.",
    ),
    CandidateSpec(
        "vllm.v1.worker.block_table",
        "BlockTable",
        "append_row",
        "block_table_construction",
        "high",
        "high",
        "Block-table mutation controls logical-to-physical mapping.",
    ),
    CandidateSpec(
        "vllm.v1.worker.block_table",
        "BlockTable",
        "compute_slot_mapping",
        "slot_mapping",
        "high",
        "high",
        "Slot mapping construction directly controls KV write locations.",
    ),
    CandidateSpec(
        "vllm.v1.worker.block_table",
        "BlockTable",
        "get_cpu_tensor",
        "block_table_metadata",
        "medium",
        "high",
        "Read-oriented table access may synchronize or expose mutable storage.",
    ),
    CandidateSpec(
        "vllm.v1.attention.backends.flash_attn",
        "FlashAttentionMetadataBuilder",
        "build",
        "attention_backend",
        "high",
        "high",
        "Backend metadata build is execution-critical and version-sensitive.",
    ),
    CandidateSpec(
        "vllm.v1.attention.backends.flash_attn",
        "FlashAttentionImpl",
        "forward",
        "attention_backend",
        "high",
        "high",
        "Attention forward is kernel-adjacent and out of scope for passive hooks.",
    ),
    CandidateSpec(
        "vllm.v1.core.kv_cache_metrics",
        "KVCacheMetrics",
        "on_block_allocated",
        "metrics_callback",
        "medium",
        "high",
        "Metrics callback shape may be a future passive observation candidate.",
    ),
)

_LEVEL_SCORE = {"low": 1, "medium": 2, "high": 3}


def classify_vllm_source_path(path: str | None) -> dict[str, Any]:
    normalized = str(Path(path).resolve()) if path else ""
    installed = "site-packages" in normalized or "dist-packages" in normalized
    return {
        "vllm_file": path,
        "normalized_vllm_file": normalized,
        "installed_wheel_path": installed,
        "repo_local_source_detected": bool(path) and not installed,
    }


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _safe_source_file(value: Any) -> str | None:
    try:
        path = inspect.getsourcefile(value) or inspect.getfile(value)
        return str(path) if path else None
    except (OSError, TypeError):
        return None


def _safe_doc_preview(value: Any, limit: int) -> str | None:
    try:
        doc = inspect.getdoc(value)
    except Exception:
        return None
    if not doc:
        return None
    compact = " ".join(doc.split())
    return compact[:limit]


def _safe_source_preview(value: Any, limit: int = 800) -> str | None:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        return None
    return source[:limit]


def _rank_key(candidate: dict[str, Any]) -> tuple[int, int, str]:
    usefulness = _LEVEL_SCORE[candidate["usefulness_level"]]
    risk = _LEVEL_SCORE[candidate["risk_level"]]
    return (-usefulness, risk, candidate["qualified_name"])


def discover_internal_hooks(
    *,
    import_module: ImportModule = importlib.import_module,
    specs: tuple[CandidateSpec, ...] = CANDIDATE_SPECS,
    include_source_previews: bool = False,
    max_doc_preview_chars: int = 240,
) -> dict[str, Any]:
    """Inspect a fixed hook catalog without modifying imported objects."""

    module_cache: dict[str, Any] = {}
    module_errors: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []

    for spec in specs:
        module = module_cache.get(spec.module_path)
        if module is None and spec.module_path not in module_errors:
            try:
                module = import_module(spec.module_path)
                module_cache[spec.module_path] = module
            except Exception as exc:
                module_errors[spec.module_path] = (
                    f"{type(exc).__name__}: {exc}"
                )

        owner = None
        value = None
        if module is not None:
            owner = (
                getattr(module, spec.class_name, None)
                if spec.class_name
                else module
            )
            value = getattr(owner, spec.method_name, None) if owner else None

        importable = module is not None and owner is not None
        callable_value = callable(value)
        qualified_name = ".".join(
            item
            for item in (
                spec.module_path,
                spec.class_name,
                spec.method_name,
            )
            if item
        )
        candidate = {
            **asdict(spec),
            "qualified_name": qualified_name,
            "module_importable": module is not None,
            "importable": importable,
            "callable": callable_value,
            "signature": _safe_signature(value) if callable_value else None,
            "source_file": _safe_source_file(value) if callable_value else None,
            "docstring_preview": (
                _safe_doc_preview(value, max_doc_preview_chars)
                if callable_value
                else None
            ),
            "source_preview": (
                _safe_source_preview(value)
                if callable_value and include_source_previews
                else None
            ),
            "safe_to_patch_in_phase12_6c": False,
            "discovery_only": True,
            "signals": {
                name: spec.category == name
                for name in (
                    "generate_boundary",
                    "scheduler_step",
                    "engine_step",
                    "model_execution",
                    "kv_cache_allocation",
                    "kv_cache_metadata",
                    "kv_cache_free",
                    "block_table_construction",
                    "block_table_metadata",
                    "slot_mapping",
                    "attention_metadata",
                    "attention_backend",
                    "metrics_callback",
                )
            },
        }
        candidates.append(candidate)

    candidates.sort(key=_rank_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    available = [item for item in candidates if item["callable"]]
    missing_modules = [
        {"module_path": path, "error": error}
        for path, error in sorted(module_errors.items())
    ]
    counts = {
        "candidate_count": len(candidates),
        "callable_candidate_count": len(available),
        "missing_module_count": len(missing_modules),
        "risk_counts": {
            level: sum(
                item["risk_level"] == level for item in candidates
            )
            for level in ("low", "medium", "high")
        },
        "usefulness_counts": {
            level: sum(
                item["usefulness_level"] == level for item in candidates
            )
            for level in ("low", "medium", "high")
        },
    }
    recommendations = [
        {
            "qualified_name": item["qualified_name"],
            "risk_level": item["risk_level"],
            "usefulness_level": item["usefulness_level"],
            "recommendation": (
                "Inspect call semantics and copied metadata only; do not patch "
                "in Phase 12.6C."
            ),
        }
        for item in available
        if item["usefulness_level"] == "high"
        and item["risk_level"] in {"low", "medium"}
    ]
    return {
        "candidates": candidates,
        "missing_modules": missing_modules,
        "summary": counts,
        "recommendations": recommendations,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "runtime_behavior_changed": False,
        "patch_installed": False,
        "discovery_only": True,
    }
