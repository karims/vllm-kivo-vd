# SPDX-License-Identifier: Apache-2.0

"""Opt-in public-boundary shadow plugin for Kivo-VD Phase 12.6."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .shadow_events import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_RATIO_POLICY,
    emit_shadow_events_from_generate_call,
    parse_layers,
)

PLUGIN_NAME = "kivo_shadow"
MARKER_ENV = "KIVO_SHADOW_PLUGIN_MARKER"
PATCH_GENERATE_ENV = "KIVO_SHADOW_PLUGIN_PATCH_GENERATE"
EVENTS_ENV = "KIVO_SHADOW_PLUGIN_EVENTS"
LAYERS_ENV = "KIVO_SHADOW_PLUGIN_LAYERS"
BLOCK_SIZE_ENV = "KIVO_SHADOW_PLUGIN_BLOCK_SIZE"
RATIO_POLICY_ENV = "KIVO_SHADOW_PLUGIN_RATIO_POLICY"
PATCH_SENTINEL = "_kivo_shadow_generate_wrapper"
ORIGINAL_GENERATE = "_kivo_shadow_original_generate"


@dataclass(frozen=True)
class KivoShadowPluginState:
    loaded: bool
    plugin_name: str
    timestamp: float
    python_executable: str
    cwd: str
    sys_path_preview: list[str]
    process_id: int
    vllm_version: str | None
    vllm_file: str | None
    caveats: list[str]
    patch_generate_requested: bool = False
    patch_generate_installed: bool = False
    original_generate_qualname: str | None = None
    runtime_warnings: list[str] | None = None
    internal_discovery_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_plugin_state(
    *,
    patch_generate_requested: bool = False,
    patch_generate_installed: bool = False,
    original_generate_qualname: str | None = None,
    runtime_warnings: list[str] | None = None,
) -> KivoShadowPluginState:
    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", None)
        vllm_file = str(getattr(vllm, "__file__", "") or "")
    except Exception:
        vllm_version = None
        vllm_file = None
    return KivoShadowPluginState(
        loaded=True,
        plugin_name=PLUGIN_NAME,
        timestamp=time.time(),
        python_executable=sys.executable,
        cwd=str(Path.cwd()),
        sys_path_preview=list(sys.path[:12]),
        process_id=os.getpid(),
        vllm_version=vllm_version,
        vllm_file=vllm_file,
        caveats=[
            "public LLM.generate wrapper is disabled unless explicitly enabled",
            "no scheduler or attention patch",
            "no KV or block-table access",
            "active routing is disabled",
            "no measured runtime reduction",
        ],
        patch_generate_requested=patch_generate_requested,
        patch_generate_installed=patch_generate_installed,
        original_generate_qualname=original_generate_qualname,
        runtime_warnings=list(runtime_warnings or []),
        internal_discovery_available=True,
    )


def write_load_marker(
    path: str | Path,
    state: KivoShadowPluginState | None = None,
) -> Path:
    marker_path = Path(path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (state or build_plugin_state()).to_dict()
    temporary_path = marker_path.with_name(
        f".{marker_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(marker_path)
    return marker_path


def _record_runtime_warning(message: str) -> None:
    marker_value = os.getenv(MARKER_ENV)
    if not marker_value:
        return
    marker_path = Path(marker_value)
    try:
        payload = (
            json.loads(marker_path.read_text(encoding="utf-8"))
            if marker_path.exists()
            else build_plugin_state().to_dict()
        )
        warnings = payload.setdefault("runtime_warnings", [])
        warnings.append(message)
        temporary_path = marker_path.with_name(
            f".{marker_path.name}.{os.getpid()}.warning.tmp"
        )
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(marker_path)
    except Exception:
        return


def install_generate_patch(
    vllm_module: Any,
    *,
    emitter: Callable[..., int] = emit_shadow_events_from_generate_call,
) -> tuple[bool, str | None]:
    """Install the fail-closed public generate wrapper at most once."""

    llm_class = getattr(vllm_module, "LLM", None)
    if llm_class is None:
        raise AttributeError("vllm.LLM is unavailable")
    current_generate = getattr(llm_class, "generate", None)
    if current_generate is None:
        raise AttributeError("vllm.LLM.generate is unavailable")

    if getattr(current_generate, PATCH_SENTINEL, False):
        original = getattr(llm_class, ORIGINAL_GENERATE, current_generate)
        return True, getattr(original, "__qualname__", None)

    original_generate = current_generate

    def generate_wrapper(
        self: Any,
        prompts: Any,
        sampling_params: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_generate(
            self,
            prompts,
            sampling_params,
            *args,
            **kwargs,
        )
        try:
            events_path = os.getenv(EVENTS_ENV)
            if events_path:
                emitter(
                    prompts=prompts,
                    result=result,
                    events_path=events_path,
                    layers=parse_layers(os.getenv(LAYERS_ENV)),
                    block_size=int(
                        os.getenv(BLOCK_SIZE_ENV, str(DEFAULT_BLOCK_SIZE))
                    ),
                    ratio_policy=os.getenv(
                        RATIO_POLICY_ENV,
                        DEFAULT_RATIO_POLICY,
                    ),
                )
        except Exception as exc:
            _record_runtime_warning(
                f"shadow event emission failed: {type(exc).__name__}: {exc}"
            )
        return result

    setattr(generate_wrapper, PATCH_SENTINEL, True)
    setattr(llm_class, ORIGINAL_GENERATE, original_generate)
    llm_class.generate = generate_wrapper
    return True, getattr(original_generate, "__qualname__", None)


def register() -> None:
    """Record loading and optionally wrap the public generate boundary."""

    patch_requested = _env_enabled(PATCH_GENERATE_ENV)
    patch_installed = False
    original_qualname = None
    warnings: list[str] = []
    if patch_requested:
        try:
            import vllm

            patch_installed, original_qualname = install_generate_patch(vllm)
        except Exception as exc:
            warnings.append(
                f"generate patch installation failed: "
                f"{type(exc).__name__}: {exc}"
            )
    marker_path = os.getenv(MARKER_ENV)
    if marker_path:
        write_load_marker(
            marker_path,
            build_plugin_state(
                patch_generate_requested=patch_requested,
                patch_generate_installed=patch_installed,
                original_generate_qualname=original_qualname,
                runtime_warnings=warnings,
            ),
        )
