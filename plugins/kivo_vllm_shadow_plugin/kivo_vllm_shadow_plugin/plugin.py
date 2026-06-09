# SPDX-License-Identifier: Apache-2.0

"""Marker-only vLLM plugin for Kivo-VD Phase 12.6A."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PLUGIN_NAME = "kivo_shadow"
MARKER_ENV = "KIVO_SHADOW_PLUGIN_MARKER"


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_plugin_state() -> KivoShadowPluginState:
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
            "plugin load marker only",
            "no scheduler or attention monkeypatch",
            "no KV or block-table access",
            "active routing is disabled",
            "no measured runtime reduction",
        ],
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


def register() -> None:
    """Record plugin loading without changing vLLM runtime behavior."""

    marker_path = os.getenv(MARKER_ENV)
    if marker_path:
        write_load_marker(marker_path)
