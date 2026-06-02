#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check whether the local environment can run vLLM runtime dry-runs."""

import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _safe_check(fn: Any) -> dict[str, Any]:
    try:
        return {"ok": True, **fn()}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _check_torch() -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }


def _check_vllm() -> dict[str, Any]:
    import vllm

    return {
        "vllm_module": str(getattr(vllm, "__file__", "")),
        "vllm_version": getattr(vllm, "__version__", None),
    }


def _check_compiled_extension() -> dict[str, Any]:
    import torch

    namespace = getattr(torch.ops, "_C", None)
    has_cpu_memory_env = bool(
        namespace is not None and hasattr(namespace, "init_cpu_memory_env")
    )
    return {
        "torch_ops_C_present": namespace is not None,
        "has_init_cpu_memory_env": has_cpu_memory_env,
    }


def main() -> int:
    summary = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repo_root": str(REPO_ROOT),
        "torch": _safe_check(_check_torch),
        "vllm": _safe_check(_check_vllm),
        "compiled_extension": _safe_check(_check_compiled_extension),
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
