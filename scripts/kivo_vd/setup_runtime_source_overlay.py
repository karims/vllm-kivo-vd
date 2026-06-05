#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Symlink installed vLLM wheel artifacts into this source checkout.

This helper is for Linux/NVIDIA runtime validation environments where the
Python source tree is overlaid via PYTHONPATH, while compiled extensions come
from a prebuilt vLLM wheel.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link installed vLLM wheel artifacts into a source checkout."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to the vllm-kivo-vd source checkout.",
    )
    parser.add_argument(
        "--installed-vllm-path",
        type=Path,
        default=None,
        help=(
            "Optional installed wheel vLLM package path. If omitted, the "
            "script discovers it with PYTHONPATH removed."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to discover the installed vLLM package.",
    )
    parser.add_argument(
        "--replace-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replace existing destination files/symlinks. Disable this to "
            "fail instead of overwriting source-tree artifacts."
        ),
    )
    return parser.parse_args(argv)


def discover_installed_vllm_path(python: str = sys.executable) -> Path:
    """Find installed vLLM while avoiding the repo PYTHONPATH overlay."""

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    code = (
        "import json, pathlib, vllm; "
        "print(json.dumps(str(pathlib.Path(vllm.__file__).resolve().parent)))"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [python, "-c", code],
            check=True,
            capture_output=True,
            cwd=tmpdir,
            env=env,
            text=True,
        )
    return Path(json.loads(proc.stdout))


def _link_file(src: Path, dst: Path, *, replace_existing: bool) -> bool:
    """Create or refresh one symlink.

    Returns True when a new symlink was created or an existing destination was
    changed. Returns False when the destination already pointed at the source.
    """

    src = src.resolve()
    if dst.is_symlink():
        if dst.resolve() == src:
            return False
        dst.unlink()
    elif dst.exists():
        if not replace_existing:
            raise FileExistsError(f"Destination exists and is not a symlink: {dst}")
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)
    return True


def link_runtime_artifacts(
    *,
    installed_vllm_path: Path,
    repo_root: Path,
    replace_existing: bool = True,
) -> dict[str, Any]:
    installed_vllm_path = installed_vllm_path.resolve()
    repo_vllm_path = (repo_root / "vllm").resolve()
    if not installed_vllm_path.exists():
        raise FileNotFoundError(f"Installed vLLM path does not exist: {installed_vllm_path}")
    if not repo_vllm_path.exists():
        raise FileNotFoundError(f"Repo vLLM path does not exist: {repo_vllm_path}")

    top_level_extensions_changed = 0
    top_level_extensions = sorted(installed_vllm_path.glob("*.so"))
    for src in top_level_extensions:
        dst = repo_vllm_path / src.name
        top_level_extensions_changed += int(
            _link_file(src, dst, replace_existing=replace_existing)
        )

    linked_version_file = False
    version_src = installed_vllm_path / "_version.py"
    if version_src.exists():
        linked_version_file = _link_file(
            version_src,
            repo_vllm_path / "_version.py",
            replace_existing=replace_existing,
        ) or (repo_vllm_path / "_version.py").is_symlink()

    flash_attn_extensions_changed = 0
    flash_attn_root = installed_vllm_path / "vllm_flash_attn"
    if flash_attn_root.exists():
        for src in sorted(flash_attn_root.rglob("*.so")):
            rel = src.relative_to(installed_vllm_path)
            dst = repo_vllm_path / rel
            flash_attn_extensions_changed += int(
                _link_file(src, dst, replace_existing=replace_existing)
            )

    return {
        "installed_vllm_path": str(installed_vllm_path),
        "repo_vllm_path": str(repo_vllm_path),
        "num_top_level_extensions_linked": len(top_level_extensions),
        "num_top_level_extensions_changed": top_level_extensions_changed,
        "num_flash_attn_extensions_linked": len(
            list(flash_attn_root.rglob("*.so")) if flash_attn_root.exists() else []
        ),
        "num_flash_attn_extensions_changed": flash_attn_extensions_changed,
        "linked_version_file": linked_version_file,
    }


def main() -> int:
    try:
        args = _parse_args()
        installed_vllm_path = (
            args.installed_vllm_path
            if args.installed_vllm_path is not None
            else discover_installed_vllm_path(args.python)
        )
        summary = link_runtime_artifacts(
            installed_vllm_path=installed_vllm_path,
            repo_root=args.repo_root,
            replace_existing=args.replace_existing,
        )
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
