#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


def _ensure_kivo_package_stubs() -> None:
    for name in ("vllm", "vllm.v1", "vllm.v1.core"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = module


def _load_module(fullname: str, relative_path: str) -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(fullname, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Kivo-VD dry-run events and export JSONL."
    )
    parser.add_argument(
        "--output", default="outputs/kivo_vd/debug_dry_run_events.jsonl"
    )
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--candidate-budget-blocks", type=int, default=16)
    parser.add_argument("--recent-window-blocks", type=int, default=4)
    parser.add_argument("--request-id", default="debug-request")
    parser.add_argument(
        "--sketch-type",
        choices=["count_sketch", "random_projection"],
        default="count_sketch",
    )
    parser.add_argument("--sketch-dim", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_blocks <= 0:
        raise ValueError("--num-blocks must be positive")

    _ensure_kivo_package_stubs()
    sketch_mod = _load_module(
        "vllm.v1.core.kivo_vd_sketch", "vllm/v1/core/kivo_vd_sketch.py"
    )
    selector_mod = _load_module(
        "vllm.v1.core.kivo_vd_candidate_selector",
        "vllm/v1/core/kivo_vd_candidate_selector.py",
    )
    observer_mod = _load_module(
        "vllm.v1.core.kivo_vd_observer", "vllm/v1/core/kivo_vd_observer.py"
    )

    sketch_type = sketch_mod.KivoVDSketchType(args.sketch_type)
    sketch_index = sketch_mod.KivoVDSketchIndex(
        config=sketch_mod.KivoVDSketchConfig(
            enabled=True,
            sketch_dim=args.sketch_dim,
            sketch_type=sketch_type,
        )
    )
    candidate_selector = selector_mod.KivoVDCandidateSelector(
        selector_mod.KivoVDCandidateSelectorConfig(
            recent_window_blocks=args.recent_window_blocks,
            candidate_budget_blocks=args.candidate_budget_blocks,
            min_candidate_blocks=1,
            sketch_type=sketch_type,
            sketch_dim=args.sketch_dim,
        )
    )
    observer = observer_mod.KivoVDObserver(
        enabled=True,
        sketch_index=sketch_index,
        candidate_selector=candidate_selector,
        event_export_path=args.output,
    )

    block_ids = list(range(1, args.num_blocks + 1))
    observer.on_after_allocate_slots(
        request_id=args.request_id,
        block_ids_by_group=(block_ids,),
        num_new_tokens=args.num_blocks,
        source="debug_allocate",
    )
    observer.dry_run_select_candidates(args.request_id, source="debug_dry_run")
    observer.on_free_request(
        request_id=args.request_id,
        block_ids_by_group=(block_ids,),
        source="debug_free",
    )

    num_events_written = observer.export_events()
    recent_events = observer.get_recent_events(limit=10)
    summary = {
        "export_path": args.output,
        "num_events_written": num_events_written,
        "counters": observer.get_counters(),
        "recent_event_types": [event["event_type"] for event in recent_events],
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
