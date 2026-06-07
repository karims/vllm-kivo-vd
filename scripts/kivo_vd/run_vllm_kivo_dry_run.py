#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a real vLLM generation path with Kivo-VD dry-run enabled."""

import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM inference with optional Kivo-VD dry-run hooks."
    )
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument(
        "--prompt",
        default="Kivo-VD is testing dry-run routing without changing attention.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--enable-kivo-vd", action="store_true")
    parser.add_argument(
        "--export-full-block-ids",
        action="store_true",
        help=(
            "Opt in to complete selected/recent/skipped block-ID arrays in "
            "Kivo routing events."
        ),
    )
    parser.add_argument(
        "--event-output",
        default="outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl",
    )
    parser.add_argument(
        "--compare-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a baseline generation with Kivo-VD disabled before dry-run.",
    )
    parser.add_argument(
        "--force-inproc-engine-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable V1 multiprocessing before importing vLLM for observer access.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.05,
        help=(
            "Fraction of GPU memory vLLM may reserve. The default is "
            "intentionally conservative for tiny dry-run validation."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=128,
        help="Conservative maximum model length for dry-run validation.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=128,
        help="Conservative scheduler token budget for dry-run validation.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Conservative sequence budget for dry-run validation.",
    )
    return parser.parse_args(argv)


@contextmanager
def _patched_kivo_vllm_config(
    *,
    enabled: bool,
    event_output: str | None,
    export_event_limit: int = 10_000,
):
    from vllm.engine.arg_utils import EngineArgs

    original = EngineArgs.create_engine_config

    def patched_create_engine_config(self: Any, *args: Any, **kwargs: Any) -> Any:
        config = original(self, *args, **kwargs)
        config.enable_kivo_vd = enabled
        config.kivo_vd_event_export_path = event_output if enabled else None
        config.kivo_vd_export_event_limit = export_event_limit
        return config

    EngineArgs.create_engine_config = patched_create_engine_config
    try:
        yield
    finally:
        EngineArgs.create_engine_config = original


def _extract_generation_text(outputs: list[Any]) -> str:
    if not outputs or not getattr(outputs[0], "outputs", None):
        return ""
    return str(outputs[0].outputs[0].text)


def _extract_prompt_token_length(outputs: list[Any]) -> int | None:
    if not outputs:
        return None
    token_ids = getattr(outputs[0], "prompt_token_ids", None)
    if token_ids is None:
        return None
    return len(token_ids)


def _get_inproc_kivo_observer(llm: Any) -> Any | None:
    engine_core_client = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    engine_core = getattr(engine_core_client, "engine_core", None)
    scheduler = getattr(engine_core, "scheduler", None)
    return getattr(scheduler, "kivo_vd_observer", None)


def _build_llm_kwargs(
    *,
    model: str,
    seed: int,
    dtype: str,
    device: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
) -> dict[str, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "dtype": dtype,
        "seed": seed,
        "enforce_eager": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
    }
    if device != "auto":
        llm_kwargs["device"] = device
    return llm_kwargs


def _run_generation(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    dtype: str,
    device: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    enable_kivo_vd: bool,
    event_output: str | None,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    llm_kwargs = _build_llm_kwargs(
        model=model,
        seed=seed,
        dtype=dtype,
        device=device,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    with _patched_kivo_vllm_config(
        enabled=enable_kivo_vd,
        event_output=event_output,
    ):
        llm = LLM(**llm_kwargs)
        try:
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            observer = _get_inproc_kivo_observer(llm)
            num_events_exported = None
            observer_note = None
            counters = None
            if observer is not None:
                counters = observer.get_counters()
                num_events_exported = observer.export_events(event_output)
            elif enable_kivo_vd:
                observer_note = (
                    "Kivo observer was not accessible from the in-process "
                    "LLM debug path. If V1 multiprocessing is enabled, export "
                    "needs a future engine-core utility hook."
                )
            return {
                "text": _extract_generation_text(outputs),
                "prompt_token_length": _extract_prompt_token_length(outputs),
                "num_events_exported": num_events_exported,
                "observer_counters": counters,
                "observer_note": observer_note,
            }
        finally:
            del llm
            gc.collect()


def main() -> int:
    try:
        args = _parse_args()
        if args.force_inproc_engine_core:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if args.export_full_block_ids:
            os.environ["KIVO_EXPORT_FULL_BLOCK_IDS"] = "1"

        baseline = None
        if args.compare_baseline:
            baseline = _run_generation(
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                seed=args.seed,
                dtype=args.dtype,
                device=args.device,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
                enable_kivo_vd=False,
                event_output=None,
            )

        event_output = str(Path(args.event_output))
        kivo = _run_generation(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            seed=args.seed,
            dtype=args.dtype,
            device=args.device,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_kivo_vd=args.enable_kivo_vd,
            event_output=event_output if args.enable_kivo_vd else None,
        )

        outputs_match = None
        if baseline is not None:
            outputs_match = baseline["text"] == kivo["text"]

        summary = {
            "model": args.model,
            "prompt_token_length": kivo["prompt_token_length"],
            "kivo_enabled": bool(args.enable_kivo_vd),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "baseline_text": baseline["text"] if baseline is not None else None,
            "kivo_text": kivo["text"],
            "outputs_match": outputs_match,
            "event_output": event_output if args.enable_kivo_vd else None,
            "num_events_exported": kivo["num_events_exported"],
            "full_block_ids_export_requested": bool(
                args.export_full_block_ids
            ),
            "observer_counters": kivo["observer_counters"],
            "observer_note": kivo["observer_note"],
            "dry_run_only": True,
        }
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "dry_run_only": True,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
