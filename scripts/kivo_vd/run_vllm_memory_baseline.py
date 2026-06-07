#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Measure GPU memory checkpoints during a real vLLM generation."""

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record CUDA memory checkpoints for baseline or Kivo dry-run "
            "vLLM generation."
        )
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--prompt",
        default="Kivo-VD is measuring a dry-run GPU memory baseline.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--enable-kivo-vd", action="store_true")
    parser.add_argument(
        "--export-full-block-ids",
        action="store_true",
        help=(
            "Opt in to complete block-ID arrays in Kivo routing events. "
            "Only applies to Kivo-enabled runs."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.05,
        help="Conservative GPU memory fraction for runtime validation.",
    )
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default="outputs/kivo_vd/vllm_memory_baseline.json",
    )
    parser.add_argument(
        "--event-output",
        default=None,
        help=(
            "Optional Kivo event JSONL path. When omitted, a path beside "
            "--output is derived for Kivo-enabled runs."
        ),
    )
    parser.add_argument(
        "--force-inproc-engine-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable V1 multiprocessing so observer export is accessible.",
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


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "enforce_eager": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.device != "auto":
        kwargs["device"] = args.device
    return kwargs


def _capture_memory_checkpoint(
    name: str,
    cuda: Any,
    *,
    timestamp_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if hasattr(cuda, "synchronize"):
        cuda.synchronize()

    checkpoint: dict[str, Any] = {
        "name": name,
        "timestamp": timestamp_fn(),
        "memory_allocated_bytes": int(cuda.memory_allocated()),
        "memory_reserved_bytes": int(cuda.memory_reserved()),
        "max_memory_allocated_bytes": int(cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(cuda.max_memory_reserved()),
        "free_memory_bytes": None,
        "total_memory_bytes": None,
    }
    try:
        free_memory, total_memory = cuda.mem_get_info()
        checkpoint["free_memory_bytes"] = int(free_memory)
        checkpoint["total_memory_bytes"] = int(total_memory)
    except (AttributeError, RuntimeError):
        pass
    return checkpoint


def _checkpoint_by_name(
    checkpoints: list[dict[str, Any]], name: str
) -> dict[str, Any]:
    return next(checkpoint for checkpoint in checkpoints if checkpoint["name"] == name)


def _compute_peak_deltas(
    checkpoints: list[dict[str, Any]],
) -> dict[str, int]:
    process_start = _checkpoint_by_name(checkpoints, "process_start")
    before_init = _checkpoint_by_name(checkpoints, "before_llm_init")
    after_init = _checkpoint_by_name(checkpoints, "after_llm_init")
    before_generate = _checkpoint_by_name(checkpoints, "before_generate")
    after_generate = _checkpoint_by_name(checkpoints, "after_generate")
    after_cleanup = _checkpoint_by_name(
        checkpoints, "after_request_or_cleanup"
    )

    return {
        "llm_init_allocated_delta_bytes": (
            after_init["memory_allocated_bytes"]
            - before_init["memory_allocated_bytes"]
        ),
        "llm_init_reserved_delta_bytes": (
            after_init["memory_reserved_bytes"]
            - before_init["memory_reserved_bytes"]
        ),
        "generation_allocated_delta_bytes": (
            after_generate["memory_allocated_bytes"]
            - before_generate["memory_allocated_bytes"]
        ),
        "generation_reserved_delta_bytes": (
            after_generate["memory_reserved_bytes"]
            - before_generate["memory_reserved_bytes"]
        ),
        "cleanup_allocated_delta_bytes": (
            after_cleanup["memory_allocated_bytes"]
            - after_generate["memory_allocated_bytes"]
        ),
        "peak_allocated_growth_bytes": (
            after_cleanup["max_memory_allocated_bytes"]
            - process_start["max_memory_allocated_bytes"]
        ),
        "peak_reserved_growth_bytes": (
            after_cleanup["max_memory_reserved_bytes"]
            - process_start["max_memory_reserved_bytes"]
        ),
    }


def _extract_generation_text(outputs: list[Any]) -> str:
    if not outputs or not getattr(outputs[0], "outputs", None):
        return ""
    return str(outputs[0].outputs[0].text)


def _extract_prompt_token_length(outputs: list[Any]) -> int | None:
    if not outputs:
        return None
    token_ids = getattr(outputs[0], "prompt_token_ids", None)
    return len(token_ids) if token_ids is not None else None


def _get_inproc_kivo_observer(llm: Any) -> Any | None:
    engine_core_client = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    engine_core = getattr(engine_core_client, "engine_core", None)
    scheduler = getattr(engine_core, "scheduler", None)
    return getattr(scheduler, "kivo_vd_observer", None)


def _derive_event_output(output: str, explicit_path: str | None) -> str:
    if explicit_path is not None:
        return explicit_path
    output_path = Path(output)
    return str(output_path.with_suffix(".events.jsonl"))


def _build_result(
    *,
    config: dict[str, Any],
    runtime: dict[str, Any],
    output_text: str,
    prompt_token_length: int | None,
    checkpoints: list[dict[str, Any]],
    kivo_enabled: bool,
    observer_counters: dict[str, int] | None,
    event_output: str | None,
    num_events_exported: int | None,
    observer_note: str | None,
) -> dict[str, Any]:
    return {
        "config": config,
        "runtime": runtime,
        "output_text": output_text,
        "prompt_token_length": prompt_token_length,
        "memory_checkpoints": checkpoints,
        "peak_deltas": _compute_peak_deltas(checkpoints),
        "kivo_enabled": kivo_enabled,
        "observer_counters": observer_counters,
        "event_output": event_output,
        "num_events_exported": num_events_exported,
        "observer_note": observer_note,
        "dry_run_only": bool(kivo_enabled),
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    try:
        args = _parse_args()
        if args.force_inproc_engine_core:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if args.export_full_block_ids:
            os.environ["KIVO_EXPORT_FULL_BLOCK_IDS"] = "1"

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Phase 7.0 requires a Linux/NVIDIA "
                "environment running a real vLLM generation."
            )

        torch.cuda.reset_peak_memory_stats()
        checkpoints = [_capture_memory_checkpoint("process_start", torch.cuda)]

        from vllm import LLM, SamplingParams
        from vllm import __version__ as vllm_version

        checkpoints.append(
            _capture_memory_checkpoint("before_llm_init", torch.cuda)
        )

        event_output = (
            _derive_event_output(args.output, args.event_output)
            if args.enable_kivo_vd
            else None
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        llm = None
        outputs: list[Any] = []
        observer_counters = None
        num_events_exported = None
        observer_note = None

        with _patched_kivo_vllm_config(
            enabled=args.enable_kivo_vd,
            event_output=event_output,
        ):
            llm = LLM(**_build_llm_kwargs(args))
            checkpoints.append(
                _capture_memory_checkpoint("after_llm_init", torch.cuda)
            )
            checkpoints.append(
                _capture_memory_checkpoint("before_generate", torch.cuda)
            )
            outputs = llm.generate(
                [args.prompt],
                sampling_params,
                use_tqdm=False,
            )
            checkpoints.append(
                _capture_memory_checkpoint("after_generate", torch.cuda)
            )

            if args.enable_kivo_vd:
                observer = _get_inproc_kivo_observer(llm)
                if observer is not None:
                    observer_counters = observer.get_counters()
                    num_events_exported = observer.export_events(event_output)
                else:
                    observer_note = (
                        "Kivo observer was not accessible from the in-process "
                        "engine-core debug path."
                    )

        output_text = _extract_generation_text(outputs)
        prompt_token_length = _extract_prompt_token_length(outputs)
        del outputs
        del llm
        gc.collect()
        checkpoints.append(
            _capture_memory_checkpoint("after_request_or_cleanup", torch.cuda)
        )

        config = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "export_full_block_ids": bool(args.export_full_block_ids),
        }
        runtime = {
            "vllm_version": vllm_version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
        }
        result = _build_result(
            config=config,
            runtime=runtime,
            output_text=output_text,
            prompt_token_length=prompt_token_length,
            checkpoints=checkpoints,
            kivo_enabled=args.enable_kivo_vd,
            observer_counters=observer_counters,
            event_output=event_output,
            num_events_exported=num_events_exported,
            observer_note=observer_note,
        )
        _write_json(args.output, result)

        summary = {
            "output": args.output,
            "model": args.model,
            "kivo_enabled": args.enable_kivo_vd,
            "dry_run_only": bool(args.enable_kivo_vd),
            "prompt_token_length": prompt_token_length,
            "num_memory_checkpoints": len(checkpoints),
            "peak_deltas": result["peak_deltas"],
            "event_output": event_output,
            "num_events_exported": num_events_exported,
            "full_block_ids_export_requested": bool(
                args.export_full_block_ids
            ),
        }
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "memory_baseline_only": True,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
