# SPDX-License-Identifier: Apache-2.0

"""Minimal worker-side KV sketch backend/store for future live integration.

This module is intentionally small and fail-closed:
- it operates only on compact sketch summaries
- it never retains full KV tensors
- it is not wired into the live demotion/free path yet
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

_DEFAULT_SKETCH_DIM = 16
_DEFAULT_SKETCH_SEED = 123
_DEFAULT_MAX_BLOCKS = 1024
_PROJECTION_CACHE_LOCK = threading.Lock()
_PROJECTION_CACHE: dict[
    tuple[int, int, str, str, int], torch.Tensor
] = {}
_COUNTSKETCH_CACHE_LOCK = threading.Lock()
_COUNTSKETCH_BUCKET_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}
_COUNTSKETCH_SIGN_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, *, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


@dataclass(slots=True, frozen=True)
class KivoKVSketchRuntimeConfig:
    enabled: bool = False
    backend: str = "random_projection"
    sketch_dim: int = _DEFAULT_SKETCH_DIM
    seed: int = _DEFAULT_SKETCH_SEED
    max_blocks: int = _DEFAULT_MAX_BLOCKS

    @classmethod
    def from_env(cls) -> "KivoKVSketchRuntimeConfig":
        return cls(
            enabled=_parse_bool_env("KIVO_KV_SKETCH_ENABLE", default=False),
            backend=os.getenv("KIVO_KV_SKETCH_BACKEND", "random_projection"),
            sketch_dim=_parse_int_env(
                "KIVO_KV_SKETCH_DIM",
                default=_DEFAULT_SKETCH_DIM,
                minimum=1,
            ),
            seed=_parse_int_env(
                "KIVO_KV_SKETCH_SEED",
                default=_DEFAULT_SKETCH_SEED,
                minimum=0,
            ),
            max_blocks=_parse_int_env(
                "KIVO_KV_SKETCH_MAX_BLOCKS",
                default=_DEFAULT_MAX_BLOCKS,
                minimum=1,
            ),
        )


@dataclass(slots=True)
class KivoKVBlockSketchRecord:
    block_id: int
    backend: str
    sketch_dim: int
    sketch: torch.Tensor
    source_shape: tuple[int, ...]
    source_numel: int
    source_dtype: str
    source_device: str
    kv_kind: str | None = None
    num_layers: int | None = None
    shape_summary: tuple[int, ...] = field(default_factory=tuple)
    created_counter: int = 0
    updated_counter: int = 0

    def as_summary(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "backend": self.backend,
            "sketch_dim": self.sketch_dim,
            "sketch_shape": list(self.sketch.shape),
            "sketch_dtype": str(self.sketch.dtype),
            "sketch_device": str(self.sketch.device),
            "source_shape": list(self.source_shape),
            "source_numel": self.source_numel,
            "source_dtype": self.source_dtype,
            "source_device": self.source_device,
            "kv_kind": self.kv_kind,
            "num_layers": self.num_layers,
            "shape_summary": list(self.shape_summary),
            "created_counter": self.created_counter,
            "updated_counter": self.updated_counter,
        }


@dataclass(slots=True)
class KivoKVSketchBuildResult:
    success: bool
    block_id: int | None
    backend: str
    sketch_dim: int
    blocker_reason: str | None = None
    record: KivoKVBlockSketchRecord | None = None
    original_bytes: int = 0
    sketch_bytes: int = 0


def extract_kv_block_tensor(
    kv_cache: Any,
    block_id: int,
) -> tuple[torch.Tensor | None, str | None]:
    """Extract one physical KV block tensor from a worker KV cache tensor.

    Expected runtime attention KV cache shape follows current source-observer
    assumptions:
    - torch.Tensor
    - ndim == 5
    - dim1 == 2 for K/V

    Returns a block tensor shaped like ``kv_cache[block_id]`` on success.
    """
    if not isinstance(kv_cache, torch.Tensor):
        return None, "kv_cache is not a torch.Tensor"
    if kv_cache.ndim != 5:
        return None, "kv_cache ndim is not 5"
    if kv_cache.shape[1] != 2:
        return None, "kv_cache second dimension is not 2"
    if block_id < 0 or block_id >= int(kv_cache.shape[0]):
        return None, "block_id out of range for kv_cache"
    try:
        return kv_cache[block_id], None
    except Exception as exc:
        return None, f"failed to index kv_cache block: {type(exc).__name__}"


def _projection_tensor(
    input_dim: int,
    sketch_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    key = (input_dim, sketch_dim, str(device), str(dtype), seed)
    with _PROJECTION_CACHE_LOCK:
        cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(
        (input_dim, sketch_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    projection /= float(sketch_dim) ** 0.5
    projection = projection.to(device=device, dtype=dtype)
    with _PROJECTION_CACHE_LOCK:
        _PROJECTION_CACHE[key] = projection
    return projection


def _countsketch_bucket_tensor(
    input_dim: int,
    sketch_dim: int,
    *,
    seed: int,
) -> torch.Tensor:
    key = (input_dim, sketch_dim, seed)
    with _COUNTSKETCH_CACHE_LOCK:
        cached = _COUNTSKETCH_BUCKET_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    buckets = torch.randint(
        low=0,
        high=sketch_dim,
        size=(input_dim,),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    with _COUNTSKETCH_CACHE_LOCK:
        _COUNTSKETCH_BUCKET_CACHE[key] = buckets
    return buckets


def _countsketch_sign_tensor(
    input_dim: int,
    *,
    seed: int,
) -> torch.Tensor:
    key = (input_dim, seed)
    with _COUNTSKETCH_CACHE_LOCK:
        cached = _COUNTSKETCH_SIGN_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(
        low=0,
        high=2,
        size=(input_dim,),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
    with _COUNTSKETCH_CACHE_LOCK:
        _COUNTSKETCH_SIGN_CACHE[key] = signs
    return signs


def countsketch_tensor(
    tensor: torch.Tensor,
    *,
    sketch_dim: int,
    seed: int,
) -> torch.Tensor:
    """Apply true CountSketch to a 1D or 2D tensor.

    For x in R^d, CountSketch y in R^m is:
        y[h(i)] += s(i) * x[i]

    This supports m < d, m == d, and m > d.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.ndim not in (1, 2):
        raise ValueError("tensor must be 1D or 2D")
    if tensor.numel() <= 0:
        raise ValueError("tensor must not be empty")
    if sketch_dim <= 0:
        raise ValueError("sketch_dim must be positive")

    input_dim = int(tensor.shape[-1])
    buckets = _countsketch_bucket_tensor(
        input_dim,
        sketch_dim,
        seed=seed,
    ).to(device=tensor.device)
    signs = _countsketch_sign_tensor(
        input_dim,
        seed=seed + 1,
    ).to(device=tensor.device, dtype=tensor.dtype)

    if tensor.ndim == 1:
        weighted = tensor * signs
        output = torch.zeros(
            sketch_dim,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        output.scatter_add_(0, buckets, weighted)
        return output

    weighted = tensor * signs.unsqueeze(0)
    output = torch.zeros(
        (int(tensor.shape[0]), sketch_dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    output.scatter_add_(
        1,
        buckets.unsqueeze(0).expand(int(tensor.shape[0]), -1),
        weighted,
    )
    return output


class KivoKVSketchBackend(ABC):
    backend_name: str

    def __init__(self, *, sketch_dim: int, seed: int) -> None:
        self.sketch_dim = int(sketch_dim)
        self.seed = int(seed)
        self._stats: dict[str, Any] = {
            "sketch_build_attempted": 0,
            "sketch_build_succeeded": 0,
            "sketch_build_failed": 0,
            "sketch_blocks_built": 0,
            "sketch_bytes_total": 0,
            "sketch_backend": self.backend_name,
        }

    @abstractmethod
    def build_block_sketch(
        self,
        *,
        block_id: int,
        block_tensor: Any,
        kv_kind: str | None = None,
        num_layers: int | None = None,
    ) -> KivoKVSketchBuildResult:
        raise NotImplementedError

    def build_many_block_sketches(
        self,
        items: Iterable[tuple[int, Any]],
        *,
        kv_kind: str | None = None,
        num_layers: int | None = None,
    ) -> list[KivoKVSketchBuildResult]:
        results: list[KivoKVSketchBuildResult] = []
        for block_id, block_tensor in items:
            results.append(
                self.build_block_sketch(
                    block_id=int(block_id),
                    block_tensor=block_tensor,
                    kv_kind=kv_kind,
                    num_layers=num_layers,
                )
            )
        return results

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def _failure(
        self,
        *,
        block_id: int | None,
        reason: str,
    ) -> KivoKVSketchBuildResult:
        self._stats["sketch_build_attempted"] += 1
        self._stats["sketch_build_failed"] += 1
        return KivoKVSketchBuildResult(
            success=False,
            block_id=block_id,
            backend=self.backend_name,
            sketch_dim=self.sketch_dim,
            blocker_reason=reason,
        )

    def _success(
        self,
        *,
        block_id: int,
        record: KivoKVBlockSketchRecord,
        original_bytes: int,
        sketch_bytes: int,
    ) -> KivoKVSketchBuildResult:
        self._stats["sketch_build_attempted"] += 1
        self._stats["sketch_build_succeeded"] += 1
        self._stats["sketch_blocks_built"] += 1
        self._stats["sketch_bytes_total"] += int(sketch_bytes)
        return KivoKVSketchBuildResult(
            success=True,
            block_id=block_id,
            backend=self.backend_name,
            sketch_dim=self.sketch_dim,
            record=record,
            original_bytes=original_bytes,
            sketch_bytes=sketch_bytes,
        )


class RandomProjectionKVSketchBackend(KivoKVSketchBackend):
    backend_name = "random_projection"

    def build_block_sketch(
        self,
        *,
        block_id: int,
        block_tensor: Any,
        kv_kind: str | None = None,
        num_layers: int | None = None,
    ) -> KivoKVSketchBuildResult:
        if not isinstance(block_tensor, torch.Tensor):
            return self._failure(
                block_id=block_id,
                reason="block_tensor is not a torch.Tensor",
            )
        if block_tensor.numel() <= 0:
            return self._failure(
                block_id=block_id,
                reason="block_tensor is empty",
            )

        try:
            flat = block_tensor.detach().reshape(-1).to(torch.float32)
        except Exception as exc:
            return self._failure(
                block_id=block_id,
                reason=f"unable to flatten block tensor: {type(exc).__name__}",
            )

        input_dim = int(flat.numel())
        if self.sketch_dim > input_dim:
            return self._failure(
                block_id=block_id,
                reason=(
                    f"sketch_dim={self.sketch_dim} exceeds input_dim={input_dim}"
                ),
            )

        try:
            projection = _projection_tensor(
                input_dim,
                self.sketch_dim,
                device=flat.device,
                dtype=flat.dtype,
                seed=self.seed,
            )
            sketch = torch.matmul(flat, projection)
            sketch_cpu = sketch.to(device="cpu", dtype=torch.float32).clone()
        except Exception as exc:
            return self._failure(
                block_id=block_id,
                reason=f"projection failed: {type(exc).__name__}",
            )

        record = KivoKVBlockSketchRecord(
            block_id=int(block_id),
            backend=self.backend_name,
            sketch_dim=self.sketch_dim,
            sketch=sketch_cpu,
            source_shape=tuple(int(dim) for dim in block_tensor.shape),
            source_numel=input_dim,
            source_dtype=str(block_tensor.dtype),
            source_device=str(block_tensor.device),
            kv_kind=kv_kind,
            num_layers=num_layers,
            shape_summary=tuple(int(dim) for dim in block_tensor.shape),
            created_counter=1,
            updated_counter=1,
        )
        original_bytes = int(block_tensor.numel() * block_tensor.element_size())
        sketch_bytes = int(sketch_cpu.numel() * sketch_cpu.element_size())
        return self._success(
            block_id=int(block_id),
            record=record,
            original_bytes=original_bytes,
            sketch_bytes=sketch_bytes,
        )


class CountSketchKVSketchBackend(KivoKVSketchBackend):
    backend_name = "countsketch"

    def build_block_sketch(
        self,
        *,
        block_id: int,
        block_tensor: Any,
        kv_kind: str | None = None,
        num_layers: int | None = None,
    ) -> KivoKVSketchBuildResult:
        if not isinstance(block_tensor, torch.Tensor):
            return self._failure(
                block_id=block_id,
                reason="block_tensor is not a torch.Tensor",
            )
        if block_tensor.numel() <= 0:
            return self._failure(
                block_id=block_id,
                reason="block_tensor is empty",
            )

        try:
            detached = block_tensor.detach().to(torch.float32)
        except Exception as exc:
            return self._failure(
                block_id=block_id,
                reason=f"unable to flatten block tensor: {type(exc).__name__}",
            )

        if detached.ndim <= 2:
            sketch_input = detached
            input_dim = int(detached.shape[-1])
        else:
            sketch_input = detached.reshape(-1)
            input_dim = int(sketch_input.shape[-1])

        try:
            sketch = countsketch_tensor(
                sketch_input,
                sketch_dim=self.sketch_dim,
                seed=self.seed,
            )
            sketch_cpu = sketch.to(device="cpu", dtype=torch.float32).clone()
        except Exception as exc:
            return self._failure(
                block_id=block_id,
                reason=f"countsketch failed: {type(exc).__name__}",
            )

        record = KivoKVBlockSketchRecord(
            block_id=int(block_id),
            backend=self.backend_name,
            sketch_dim=self.sketch_dim,
            sketch=sketch_cpu,
            source_shape=tuple(int(dim) for dim in block_tensor.shape),
            source_numel=input_dim,
            source_dtype=str(block_tensor.dtype),
            source_device=str(block_tensor.device),
            kv_kind=kv_kind,
            num_layers=num_layers,
            shape_summary=tuple(int(dim) for dim in block_tensor.shape),
            created_counter=1,
            updated_counter=1,
        )
        original_bytes = int(block_tensor.numel() * block_tensor.element_size())
        sketch_bytes = int(sketch_cpu.numel() * sketch_cpu.element_size())
        return self._success(
            block_id=int(block_id),
            record=record,
            original_bytes=original_bytes,
            sketch_bytes=sketch_bytes,
        )


class KivoKVSketchStore:
    def __init__(self, *, max_blocks: int) -> None:
        self.max_blocks = int(max_blocks)
        self._records: OrderedDict[int, KivoKVBlockSketchRecord] = OrderedDict()
        self._updates = 0
        self._evictions = 0

    def update(self, record: KivoKVBlockSketchRecord) -> None:
        block_id = int(record.block_id)
        existing = self._records.pop(block_id, None)
        if existing is not None:
            record.created_counter = existing.created_counter
            record.updated_counter = existing.updated_counter + 1
        self._records[block_id] = record
        self._updates += 1
        while len(self._records) > self.max_blocks:
            self._records.popitem(last=False)
            self._evictions += 1

    def get(self, block_id: int) -> KivoKVBlockSketchRecord | None:
        record = self._records.get(int(block_id))
        if record is None:
            return None
        return record

    def clear(self) -> None:
        self._records.clear()
        self._updates = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        total_bytes = sum(
            int(record.sketch.numel() * record.sketch.element_size())
            for record in self._records.values()
        )
        return {
            "entry_count": len(self._records),
            "max_blocks": self.max_blocks,
            "sketch_store_updates": self._updates,
            "sketch_store_evictions": self._evictions,
            "stored_sketch_bytes": total_bytes,
            "block_ids_sample": list(self._records.keys())[:16],
        }

    def score(self, block_id: int) -> float | None:
        record = self.get(block_id)
        if record is None:
            return None
        try:
            return float(torch.linalg.vector_norm(record.sketch.float()).item())
        except Exception:
            return None


class KivoKVSketchRuntime:
    """Small runtime-owned wrapper around backend + bounded CPU sketch store."""

    def __init__(
        self,
        *,
        config: KivoKVSketchRuntimeConfig,
        backend: KivoKVSketchBackend,
        store: KivoKVSketchStore,
    ) -> None:
        self.config = config
        self.backend = backend
        self.store = store

    @classmethod
    def from_env(cls) -> "KivoKVSketchRuntime | None":
        config = KivoKVSketchRuntimeConfig.from_env()
        if not config.enabled:
            return None
        backend = make_kivo_kv_sketch_backend(config)
        store = KivoKVSketchStore(max_blocks=config.max_blocks)
        return cls(config=config, backend=backend, store=store)

    def build_and_store_block_sketch(
        self,
        *,
        block_id: int,
        block_tensor: Any,
        kv_kind: str | None = None,
        num_layers: int | None = None,
    ) -> KivoKVSketchBuildResult:
        result = self.backend.build_block_sketch(
            block_id=block_id,
            block_tensor=block_tensor,
            kv_kind=kv_kind,
            num_layers=num_layers,
        )
        if result.success and result.record is not None:
            self.store.update(result.record)
        return result

    def build_and_store_many_block_sketches(
        self,
        items: Iterable[tuple[int, Any]],
        *,
        kv_kind: str | None = None,
        num_layers: int | None = None,
    ) -> list[KivoKVSketchBuildResult]:
        results = self.backend.build_many_block_sketches(
            items,
            kv_kind=kv_kind,
            num_layers=num_layers,
        )
        for result in results:
            if result.success and result.record is not None:
                self.store.update(result.record)
        return results

    def stats(self) -> dict[str, Any]:
        backend_stats = self.backend.stats()
        store_stats = self.store.stats()
        merged = dict(backend_stats)
        merged.update(store_stats)
        merged["sketch_backend"] = self.backend.backend_name
        return merged

    def score_blocks(self, block_ids: Iterable[int]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for block_id in block_ids:
            score = self.store.score(int(block_id))
            if score is not None:
                scores[int(block_id)] = score
        return scores

    def ensure_scores_for_blocks(
        self,
        block_ids: Iterable[int],
        *,
        kv_cache_tensor: Any | None,
        kv_kind: str | None = "kv",
    ) -> dict[int, float]:
        """Return available scores, building missing block sketches if possible."""
        block_id_tuple = tuple(int(block_id) for block_id in block_ids)
        if not self.config.enabled:
            return {}

        for block_id in block_id_tuple:
            if self.store.get(block_id) is not None:
                continue
            block_tensor, _ = extract_kv_block_tensor(kv_cache_tensor, block_id)
            if block_tensor is None:
                continue
            self.build_and_store_block_sketch(
                block_id=block_id,
                block_tensor=block_tensor,
                kv_kind=kv_kind,
            )
        return self.score_blocks(block_id_tuple)


def make_kivo_kv_sketch_backend(
    config: KivoKVSketchRuntimeConfig,
) -> KivoKVSketchBackend:
    backend = str(config.backend).strip().lower()
    if backend == "random_projection":
        return RandomProjectionKVSketchBackend(
            sketch_dim=config.sketch_dim,
            seed=config.seed,
        )
    if backend in {"countsketch", "count_sketch"}:
        return CountSketchKVSketchBackend(
            sketch_dim=config.sketch_dim,
            seed=config.seed,
        )
    raise ValueError(f"Unsupported live KV sketch backend: {config.backend}")
