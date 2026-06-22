# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.worker.kivo_kv_sketch_runtime import (  # noqa: E402
    KivoKVSketchRuntime,
    KivoKVSketchRuntimeConfig,
    KivoKVSketchStore,
    RandomProjectionKVSketchBackend,
    make_kivo_kv_sketch_backend,
)


def test_runtime_config_from_env_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("KIVO_KV_SKETCH_ENABLE", raising=False)
    monkeypatch.delenv("KIVO_KV_SKETCH_BACKEND", raising=False)
    monkeypatch.delenv("KIVO_KV_SKETCH_DIM", raising=False)
    monkeypatch.delenv("KIVO_KV_SKETCH_SEED", raising=False)
    monkeypatch.delenv("KIVO_KV_SKETCH_MAX_BLOCKS", raising=False)

    config = KivoKVSketchRuntimeConfig.from_env()

    assert config.enabled is False
    assert config.backend == "random_projection"
    assert config.sketch_dim == 16
    assert config.seed == 123
    assert config.max_blocks == 1024
    assert KivoKVSketchRuntime.from_env() is None


def test_runtime_config_from_env_enabled(monkeypatch) -> None:
    monkeypatch.setenv("KIVO_KV_SKETCH_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_SKETCH_BACKEND", "random_projection")
    monkeypatch.setenv("KIVO_KV_SKETCH_DIM", "7")
    monkeypatch.setenv("KIVO_KV_SKETCH_SEED", "99")
    monkeypatch.setenv("KIVO_KV_SKETCH_MAX_BLOCKS", "5")

    runtime = KivoKVSketchRuntime.from_env()

    assert runtime is not None
    assert runtime.config == KivoKVSketchRuntimeConfig(
        enabled=True,
        backend="random_projection",
        sketch_dim=7,
        seed=99,
        max_blocks=5,
    )


def test_random_projection_is_deterministic_for_same_seed() -> None:
    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    a = RandomProjectionKVSketchBackend(sketch_dim=6, seed=11)
    b = RandomProjectionKVSketchBackend(sketch_dim=6, seed=11)

    result_a = a.build_block_sketch(block_id=1, block_tensor=tensor)
    result_b = b.build_block_sketch(block_id=1, block_tensor=tensor)

    assert result_a.success is True
    assert result_b.success is True
    assert torch.allclose(result_a.record.sketch, result_b.record.sketch)


def test_random_projection_changes_with_different_seed() -> None:
    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    a = RandomProjectionKVSketchBackend(sketch_dim=6, seed=11)
    b = RandomProjectionKVSketchBackend(sketch_dim=6, seed=12)

    result_a = a.build_block_sketch(block_id=1, block_tensor=tensor)
    result_b = b.build_block_sketch(block_id=1, block_tensor=tensor)

    assert result_a.success is True
    assert result_b.success is True
    assert not torch.allclose(result_a.record.sketch, result_b.record.sketch)


def test_store_insert_get_and_evict_behavior() -> None:
    backend = RandomProjectionKVSketchBackend(sketch_dim=4, seed=3)
    store = KivoKVSketchStore(max_blocks=2)
    runtime = KivoKVSketchRuntime(
        config=KivoKVSketchRuntimeConfig(
            enabled=True,
            backend="random_projection",
            sketch_dim=4,
            seed=3,
            max_blocks=2,
        ),
        backend=backend,
        store=store,
    )

    tensors = [
        torch.full((2, 4), fill_value=float(i), dtype=torch.float32)
        for i in range(3)
    ]
    for i, tensor in enumerate(tensors):
        result = runtime.build_and_store_block_sketch(
            block_id=i,
            block_tensor=tensor,
            kv_kind="key",
            num_layers=12,
        )
        assert result.success is True

    assert store.get(0) is None
    assert store.get(1) is not None
    assert store.get(2) is not None
    stats = runtime.stats()
    assert stats["entry_count"] == 2
    assert stats["sketch_store_updates"] == 3
    assert stats["sketch_store_evictions"] == 1


def test_runtime_scores_blocks_from_stored_sketch_norms() -> None:
    runtime = KivoKVSketchRuntime(
        config=KivoKVSketchRuntimeConfig(
            enabled=True,
            backend="random_projection",
            sketch_dim=4,
            seed=7,
            max_blocks=8,
        ),
        backend=RandomProjectionKVSketchBackend(sketch_dim=4, seed=7),
        store=KivoKVSketchStore(max_blocks=8),
    )
    low = runtime.build_and_store_block_sketch(
        block_id=1,
        block_tensor=torch.ones(2, 8, dtype=torch.float32),
    )
    high = runtime.build_and_store_block_sketch(
        block_id=2,
        block_tensor=torch.full((2, 8), 5.0, dtype=torch.float32),
    )

    scores = runtime.score_blocks([1, 2, 3])

    assert low.success is True
    assert high.success is True
    assert set(scores) == {1, 2}
    assert scores[2] > scores[1]


def test_runtime_ensure_scores_builds_missing_sketches_from_kv_cache() -> None:
    runtime = KivoKVSketchRuntime(
        config=KivoKVSketchRuntimeConfig(
            enabled=True,
            backend="random_projection",
            sketch_dim=4,
            seed=7,
            max_blocks=8,
        ),
        backend=RandomProjectionKVSketchBackend(sketch_dim=4, seed=7),
        store=KivoKVSketchStore(max_blocks=8),
    )
    kv_cache = torch.arange(
        4 * 2 * 2 * 2 * 2,
        dtype=torch.float32,
    ).reshape(4, 2, 2, 2, 2)

    scores = runtime.ensure_scores_for_blocks([1, 2, 99], kv_cache_tensor=kv_cache)

    assert set(scores) == {1, 2}
    assert runtime.store.get(1) is not None
    assert runtime.store.get(2) is not None
    assert runtime.store.get(99) is None


def test_unsupported_shape_fails_safely() -> None:
    backend = RandomProjectionKVSketchBackend(sketch_dim=4, seed=3)
    too_wide_backend = RandomProjectionKVSketchBackend(sketch_dim=5, seed=3)

    scalar_result = backend.build_block_sketch(
        block_id=1, block_tensor=torch.tensor([], dtype=torch.float32)
    )
    bad_type_result = backend.build_block_sketch(block_id=2, block_tensor="bad")
    too_wide_result = too_wide_backend.build_block_sketch(
        block_id=3,
        block_tensor=torch.ones(2, 2, dtype=torch.float32),
    )

    assert scalar_result.success is False
    assert scalar_result.blocker_reason == "block_tensor is empty"
    assert bad_type_result.success is False
    assert "not a torch.Tensor" in bad_type_result.blocker_reason
    assert too_wide_result.success is False
    assert "sketch_dim=5 exceeds input_dim=4" == too_wide_result.blocker_reason


def test_sketch_is_smaller_than_original_input() -> None:
    backend = RandomProjectionKVSketchBackend(sketch_dim=8, seed=5)
    tensor = torch.randn(4, 16, dtype=torch.float16)

    result = backend.build_block_sketch(block_id=9, block_tensor=tensor)

    assert result.success is True
    assert result.original_bytes > result.sketch_bytes
    assert result.record.sketch.device.type == "cpu"
    assert result.record.sketch.numel() == 8


def test_store_does_not_retain_full_input_tensor() -> None:
    runtime = KivoKVSketchRuntime(
        config=KivoKVSketchRuntimeConfig(
            enabled=True,
            backend="random_projection",
            sketch_dim=4,
            seed=7,
            max_blocks=8,
        ),
        backend=RandomProjectionKVSketchBackend(sketch_dim=4, seed=7),
        store=KivoKVSketchStore(max_blocks=8),
    )

    def _build() -> tuple[weakref.ReferenceType[torch.Tensor], int]:
        tensor = torch.randn(2, 8, dtype=torch.float32)
        ref = weakref.ref(tensor)
        runtime.build_and_store_block_sketch(block_id=4, block_tensor=tensor)
        return ref, tensor.numel()

    ref, numel = _build()
    gc.collect()

    stored = runtime.store.get(4)
    assert ref() is None
    assert stored is not None
    assert stored.sketch.numel() < numel
    assert stored.source_numel == numel


def test_backend_factory_supports_random_projection() -> None:
    backend = make_kivo_kv_sketch_backend(
        KivoKVSketchRuntimeConfig(
            enabled=True,
            backend="random_projection",
            sketch_dim=4,
            seed=1,
            max_blocks=2,
        )
    )

    assert isinstance(backend, RandomProjectionKVSketchBackend)
    assert backend.stats()["sketch_backend"] == "random_projection"
