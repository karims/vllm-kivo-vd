# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch

from vllm.v1.core.kivo_vd_sketch import KivoVDSketchType


class TorchKivoSketchBackend(ABC):
    """Torch-only offline sketch backend for Phase 2.6 benchmarking."""

    def __init__(
        self,
        input_dim: int,
        sketch_dim: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype,
        block_score_mode: str = "max",
    ) -> None:
        if input_dim <= 0 or sketch_dim <= 0:
            raise ValueError("input_dim and sketch_dim must be positive")
        if block_score_mode not in ("max", "mean"):
            raise ValueError("block_score_mode must be 'max' or 'mean'")
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.block_score_mode = block_score_mode

    @abstractmethod
    def sketch_keys(self, keys: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sketch_query(self, query: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def block_sketches_from_key_sketches(
        self, key_sketches: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        if key_sketches.ndim != 2:
            raise ValueError("key_sketches must have shape [num_tokens, sketch_dim]")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        num_tokens = key_sketches.shape[0]
        if num_tokens % block_size != 0:
            trimmed = (num_tokens // block_size) * block_size
            key_sketches = key_sketches[:trimmed]
        if key_sketches.shape[0] == 0:
            return key_sketches.new_empty((0, self.sketch_dim))
        blocks = key_sketches.reshape(-1, block_size, self.sketch_dim)
        if self.block_score_mode == "mean":
            return blocks.mean(dim=1)
        return blocks.max(dim=1).values

    def score_blocks(
        self, query_sketch: torch.Tensor, block_sketches: torch.Tensor
    ) -> torch.Tensor:
        if query_sketch.ndim != 1:
            raise ValueError("query_sketch must have shape [sketch_dim]")
        if block_sketches.ndim != 2:
            raise ValueError("block_sketches must have shape [num_blocks, sketch_dim]")
        return block_sketches @ query_sketch

    def rank_blocks(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.argsort(scores, descending=True)


class TorchCountSketchBackend(TorchKivoSketchBackend):
    def __init__(
        self,
        input_dim: int,
        sketch_dim: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype,
        block_score_mode: str = "max",
    ) -> None:
        super().__init__(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.bucket_index = torch.randint(
            0,
            sketch_dim,
            (input_dim,),
            generator=generator,
            dtype=torch.long,
            device="cpu",
        ).to(self.device)
        signs = torch.randint(
            0,
            2,
            (input_dim,),
            generator=generator,
            dtype=torch.int8,
            device="cpu",
        )
        self.bucket_sign = (signs.to(dtype) * 2 - 1).to(self.device)

    def sketch_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[1] != self.input_dim:
            raise ValueError("keys must have shape [num_tokens, input_dim]")
        keys = keys.to(device=self.device, dtype=self.dtype)
        signed = keys * self.bucket_sign
        out = torch.zeros(
            (keys.shape[0], self.sketch_dim),
            device=self.device,
            dtype=self.dtype,
        )
        bucket_index = self.bucket_index.expand(keys.shape[0], -1)
        return out.scatter_add(dim=1, index=bucket_index, src=signed)

    def sketch_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 1 or query.shape[0] != self.input_dim:
            raise ValueError("query must have shape [input_dim]")
        query = query.to(device=self.device, dtype=self.dtype)
        signed = query * self.bucket_sign
        out = torch.zeros(self.sketch_dim, device=self.device, dtype=self.dtype)
        return out.scatter_add(dim=0, index=self.bucket_index, src=signed)


class TorchRandomProjectionBackend(TorchKivoSketchBackend):
    def __init__(
        self,
        input_dim: int,
        sketch_dim: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype,
        block_score_mode: str = "max",
    ) -> None:
        super().__init__(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projection = torch.randn(
            (input_dim, sketch_dim),
            generator=generator,
            dtype=dtype,
            device="cpu",
        )
        projection /= float(sketch_dim) ** 0.5
        self.projection = projection.to(self.device)

    def sketch_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[1] != self.input_dim:
            raise ValueError("keys must have shape [num_tokens, input_dim]")
        keys = keys.to(device=self.device, dtype=self.dtype)
        return keys @ self.projection

    def sketch_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 1 or query.shape[0] != self.input_dim:
            raise ValueError("query must have shape [input_dim]")
        query = query.to(device=self.device, dtype=self.dtype)
        return query @ self.projection


def make_torch_sketch_backend(
    sketch_type: KivoVDSketchType | str,
    input_dim: int,
    sketch_dim: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
    block_score_mode: str = "max",
) -> TorchKivoSketchBackend:
    normalized = KivoVDSketchType(sketch_type)
    if normalized == KivoVDSketchType.COUNT_SKETCH:
        return TorchCountSketchBackend(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
    if normalized == KivoVDSketchType.RANDOM_PROJECTION:
        return TorchRandomProjectionBackend(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
    raise ValueError(f"Unsupported torch sketch backend type: {normalized.value}")
