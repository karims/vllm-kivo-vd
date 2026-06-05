# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch

from vllm.v1.core.kivo_vd_sketch import KivoVDSketchType


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _torch_fwht(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    n = out.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError("FWHT input dimension must be a power of two")
    h = 1
    while h < n:
        reshaped = out.reshape(*out.shape[:-1], -1, h * 2)
        left = reshaped[..., :, :h].clone()
        right = reshaped[..., :, h : h * 2].clone()
        reshaped[..., :, :h] = left + right
        reshaped[..., :, h : h * 2] = left - right
        h *= 2
    return out


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


class TorchSRHTBackend(TorchKivoSketchBackend):
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
        self.padded_dim = _next_power_of_two(input_dim)
        if sketch_dim > self.padded_dim:
            raise ValueError("sketch_dim must be <= padded_dim for SRHT")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(
            0,
            2,
            (self.padded_dim,),
            generator=generator,
            dtype=torch.int8,
            device="cpu",
        )
        self.signs = (signs.to(dtype) * 2 - 1).to(self.device)
        self.sampled_indices = torch.randperm(
            self.padded_dim,
            generator=generator,
            device="cpu",
        )[:sketch_dim].sort().values.to(self.device)
        self.hadamard_scale = float(self.padded_dim) ** -0.5
        self.sample_scale = (float(self.padded_dim) / float(sketch_dim)) ** 0.5

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.padded_dim:
            return x
        pad_shape = (*x.shape[:-1], self.padded_dim - self.input_dim)
        padding = torch.zeros(pad_shape, device=self.device, dtype=self.dtype)
        return torch.cat([x, padding], dim=-1)

    def sketch_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[1] != self.input_dim:
            raise ValueError("keys must have shape [num_tokens, input_dim]")
        keys = keys.to(device=self.device, dtype=self.dtype)
        signed = self._pad(keys) * self.signs
        transformed = _torch_fwht(signed) * self.hadamard_scale
        return transformed[:, self.sampled_indices] * self.sample_scale

    def sketch_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 1 or query.shape[0] != self.input_dim:
            raise ValueError("query must have shape [input_dim]")
        query = query.to(device=self.device, dtype=self.dtype)
        signed = self._pad(query) * self.signs
        transformed = _torch_fwht(signed) * self.hadamard_scale
        return transformed[self.sampled_indices] * self.sample_scale


class TorchBidiagonalSignBackend(TorchKivoSketchBackend):
    def __init__(
        self,
        input_dim: int,
        sketch_dim: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype,
        block_score_mode: str = "max",
        alpha: float = 0.5,
    ) -> None:
        super().__init__(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
        if sketch_dim > input_dim:
            raise ValueError("sketch_dim must be <= input_dim for bidiagonal_sign")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(
            0,
            2,
            (input_dim,),
            generator=generator,
            dtype=torch.int8,
            device="cpu",
        )
        self.signs = (signs.to(dtype) * 2 - 1).to(self.device)
        self.sampled_indices = torch.randperm(
            input_dim,
            generator=generator,
            device="cpu",
        )[:sketch_dim].sort().values.to(self.device)
        self.alpha = float(alpha)
        self.sample_scale = (float(input_dim) / float(sketch_dim)) ** 0.5

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        signed = x * self.signs
        mixed = signed.clone()
        if self.input_dim > 1:
            mixed[..., 1:] = mixed[..., 1:] + self.alpha * signed[..., :-1]
        return mixed[..., self.sampled_indices] * self.sample_scale

    def sketch_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[1] != self.input_dim:
            raise ValueError("keys must have shape [num_tokens, input_dim]")
        keys = keys.to(device=self.device, dtype=self.dtype)
        return self._mix(keys)

    def sketch_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 1 or query.shape[0] != self.input_dim:
            raise ValueError("query must have shape [input_dim]")
        query = query.to(device=self.device, dtype=self.dtype)
        return self._mix(query)


class TorchBidiagonalSignSubsampleBackend(TorchBidiagonalSignBackend):
    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.sampled_indices
        sampled = x[..., idx] * self.signs[idx]
        prev_mask = idx > 0
        if bool(prev_mask.any()):
            prev_idx = idx[prev_mask] - 1
            sampled[..., prev_mask] = sampled[..., prev_mask] + (
                self.alpha * x[..., prev_idx] * self.signs[prev_idx]
            )
        return sampled * self.sample_scale


class TorchTridiagonalSignBackend(TorchKivoSketchBackend):
    def __init__(
        self,
        input_dim: int,
        sketch_dim: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype,
        block_score_mode: str = "max",
        alpha_left: float = 0.25,
        alpha_right: float = 0.25,
    ) -> None:
        super().__init__(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
        if sketch_dim > input_dim:
            raise ValueError("sketch_dim must be <= input_dim for tridiagonal_sign")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(
            0,
            2,
            (input_dim,),
            generator=generator,
            dtype=torch.int8,
            device="cpu",
        )
        self.signs = (signs.to(dtype) * 2 - 1).to(self.device)
        self.sampled_indices = torch.randperm(
            input_dim,
            generator=generator,
            device="cpu",
        )[:sketch_dim].sort().values.to(self.device)
        self.alpha_left = float(alpha_left)
        self.alpha_right = float(alpha_right)
        self.sample_scale = (float(input_dim) / float(sketch_dim)) ** 0.5

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.sampled_indices
        sampled = x[..., idx] * self.signs[idx]
        prev_mask = idx > 0
        if bool(prev_mask.any()):
            prev_idx = idx[prev_mask] - 1
            sampled[..., prev_mask] = sampled[..., prev_mask] + (
                self.alpha_left * x[..., prev_idx] * self.signs[prev_idx]
            )
        next_mask = idx < (self.input_dim - 1)
        if bool(next_mask.any()):
            next_idx = idx[next_mask] + 1
            sampled[..., next_mask] = sampled[..., next_mask] + (
                self.alpha_right * x[..., next_idx] * self.signs[next_idx]
            )
        return sampled * self.sample_scale

    def sketch_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[1] != self.input_dim:
            raise ValueError("keys must have shape [num_tokens, input_dim]")
        keys = keys.to(device=self.device, dtype=self.dtype)
        return self._mix(keys)

    def sketch_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 1 or query.shape[0] != self.input_dim:
            raise ValueError("query must have shape [input_dim]")
        query = query.to(device=self.device, dtype=self.dtype)
        return self._mix(query)


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
    if normalized == KivoVDSketchType.SRHT:
        return TorchSRHTBackend(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
    if normalized == KivoVDSketchType.BIDIAGONAL_SIGN:
        return TorchBidiagonalSignBackend(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
    if normalized == KivoVDSketchType.BIDIAGONAL_SIGN_SUBSAMPLE:
        return TorchBidiagonalSignSubsampleBackend(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
    if normalized == KivoVDSketchType.TRIDIAGONAL_SIGN:
        return TorchTridiagonalSignBackend(
            input_dim, sketch_dim, seed, device, dtype, block_score_mode
        )
    raise ValueError(f"Unsupported torch sketch backend type: {normalized.value}")
