# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from vllm.v1.core.kivo_vd_sketch import KivoVDSketchType


@dataclass(slots=True)
class CountSketchSpec:
    sketch_dim: int
    bucket_index: np.ndarray
    bucket_sign: np.ndarray


@dataclass(slots=True)
class SRHTSpec:
    input_dim: int
    padded_dim: int
    sketch_dim: int
    signs: np.ndarray
    sampled_indices: np.ndarray


@dataclass(slots=True)
class BidiagonalSignSpec:
    input_dim: int
    sketch_dim: int
    signs: np.ndarray
    sampled_indices: np.ndarray
    alpha: float = 0.5
    coordinate_strategy: str = "uniform"


@dataclass(slots=True)
class TridiagonalSignSpec:
    input_dim: int
    sketch_dim: int
    signs: np.ndarray
    sampled_indices: np.ndarray
    alpha_left: float = 0.25
    alpha_right: float = 0.25
    coordinate_strategy: str = "uniform"


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _fwht(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    n = out.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError("FWHT input dimension must be a power of two")
    h = 1
    while h < n:
        reshaped = out.reshape(*out.shape[:-1], -1, h * 2)
        left = reshaped[..., :, :h].copy()
        right = reshaped[..., :, h : h * 2].copy()
        reshaped[..., :, :h] = left + right
        reshaped[..., :, h : h * 2] = left - right
        h *= 2
    return out


def select_structured_coordinates(
    input_dim: int,
    sketch_dim: int,
    seed: int,
    strategy: str = "uniform",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if input_dim <= 0 or sketch_dim <= 0:
        raise ValueError("input_dim and sketch_dim must be positive")
    if sketch_dim > input_dim:
        raise ValueError("sketch_dim must be <= input_dim")
    if strategy == "uniform":
        generator = rng if rng is not None else np.random.default_rng(seed)
        coords = generator.choice(input_dim, size=sketch_dim, replace=False)
    elif strategy == "stride":
        coords = np.round(
            np.linspace(0, input_dim - 1, sketch_dim)
        ).astype(np.int64)
    elif strategy == "low":
        coords = np.arange(sketch_dim, dtype=np.int64)
    elif strategy == "high":
        coords = np.arange(input_dim - sketch_dim, input_dim, dtype=np.int64)
    elif strategy == "alternating":
        coords_list: list[int] = []
        low = 0
        high = input_dim - 1
        take_low = True
        while len(coords_list) < sketch_dim:
            candidate = low if take_low else high
            low += int(take_low)
            high -= int(not take_low)
            take_low = not take_low
            if candidate not in coords_list:
                coords_list.append(candidate)
        coords = np.array(coords_list, dtype=np.int64)
    else:
        raise ValueError(f"Unknown coordinate strategy: {strategy}")
    if np.unique(coords).shape[0] != sketch_dim:
        raise ValueError(
            f"coordinate strategy {strategy!r} produced duplicate coordinates"
        )
    return np.sort(coords.astype(np.int64))


class KivoVDSketchBackend(ABC):
    """NumPy-only sketch backend interface for runtime dry-run planning."""

    sketch_type: KivoVDSketchType

    @abstractmethod
    def make_params(self, input_dim: int, sketch_dim: int, seed: int):
        raise NotImplementedError

    @abstractmethod
    def sketch_vector(self, vector: np.ndarray, params) -> np.ndarray:
        raise NotImplementedError

    def sketch_matrix(self, matrix: np.ndarray, params) -> np.ndarray:
        if matrix.ndim != 2:
            raise ValueError("matrix must have shape [n, d]")
        return np.stack([self.sketch_vector(row, params) for row in matrix], axis=0)

    @abstractmethod
    def score_query_against_blocks(
        self,
        query_sketch: np.ndarray,
        block_sketches: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def rank_block_ids(
        self,
        block_ids: list[int],
        scores: np.ndarray,
    ) -> list[int]:
        if len(block_ids) != int(scores.shape[0]):
            raise ValueError("block_ids length must match scores length")
        ordered = sorted(zip(block_ids, scores.tolist()), key=lambda x: (-x[1], x[0]))
        return [bid for bid, _ in ordered]


class CountSketchBackend(KivoVDSketchBackend):
    sketch_type = KivoVDSketchType.COUNT_SKETCH

    def make_params(self, input_dim: int, sketch_dim: int, seed: int) -> CountSketchSpec:
        if input_dim <= 0 or sketch_dim <= 0:
            raise ValueError("input_dim and sketch_dim must be positive")
        rng = np.random.default_rng(seed)
        bucket_index = rng.integers(0, sketch_dim, size=input_dim, dtype=np.int64)
        bucket_sign = rng.choice(np.array([-1.0, 1.0]), size=input_dim).astype(np.float64)
        return CountSketchSpec(
            sketch_dim=sketch_dim,
            bucket_index=bucket_index,
            bucket_sign=bucket_sign,
        )

    def sketch_vector(self, vector: np.ndarray, params: CountSketchSpec) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")
        if vector.shape[0] != params.bucket_index.shape[0]:
            raise ValueError("vector length mismatch with sketch params")
        out = np.zeros(params.sketch_dim, dtype=np.float64)
        np.add.at(out, params.bucket_index, vector * params.bucket_sign)
        return out

    def score_query_against_blocks(
        self,
        query_sketch: np.ndarray,
        block_sketches: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(block_sketches, dtype=np.float64) @ np.asarray(
            query_sketch, dtype=np.float64
        )


class RandomProjectionBackend(KivoVDSketchBackend):
    sketch_type = KivoVDSketchType.RANDOM_PROJECTION

    def make_params(self, input_dim: int, sketch_dim: int, seed: int) -> np.ndarray:
        if input_dim <= 0 or sketch_dim <= 0:
            raise ValueError("input_dim and sketch_dim must be positive")
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((input_dim, sketch_dim)).astype(np.float64)
        projection /= np.sqrt(float(sketch_dim))
        return projection

    def sketch_vector(self, vector: np.ndarray, params: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")
        return vector @ np.asarray(params, dtype=np.float64)

    def score_query_against_blocks(
        self,
        query_sketch: np.ndarray,
        block_sketches: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(block_sketches, dtype=np.float64) @ np.asarray(
            query_sketch, dtype=np.float64
        )


class SRHTBackend(KivoVDSketchBackend):
    sketch_type = KivoVDSketchType.SRHT

    def make_params(self, input_dim: int, sketch_dim: int, seed: int) -> SRHTSpec:
        if input_dim <= 0 or sketch_dim <= 0:
            raise ValueError("input_dim and sketch_dim must be positive")
        padded_dim = _next_power_of_two(input_dim)
        if sketch_dim > padded_dim:
            raise ValueError("sketch_dim must be <= padded_dim for SRHT")
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=padded_dim)
        sampled_indices = rng.choice(padded_dim, size=sketch_dim, replace=False)
        return SRHTSpec(
            input_dim=input_dim,
            padded_dim=padded_dim,
            sketch_dim=sketch_dim,
            signs=signs.astype(np.float64),
            sampled_indices=np.sort(sampled_indices.astype(np.int64)),
        )

    def sketch_vector(self, vector: np.ndarray, params: SRHTSpec) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")
        if vector.shape[0] != params.input_dim:
            raise ValueError("vector length mismatch with sketch params")
        if params.padded_dim != params.input_dim:
            vector = np.pad(vector, (0, params.padded_dim - params.input_dim))
        signed = vector * params.signs
        transformed = _fwht(signed) / np.sqrt(float(params.padded_dim))
        sampled = transformed[params.sampled_indices]
        return sampled * np.sqrt(float(params.padded_dim) / float(params.sketch_dim))

    def score_query_against_blocks(
        self,
        query_sketch: np.ndarray,
        block_sketches: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(block_sketches, dtype=np.float64) @ np.asarray(
            query_sketch, dtype=np.float64
        )


class BidiagonalSignBackend(KivoVDSketchBackend):
    sketch_type = KivoVDSketchType.BIDIAGONAL_SIGN

    def __init__(
        self,
        alpha: float = 0.5,
        coordinate_strategy: str = "uniform",
    ) -> None:
        self.alpha = float(alpha)
        self.coordinate_strategy = coordinate_strategy

    def make_params(
        self, input_dim: int, sketch_dim: int, seed: int
    ) -> BidiagonalSignSpec:
        if input_dim <= 0 or sketch_dim <= 0:
            raise ValueError("input_dim and sketch_dim must be positive")
        if sketch_dim > input_dim:
            raise ValueError("sketch_dim must be <= input_dim for bidiagonal_sign")
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=input_dim)
        sampled_indices = select_structured_coordinates(
            input_dim,
            sketch_dim,
            seed,
            strategy=self.coordinate_strategy,
            rng=rng,
        )
        return BidiagonalSignSpec(
            input_dim=input_dim,
            sketch_dim=sketch_dim,
            signs=signs.astype(np.float64),
            sampled_indices=sampled_indices,
            alpha=self.alpha,
            coordinate_strategy=self.coordinate_strategy,
        )

    def sketch_vector(
        self, vector: np.ndarray, params: BidiagonalSignSpec
    ) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")
        if vector.shape[0] != params.input_dim:
            raise ValueError("vector length mismatch with sketch params")
        signed = vector * params.signs
        mixed = signed.copy()
        if params.input_dim > 1:
            mixed[1:] += params.alpha * signed[:-1]
        sampled = mixed[params.sampled_indices]
        return sampled * np.sqrt(float(params.input_dim) / float(params.sketch_dim))

    def score_query_against_blocks(
        self,
        query_sketch: np.ndarray,
        block_sketches: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(block_sketches, dtype=np.float64) @ np.asarray(
            query_sketch, dtype=np.float64
        )


class BidiagonalSignSubsampleBackend(BidiagonalSignBackend):
    sketch_type = KivoVDSketchType.BIDIAGONAL_SIGN_SUBSAMPLE

    def sketch_vector(
        self, vector: np.ndarray, params: BidiagonalSignSpec
    ) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")
        if vector.shape[0] != params.input_dim:
            raise ValueError("vector length mismatch with sketch params")
        idx = params.sampled_indices
        sampled = vector[idx] * params.signs[idx]
        prev_mask = idx > 0
        if np.any(prev_mask):
            prev_idx = idx[prev_mask] - 1
            sampled[prev_mask] += (
                params.alpha * vector[prev_idx] * params.signs[prev_idx]
            )
        return sampled * np.sqrt(float(params.input_dim) / float(params.sketch_dim))


class TridiagonalSignBackend(KivoVDSketchBackend):
    sketch_type = KivoVDSketchType.TRIDIAGONAL_SIGN

    def __init__(
        self,
        alpha_left: float = 0.25,
        alpha_right: float = 0.25,
        coordinate_strategy: str = "uniform",
    ) -> None:
        self.alpha_left = float(alpha_left)
        self.alpha_right = float(alpha_right)
        self.coordinate_strategy = coordinate_strategy

    def make_params(
        self, input_dim: int, sketch_dim: int, seed: int
    ) -> TridiagonalSignSpec:
        if input_dim <= 0 or sketch_dim <= 0:
            raise ValueError("input_dim and sketch_dim must be positive")
        if sketch_dim > input_dim:
            raise ValueError("sketch_dim must be <= input_dim for tridiagonal_sign")
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=input_dim)
        sampled_indices = select_structured_coordinates(
            input_dim,
            sketch_dim,
            seed,
            strategy=self.coordinate_strategy,
            rng=rng,
        )
        return TridiagonalSignSpec(
            input_dim=input_dim,
            sketch_dim=sketch_dim,
            signs=signs.astype(np.float64),
            sampled_indices=sampled_indices,
            alpha_left=self.alpha_left,
            alpha_right=self.alpha_right,
            coordinate_strategy=self.coordinate_strategy,
        )

    def sketch_vector(
        self, vector: np.ndarray, params: TridiagonalSignSpec
    ) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("vector must be 1D")
        if vector.shape[0] != params.input_dim:
            raise ValueError("vector length mismatch with sketch params")
        idx = params.sampled_indices
        sampled = vector[idx] * params.signs[idx]
        prev_mask = idx > 0
        if np.any(prev_mask):
            prev_idx = idx[prev_mask] - 1
            sampled[prev_mask] += (
                params.alpha_left * vector[prev_idx] * params.signs[prev_idx]
            )
        next_mask = idx < (params.input_dim - 1)
        if np.any(next_mask):
            next_idx = idx[next_mask] + 1
            sampled[next_mask] += (
                params.alpha_right * vector[next_idx] * params.signs[next_idx]
            )
        return sampled * np.sqrt(float(params.input_dim) / float(params.sketch_dim))

    def score_query_against_blocks(
        self,
        query_sketch: np.ndarray,
        block_sketches: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(block_sketches, dtype=np.float64) @ np.asarray(
            query_sketch, dtype=np.float64
        )


def make_sketch_backend(sketch_type: KivoVDSketchType | str) -> KivoVDSketchBackend:
    normalized = KivoVDSketchType(sketch_type)
    if normalized == KivoVDSketchType.COUNT_SKETCH:
        return CountSketchBackend()
    if normalized == KivoVDSketchType.RANDOM_PROJECTION:
        return RandomProjectionBackend()
    if normalized == KivoVDSketchType.SRHT:
        return SRHTBackend()
    if normalized == KivoVDSketchType.BIDIAGONAL_SIGN:
        return BidiagonalSignBackend()
    if normalized == KivoVDSketchType.BIDIAGONAL_SIGN_SUBSAMPLE:
        return BidiagonalSignSubsampleBackend()
    if normalized == KivoVDSketchType.TRIDIAGONAL_SIGN:
        return TridiagonalSignBackend()
    raise ValueError(f"Unsupported sketch backend type: {normalized.value}")
