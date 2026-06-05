# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class CountSketchSpec:
    input_dim: int
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
        raise ValueError(
            f"sketch_dim={sketch_dim} must be <= input_dim={input_dim}"
        )
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
            if take_low:
                candidate = low
                low += 1
            else:
                candidate = high
                high -= 1
            take_low = not take_low
            if candidate not in coords_list:
                coords_list.append(candidate)
        coords = np.array(coords_list, dtype=np.int64)
    else:
        raise ValueError(
            "Unknown coordinate strategy "
            f"{strategy!r}; expected one of "
            "['uniform', 'stride', 'low', 'high', 'alternating']"
        )
    if np.unique(coords).shape[0] != sketch_dim:
        raise ValueError(
            f"coordinate strategy {strategy!r} produced duplicate coordinates"
        )
    return np.sort(coords.astype(np.int64))


def generate_synthetic_keys_and_query(
    num_tokens: int,
    input_dim: int,
    seed: int,
    mode: str = "gaussian",
    num_clusters: int = 16,
    cluster_noise: float = 0.1,
    sequence_noise: float = 0.05,
    needle_strength: float = 3.0,
    num_needle_blocks: int = 2,
    block_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((num_tokens, input_dim), dtype=np.float64)
    query = rng.standard_normal((input_dim,), dtype=np.float64)

    if mode == "gaussian":
        return keys, query

    if mode == "clustered":
        centroids = rng.standard_normal((num_clusters, input_dim), dtype=np.float64)
        # contiguous token spans receive cluster ids in sequence-like chunks
        tokens_per_cluster = max(1, num_tokens // num_clusters)
        cluster_ids = np.repeat(np.arange(num_clusters), tokens_per_cluster)[:num_tokens]
        if cluster_ids.shape[0] < num_tokens:
            tail = rng.integers(0, num_clusters, size=num_tokens - cluster_ids.shape[0])
            cluster_ids = np.concatenate([cluster_ids, tail], axis=0)
        keys = centroids[cluster_ids] + cluster_noise * rng.standard_normal(
            (num_tokens, input_dim), dtype=np.float64
        )
        focus = rng.choice(num_clusters, size=min(2, num_clusters), replace=False)
        query = centroids[focus].mean(axis=0) + cluster_noise * rng.standard_normal(
            input_dim, dtype=np.float64
        )
        return keys, query

    if mode == "smooth_sequence":
        # random walk for smooth local continuity
        steps = rng.standard_normal((num_tokens, input_dim), dtype=np.float64)
        keys = np.cumsum(steps, axis=0)
        keys /= np.sqrt(np.arange(1, num_tokens + 1)[:, None])
        keys += sequence_noise * rng.standard_normal((num_tokens, input_dim), dtype=np.float64)
        center = int(rng.integers(0, num_tokens))
        span = max(2, min(block_size, num_tokens))
        s = max(0, center - span // 2)
        e = min(num_tokens, s + span)
        query = keys[s:e].mean(axis=0) + sequence_noise * rng.standard_normal(
            input_dim, dtype=np.float64
        )
        return keys, query

    if mode == "needle_blocks":
        num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
        query = rng.standard_normal((input_dim,), dtype=np.float64)
        query /= np.linalg.norm(query) + 1e-12
        keys = 0.5 * rng.standard_normal((num_tokens, input_dim), dtype=np.float64)
        needle_count = min(max(1, num_needle_blocks), num_blocks)
        needle_blocks = rng.choice(num_blocks, size=needle_count, replace=False)
        for b in needle_blocks:
            s = b * block_size
            e = min(num_tokens, s + block_size)
            keys[s:e] += needle_strength * query
        return keys, query

    if mode == "mixed":
        # clustered base
        centroids = rng.standard_normal((num_clusters, input_dim), dtype=np.float64)
        cluster_ids = rng.integers(0, num_clusters, size=num_tokens)
        keys = centroids[cluster_ids] + cluster_noise * rng.standard_normal(
            (num_tokens, input_dim), dtype=np.float64
        )
        # smooth trend
        trend = np.cumsum(
            sequence_noise * rng.standard_normal((num_tokens, input_dim), dtype=np.float64),
            axis=0,
        )
        keys += trend / np.sqrt(np.arange(1, num_tokens + 1)[:, None])
        # query near selected centroid
        c = int(rng.integers(0, num_clusters))
        query = centroids[c] + cluster_noise * rng.standard_normal(input_dim, dtype=np.float64)
        query /= np.linalg.norm(query) + 1e-12
        # inject needle blocks
        num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
        needle_count = min(max(1, num_needle_blocks), num_blocks)
        needle_blocks = rng.choice(num_blocks, size=needle_count, replace=False)
        for b in needle_blocks:
            s = b * block_size
            e = min(num_tokens, s + block_size)
            keys[s:e] += needle_strength * query
        # extra isotropic noise
        keys += 0.05 * rng.standard_normal((num_tokens, input_dim), dtype=np.float64)
        return keys, query

    raise ValueError(f"Unknown synthetic mode: {mode}")


def make_random_projection(
    input_dim: int,
    sketch_dim: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((input_dim, sketch_dim), dtype=np.float64)
    proj /= np.sqrt(max(sketch_dim, 1))
    return proj


def make_count_sketch(
    input_dim: int,
    sketch_dim: int,
    seed: int,
) -> CountSketchSpec:
    rng = np.random.default_rng(seed)
    bucket_index = rng.integers(0, sketch_dim, size=input_dim, dtype=np.int64)
    bucket_sign = rng.choice(np.array([-1.0, 1.0]), size=input_dim)
    return CountSketchSpec(
        input_dim=input_dim,
        sketch_dim=sketch_dim,
        bucket_index=bucket_index,
        bucket_sign=bucket_sign.astype(np.float64),
    )


def make_srht(input_dim: int, sketch_dim: int, seed: int) -> SRHTSpec:
    if input_dim <= 0 or sketch_dim <= 0:
        raise ValueError("input_dim and sketch_dim must be positive")
    padded_dim = _next_power_of_two(input_dim)
    if sketch_dim > padded_dim:
        raise ValueError(
            f"sketch_dim={sketch_dim} must be <= padded_dim={padded_dim}"
        )
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=padded_dim).astype(np.float64)
    sampled_indices = rng.choice(padded_dim, size=sketch_dim, replace=False)
    sampled_indices = np.sort(sampled_indices.astype(np.int64))
    return SRHTSpec(
        input_dim=input_dim,
        padded_dim=padded_dim,
        sketch_dim=sketch_dim,
        signs=signs,
        sampled_indices=sampled_indices,
    )


def make_bidiagonal_sign(
    input_dim: int,
    sketch_dim: int,
    seed: int,
    alpha: float = 0.5,
    coordinate_strategy: str = "uniform",
) -> BidiagonalSignSpec:
    if input_dim <= 0 or sketch_dim <= 0:
        raise ValueError("input_dim and sketch_dim must be positive")
    if sketch_dim > input_dim:
        raise ValueError(
            f"sketch_dim={sketch_dim} must be <= input_dim={input_dim}"
        )
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=input_dim).astype(np.float64)
    sampled_indices = select_structured_coordinates(
        input_dim,
        sketch_dim,
        seed,
        strategy=coordinate_strategy,
        rng=rng,
    )
    return BidiagonalSignSpec(
        input_dim=input_dim,
        sketch_dim=sketch_dim,
        signs=signs,
        sampled_indices=sampled_indices,
        alpha=float(alpha),
        coordinate_strategy=coordinate_strategy,
    )


def make_bidiagonal_sign_subsample(
    input_dim: int,
    sketch_dim: int,
    seed: int,
    alpha: float = 0.5,
    coordinate_strategy: str = "uniform",
) -> BidiagonalSignSpec:
    return make_bidiagonal_sign(
        input_dim,
        sketch_dim,
        seed,
        alpha,
        coordinate_strategy,
    )


def make_tridiagonal_sign(
    input_dim: int,
    sketch_dim: int,
    seed: int,
    alpha_left: float = 0.25,
    alpha_right: float = 0.25,
    coordinate_strategy: str = "uniform",
) -> TridiagonalSignSpec:
    if input_dim <= 0 or sketch_dim <= 0:
        raise ValueError("input_dim and sketch_dim must be positive")
    if sketch_dim > input_dim:
        raise ValueError(
            f"sketch_dim={sketch_dim} must be <= input_dim={input_dim}"
        )
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=input_dim).astype(np.float64)
    sampled_indices = select_structured_coordinates(
        input_dim,
        sketch_dim,
        seed,
        strategy=coordinate_strategy,
        rng=rng,
    )
    return TridiagonalSignSpec(
        input_dim=input_dim,
        sketch_dim=sketch_dim,
        signs=signs,
        sampled_indices=sampled_indices,
        alpha_left=float(alpha_left),
        alpha_right=float(alpha_right),
        coordinate_strategy=coordinate_strategy,
    )


def apply_random_projection(x: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(projection, dtype=np.float64)


def apply_count_sketch(x: np.ndarray, sketch_spec: CountSketchSpec) -> np.ndarray:
    x_2d = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x_2d.shape[1] != sketch_spec.input_dim:
        raise ValueError(
            f"Expected input dim {sketch_spec.input_dim}, got {x_2d.shape[1]}"
        )
    out = np.zeros((x_2d.shape[0], sketch_spec.sketch_dim), dtype=np.float64)
    weighted = x_2d * sketch_spec.bucket_sign[None, :]
    for dim in range(sketch_spec.input_dim):
        out[:, sketch_spec.bucket_index[dim]] += weighted[:, dim]
    return out if np.asarray(x).ndim > 1 else out[0]


def apply_srht(x: np.ndarray, sketch_spec: SRHTSpec) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    x_2d = np.atleast_2d(x_arr)
    if x_2d.shape[1] != sketch_spec.input_dim:
        raise ValueError(
            f"Expected input dim {sketch_spec.input_dim}, got {x_2d.shape[1]}"
        )
    if sketch_spec.padded_dim != sketch_spec.input_dim:
        pad_width = sketch_spec.padded_dim - sketch_spec.input_dim
        x_2d = np.pad(x_2d, ((0, 0), (0, pad_width)), mode="constant")
    signed = x_2d * sketch_spec.signs[None, :]
    transformed = _fwht(signed) / np.sqrt(float(sketch_spec.padded_dim))
    sampled = transformed[:, sketch_spec.sampled_indices]
    sampled *= np.sqrt(float(sketch_spec.padded_dim) / float(sketch_spec.sketch_dim))
    return sampled if x_arr.ndim > 1 else sampled[0]


def apply_bidiagonal_sign(
    x: np.ndarray,
    sketch_spec: BidiagonalSignSpec,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    x_2d = np.atleast_2d(x_arr)
    if x_2d.shape[1] != sketch_spec.input_dim:
        raise ValueError(
            f"Expected input dim {sketch_spec.input_dim}, got {x_2d.shape[1]}"
        )
    signed = x_2d * sketch_spec.signs[None, :]
    mixed = signed.copy()
    if sketch_spec.input_dim > 1:
        mixed[:, 1:] += sketch_spec.alpha * signed[:, :-1]
    sampled = mixed[:, sketch_spec.sampled_indices]
    sampled *= np.sqrt(float(sketch_spec.input_dim) / float(sketch_spec.sketch_dim))
    return sampled if x_arr.ndim > 1 else sampled[0]


def apply_bidiagonal_sign_subsample(
    x: np.ndarray,
    sketch_spec: BidiagonalSignSpec,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    x_2d = np.atleast_2d(x_arr)
    if x_2d.shape[1] != sketch_spec.input_dim:
        raise ValueError(
            f"Expected input dim {sketch_spec.input_dim}, got {x_2d.shape[1]}"
        )
    idx = sketch_spec.sampled_indices
    sampled = x_2d[:, idx] * sketch_spec.signs[idx][None, :]
    prev_mask = idx > 0
    if np.any(prev_mask):
        prev_idx = idx[prev_mask] - 1
        sampled[:, prev_mask] += (
            sketch_spec.alpha
            * x_2d[:, prev_idx]
            * sketch_spec.signs[prev_idx][None, :]
        )
    sampled *= np.sqrt(float(sketch_spec.input_dim) / float(sketch_spec.sketch_dim))
    return sampled if x_arr.ndim > 1 else sampled[0]


def apply_tridiagonal_sign(
    x: np.ndarray,
    sketch_spec: TridiagonalSignSpec,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    x_2d = np.atleast_2d(x_arr)
    if x_2d.shape[1] != sketch_spec.input_dim:
        raise ValueError(
            f"Expected input dim {sketch_spec.input_dim}, got {x_2d.shape[1]}"
        )
    idx = sketch_spec.sampled_indices
    sampled = x_2d[:, idx] * sketch_spec.signs[idx][None, :]
    prev_mask = idx > 0
    if np.any(prev_mask):
        prev_idx = idx[prev_mask] - 1
        sampled[:, prev_mask] += (
            sketch_spec.alpha_left
            * x_2d[:, prev_idx]
            * sketch_spec.signs[prev_idx][None, :]
        )
    next_mask = idx < (sketch_spec.input_dim - 1)
    if np.any(next_mask):
        next_idx = idx[next_mask] + 1
        sampled[:, next_mask] += (
            sketch_spec.alpha_right
            * x_2d[:, next_idx]
            * sketch_spec.signs[next_idx][None, :]
        )
    sampled *= np.sqrt(float(sketch_spec.input_dim) / float(sketch_spec.sketch_dim))
    return sampled if x_arr.ndim > 1 else sampled[0]


def compute_exact_scores(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    return k @ q


def compute_sketched_scores(
    query: np.ndarray,
    keys: np.ndarray,
    sketch_type: str,
    sketch_dim: int,
    seed: int,
    structured_alpha: float | None = None,
    structured_coordinate_strategy: str = "uniform",
) -> np.ndarray:
    keys_arr = np.asarray(keys, dtype=np.float64)
    query_arr = np.asarray(query, dtype=np.float64)
    input_dim = keys_arr.shape[1]

    if sketch_type == "random_projection":
        proj = make_random_projection(input_dim, sketch_dim, seed)
        keys_s = apply_random_projection(keys_arr, proj)
        query_s = apply_random_projection(query_arr, proj)
    elif sketch_type == "count_sketch":
        spec = make_count_sketch(input_dim, sketch_dim, seed)
        keys_s = apply_count_sketch(keys_arr, spec)
        query_s = apply_count_sketch(query_arr, spec)
    elif sketch_type == "srht":
        spec = make_srht(input_dim, sketch_dim, seed)
        keys_s = apply_srht(keys_arr, spec)
        query_s = apply_srht(query_arr, spec)
    elif sketch_type == "bidiagonal_sign":
        spec = make_bidiagonal_sign(
            input_dim,
            sketch_dim,
            seed,
            alpha=0.5 if structured_alpha is None else structured_alpha,
            coordinate_strategy=structured_coordinate_strategy,
        )
        keys_s = apply_bidiagonal_sign(keys_arr, spec)
        query_s = apply_bidiagonal_sign(query_arr, spec)
    elif sketch_type == "bidiagonal_sign_subsample":
        spec = make_bidiagonal_sign_subsample(
            input_dim,
            sketch_dim,
            seed,
            alpha=0.5 if structured_alpha is None else structured_alpha,
            coordinate_strategy=structured_coordinate_strategy,
        )
        keys_s = apply_bidiagonal_sign_subsample(keys_arr, spec)
        query_s = apply_bidiagonal_sign_subsample(query_arr, spec)
    elif sketch_type == "tridiagonal_sign":
        alpha = 0.25 if structured_alpha is None else structured_alpha
        spec = make_tridiagonal_sign(
            input_dim,
            sketch_dim,
            seed,
            alpha_left=alpha,
            alpha_right=alpha,
            coordinate_strategy=structured_coordinate_strategy,
        )
        keys_s = apply_tridiagonal_sign(keys_arr, spec)
        query_s = apply_tridiagonal_sign(query_arr, spec)
    else:
        raise ValueError(f"Unknown sketch_type: {sketch_type}")

    return keys_s @ query_s


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def topk_recall(exact_topk: np.ndarray, approx_topk: np.ndarray) -> float:
    if exact_topk.size == 0:
        return 1.0
    exact_set = set(np.asarray(exact_topk).tolist())
    approx_set = set(np.asarray(approx_topk).tolist())
    return len(exact_set & approx_set) / float(len(exact_set))


def reshape_keys_to_blocks(keys: np.ndarray, block_size: int) -> np.ndarray:
    keys_arr = np.asarray(keys, dtype=np.float64)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    num_tokens, dim = keys_arr.shape
    num_blocks = (num_tokens + block_size - 1) // block_size
    padded_tokens = num_blocks * block_size
    if padded_tokens != num_tokens:
        pad = np.zeros((padded_tokens - num_tokens, dim), dtype=keys_arr.dtype)
        keys_arr = np.concatenate([keys_arr, pad], axis=0)
    return keys_arr.reshape(num_blocks, block_size, dim)


def block_scores_from_token_scores(
    scores: np.ndarray,
    block_size: int,
    mode: Literal["max", "mean"] = "max",
) -> np.ndarray:
    score_arr = np.asarray(scores, dtype=np.float64)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    num_tokens = score_arr.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    padded_tokens = num_blocks * block_size
    if padded_tokens != num_tokens:
        pad = np.zeros(padded_tokens - num_tokens, dtype=score_arr.dtype)
        score_arr = np.concatenate([score_arr, pad], axis=0)
    by_block = score_arr.reshape(num_blocks, block_size)
    if mode == "max":
        return by_block.max(axis=1)
    if mode == "mean":
        return by_block.mean(axis=1)
    raise ValueError(f"Unknown mode: {mode}")


def topk_block_recall(
    exact_token_scores: np.ndarray,
    approx_token_scores: np.ndarray,
    block_size: int,
    k: int,
    mode: Literal["max", "mean"] = "max",
) -> float:
    exact_block_scores = block_scores_from_token_scores(
        exact_token_scores, block_size, mode=mode
    )
    approx_block_scores = block_scores_from_token_scores(
        approx_token_scores, block_size, mode=mode
    )
    exact_topk = topk_indices(exact_block_scores, k)
    approx_topk = topk_indices(approx_block_scores, k)
    return topk_recall(exact_topk, approx_topk)


def rank_positions_of_targets(
    ranked_ids: np.ndarray,
    target_ids: np.ndarray,
) -> dict[int, int | None]:
    ranked = np.asarray(ranked_ids).tolist()
    pos = {int(block_id): i for i, block_id in enumerate(ranked)}
    out: dict[int, int | None] = {}
    for target in np.asarray(target_ids).tolist():
        target_i = int(target)
        out[target_i] = pos.get(target_i)
    return out


def recall_at_budget(
    exact_top_ids: np.ndarray,
    approx_ranked_ids: np.ndarray,
    budget: int,
) -> float:
    if budget <= 0:
        return 0.0
    exact = set(np.asarray(exact_top_ids).tolist())
    if not exact:
        return 1.0
    approx_prefix = set(np.asarray(approx_ranked_ids)[:budget].tolist())
    return len(exact & approx_prefix) / float(len(exact))


def recall_at_budgets(
    exact_top_ids: np.ndarray,
    approx_ranked_ids: np.ndarray,
    budgets: list[int],
) -> dict[int, float]:
    return {int(b): recall_at_budget(exact_top_ids, approx_ranked_ids, int(b)) for b in budgets}


def mean_reciprocal_rank(exact_top_ids: np.ndarray, approx_ranked_ids: np.ndarray) -> float:
    exact = np.asarray(exact_top_ids)
    if exact.size == 0:
        return 1.0
    positions = rank_positions_of_targets(approx_ranked_ids, exact_top_ids)
    rr_sum = 0.0
    for target in exact.tolist():
        pos = positions.get(int(target))
        if pos is None:
            continue
        rr_sum += 1.0 / float(pos + 1)
    return rr_sum / float(exact.size)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(b, dtype=np.float64).reshape(-1)
    if arr_a.size != arr_b.size or arr_a.size == 0:
        return 0.0
    a_std = float(np.std(arr_a))
    b_std = float(np.std(arr_b))
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    corr = np.corrcoef(arr_a, arr_b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)
