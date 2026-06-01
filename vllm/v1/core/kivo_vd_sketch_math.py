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
