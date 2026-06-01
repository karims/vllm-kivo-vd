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
