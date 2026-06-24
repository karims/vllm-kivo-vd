# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / "scripts" / "kivo_vd" / "run_sk3_2_attention_mass_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_sk3_2_attention_mass_eval",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _make_eval_point(
    *,
    layer: int,
    head: int,
    query_position: int,
    query: list[float],
    keys: list[list[float]],
):
    return module.EvalPoint(
        layer=layer,
        head=head,
        query_position=query_position,
        query=torch.tensor(query, dtype=torch.float32),
        keys=torch.tensor(keys, dtype=torch.float32),
    )


def test_true_attention_mass_per_block_sums_to_one() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    keys = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    attn = module.compute_true_attention_weights(query, keys)
    mass = module.aggregate_block_values(attn, block_size=2)

    assert pytest.approx(float(attn.sum().item()), rel=1e-6) == 1.0
    assert pytest.approx(float(mass.sum().item()), rel=1e-6) == 1.0


def test_block_aggregation_handles_uneven_final_block() -> None:
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

    mass = module.aggregate_block_values(values, block_size=2)

    assert torch.allclose(
        mass,
        torch.tensor([0.3, 0.7, 0.5], dtype=torch.float32),
    )


def test_mean_key_score_returns_one_score_per_block() -> None:
    query = torch.tensor([1.0, 2.0], dtype=torch.float32)
    keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    scores = module.mean_key_block_scores(query, keys, block_size=2)

    assert tuple(scores.shape) == (3,)


def test_long_prompt_builders_are_substantially_longer() -> None:
    short_fact = module.build_prompt("factual_recall")
    long_fact = module.build_prompt("factual_recall_long")
    short_code = module.build_prompt("code_context")
    long_code = module.build_prompt("code_context_long")

    assert module.estimate_prompt_token_count(long_fact) > 1000
    assert module.estimate_prompt_token_count(long_code) > 1000
    assert module.estimate_prompt_token_count(long_fact) > (
        module.estimate_prompt_token_count(short_fact) * 10
    )
    assert module.estimate_prompt_token_count(long_code) > (
        module.estimate_prompt_token_count(short_code) * 10
    )


def test_layer_head_selection_parsers_handle_special_values() -> None:
    assert module.resolve_layer_selection("last", 8) == [7]
    assert module.resolve_layer_selection("last4", 8) == [4, 5, 6, 7]
    assert module.resolve_layer_selection("all", 3) == [0, 1, 2]
    assert module.resolve_layer_selection("1,3,5", 8) == [1, 3, 5]
    assert module.resolve_head_selection("all", 4) == [0, 1, 2, 3]
    assert module.resolve_head_selection("0,2", 4) == [0, 2]


def test_query_position_parser_handles_last_last4_and_lists() -> None:
    assert module.resolve_query_position_selection("last", 10) == [9]
    assert module.resolve_query_position_selection("last4", 10) == [6, 7, 8, 9]
    assert module.resolve_query_position_selection("3,5,-1", 10) == [3, 5, 9]


def test_limit_heads_applies_only_requested_cap() -> None:
    assert module.limit_heads([0, 1, 2, 3], 2) == [0, 1]
    assert module.limit_heads([0, 1, 2, 3], 0) == [0, 1, 2, 3]


def test_countsketch_scoring_is_deterministic_with_seed() -> None:
    query = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    keys = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    a = module.sketch_block_scores(
        query,
        keys,
        block_size=2,
        backend="countsketch",
        sketch_dim=3,
        seed=123,
    )
    b = module.sketch_block_scores(
        query,
        keys,
        block_size=2,
        backend="countsketch",
        sketch_dim=3,
        seed=123,
    )

    assert torch.allclose(a, b)


def test_aggregation_over_multiple_eval_points_preserves_mass_sum() -> None:
    eval_points = [
        _make_eval_point(
            layer=0,
            head=0,
            query_position=5,
            query=[1.0, 0.0],
            keys=[[4.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        ),
        _make_eval_point(
            layer=1,
            head=1,
            query_position=6,
            query=[1.0, 0.0],
            keys=[[5.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 3.0]],
        ),
    ]

    result = module.evaluate_backend_scores(
        backend="mean_key",
        eval_points=eval_points,
        block_size=2,
        sketch_dim=2,
        seed=1,
        topks=[1, 2],
    )

    assert pytest.approx(sum(result["true_mass_by_block"]), rel=1e-6) == 1.0


def test_selected_query_attention_mass_matches_weight_then_aggregate() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    keys = torch.tensor(
        [[4.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        dtype=torch.float32,
    )

    direct = module.compute_selected_query_attention_mass(query, keys, block_size=2)
    manual = module.aggregate_block_values(
        module.compute_true_attention_weights(query, keys),
        2,
    )

    assert torch.allclose(direct, manual)


def test_topk_metrics_include_recovery_fraction_of_oracle() -> None:
    eval_points = [
        _make_eval_point(
            layer=0,
            head=0,
            query_position=5,
            query=[1.0, 0.0],
            keys=[[5.0, 0.0], [4.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
    ]

    result = module.evaluate_backend_scores(
        backend="mean_key",
        eval_points=eval_points,
        block_size=2,
        sketch_dim=2,
        seed=1,
        topks=[1, 2],
    )

    assert "recovery_fraction_of_oracle" in result["topk_metrics"]["1"]


def test_max_token_score_exact_ranks_matching_block_highest() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    keys = torch.tensor(
        [
            [0.1, 1.0],
            [0.2, 1.0],
            [9.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=torch.float32,
    )

    scores = module.max_token_score_exact(query, keys, block_size=2)

    assert module.topk_indices_desc(scores, 1) == [1]


def test_old_single_layer_head_mode_still_works() -> None:
    args = module.parse_args(
        [
            "--prompt-kind",
            "synthetic_long",
            "--layer",
            "last",
            "--head",
            "0",
            "--query-position",
            "last",
            "--backend",
            "mean_key",
        ]
    )

    layer_spec = args.layers if args.layers is not None else args.layer
    head_spec = args.heads if args.heads is not None else str(args.head)
    query_spec = (
        args.query_positions if args.query_positions is not None else args.query_position
    )

    assert layer_spec == "last"
    assert head_spec == "0"
    assert query_spec == "last"


def test_prepare_input_ids_applies_max_eval_tokens() -> None:
    class FakeModel:
        config = SimpleNamespace(n_positions=4096)

    class FakeTokenizer:
        model_max_length = 4096

        def __call__(self, prompt, return_tensors="pt"):
            del prompt, return_tensors
            return {"input_ids": torch.arange(20, dtype=torch.int64).reshape(1, 20)}

    input_ids = module._prepare_input_ids(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        prompt="unused",
        device="cpu",
        max_model_len=32,
        max_eval_tokens=12,
    )

    assert tuple(input_ids.shape) == (1, 12)


def test_timing_fields_exist_in_output_payload() -> None:
    payload = module.build_output_payload(
        model="synthetic",
        prompt_kind="synthetic_long",
        token_count=12,
        block_size=4,
        num_blocks=3,
        selected_layers=[1],
        selected_heads=[0],
        selected_query_positions=[11],
        backend_results={},
    )
    payload["load_seconds"] = 0.1
    payload["forward_seconds"] = 0.2
    payload["scoring_seconds"] = 0.3
    payload["total_seconds"] = 0.6

    for field in (
        "load_seconds",
        "forward_seconds",
        "scoring_seconds",
        "total_seconds",
    ):
        assert field in payload


def test_qk_projection_helper_shape_logic_with_fake_separate_proj() -> None:
    class FakeProj(torch.nn.Module):
        def __init__(self, matrix):
            super().__init__()
            self.matrix = matrix

        def forward(self, x):
            return torch.matmul(x, self.matrix)

    class FakeAttn:
        def __init__(self):
            self.q_proj = FakeProj(torch.eye(4, dtype=torch.float32))
            self.k_proj = FakeProj(torch.eye(4, dtype=torch.float32))
            self.num_heads = 2
            self.num_key_value_heads = 2

    fake_layer = SimpleNamespace(self_attn=FakeAttn())
    fake_model = SimpleNamespace(
        model=SimpleNamespace(layers=[fake_layer]),
        config=SimpleNamespace(num_attention_heads=2, num_key_value_heads=2),
    )
    hidden_state = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    projection = module._project_layer_qk_from_hidden_state(
        model=fake_model,
        hidden_state=hidden_state,
        layer=0,
        extraction_mode="separate_qk_proj",
    )

    assert projection.resolved_mode == "separate_qk_proj"
    assert tuple(projection.q_by_head.shape) == (3, 2, 2)
    assert tuple(projection.k_by_kv_head.shape) == (3, 2, 2)


def test_output_schema_can_be_built_from_synthetic_tensors() -> None:
    backend_results = {
        "mean_key": {
            "score_by_block": [0.3, 0.2],
            "true_mass_by_block": [0.6, 0.4],
            "top_blocks_by_true_mass": [0, 1],
            "top_blocks_by_score": [0, 1],
            "topk_metrics": {
                "1": {
                    "attention_mass_recovered_by_score_topk": 0.6,
                    "oracle_true_topk_mass": 0.6,
                    "recovery_fraction_of_oracle": 1.0,
                    "topk_overlap_with_true_topk": 1,
                    "retained_block_fraction": 0.5,
                }
            },
            "rank_correlation_spearman": 1.0,
            "num_blocks": 2,
            "warnings": [],
        }
    }

    payload = module.build_output_payload(
        model="synthetic",
        prompt_kind="synthetic_long",
        token_count=12,
        block_size=4,
        num_blocks=3,
        selected_layers=[1],
        selected_heads=[0],
        selected_query_positions=[11],
        backend_results=backend_results,
    )

    assert payload["model"] == "synthetic"
    assert payload["prompt_kind"] == "synthetic_long"
    assert payload["selected_layers"] == [1]
    assert payload["backend_results"]["mean_key"]["top_blocks_by_score"] == [0, 1]


def test_invalid_backend_fails_safely() -> None:
    eval_points = [
        _make_eval_point(
            layer=0,
            head=0,
            query_position=2,
            query=[1.0, 0.0],
            keys=[[1.0, 0.0], [0.0, 1.0]],
        )
    ]

    with pytest.raises(ValueError, match="Unsupported backend"):
        module.evaluate_backend_scores(
            backend="not_real",
            eval_points=eval_points,
            block_size=1,
            sketch_dim=2,
            seed=1,
            topks=[1],
        )
