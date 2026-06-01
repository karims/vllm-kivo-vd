# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.kivo_vd_sketch import (
    KivoVDBlockSketch,
    KivoVDSketchConfig,
    KivoVDSketchIndex,
)


def _mk_sketch(
    request_id: str,
    block_id: int,
    logical_block_idx: int,
    kv_group_id: int = 0,
) -> KivoVDBlockSketch:
    return KivoVDBlockSketch(
        request_id=request_id,
        block_id=block_id,
        logical_block_idx=logical_block_idx,
        kv_group_id=kv_group_id,
        layer_id=None,
        sketch_dim=64,
        metadata={},
    )


def test_add_update_and_get_request_block_sketches() -> None:
    index = KivoVDSketchIndex()
    index.add_or_update_block_sketch(_mk_sketch("r1", 10, 1))
    index.add_or_update_block_sketch(_mk_sketch("r1", 20, 2))
    sketches = index.get_request_block_sketches("r1")
    assert [s.block_id for s in sketches] == [10, 20]

    # Update same (kv_group_id, block_id).
    index.add_or_update_block_sketch(_mk_sketch("r1", 20, 3))
    updated = index.get_request_block_sketches("r1")
    assert len(updated) == 2
    assert [s.logical_block_idx for s in updated] == [1, 3]


def test_remove_request() -> None:
    index = KivoVDSketchIndex()
    index.add_or_update_block_sketch(_mk_sketch("r2", 1, 0))
    index.add_or_update_block_sketch(_mk_sketch("r3", 2, 0))
    index.remove_request("r2")
    assert index.get_request_block_sketches("r2") == []
    assert [s.block_id for s in index.get_request_block_sketches("r3")] == [2]


def test_score_blocks_placeholder_is_deterministic() -> None:
    index = KivoVDSketchIndex()
    for i in range(4):
        index.add_or_update_block_sketch(
            _mk_sketch("r4", block_id=100 + i, logical_block_idx=i)
        )

    scores_1 = index.score_blocks_placeholder("r4")
    scores_2 = index.score_blocks_placeholder("r4")
    assert [(s.block_id, s.score) for s in scores_1] == [
        (s.block_id, s.score) for s in scores_2
    ]


def test_route_blocks_placeholder_recent_then_scored() -> None:
    index = KivoVDSketchIndex(
        config=KivoVDSketchConfig(
            enabled=True,
            max_blocks_per_query=3,
            recent_window_blocks=1,
        )
    )
    for i in range(5):
        index.add_or_update_block_sketch(
            _mk_sketch("r5", block_id=i, logical_block_idx=i)
        )

    decision = index.route_blocks_placeholder(
        "r5",
        recent_block_ids=[4, 3, 2],
    )
    assert len(decision.selected_block_ids) == 3
    assert decision.recent_block_ids == [4]
    assert 4 in decision.selected_block_ids
    assert decision.reason == "placeholder_metadata_only"


def test_reset_clears_state() -> None:
    index = KivoVDSketchIndex()
    index.add_or_update_block_sketch(_mk_sketch("r6", 11, 0))
    assert index.get_request_block_sketches("r6")
    index.reset()
    assert index.get_request_block_sketches("r6") == []
