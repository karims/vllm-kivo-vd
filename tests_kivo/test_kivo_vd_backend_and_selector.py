# SPDX-License-Identifier: Apache-2.0

import numpy as np

from vllm.v1.core.kivo_vd_candidate_selector import (
    KivoVDCandidateSelector,
    KivoVDCandidateSelectorConfig,
)
from vllm.v1.core.kivo_vd_observer import KivoVDObserver
from vllm.v1.core.kivo_vd_sketch import (
    KivoVDBlockSketch,
    KivoVDSketchIndex,
    KivoVDSketchType,
)
from vllm.v1.core.kivo_vd_sketch_backend import (
    BidiagonalSignBackend,
    BidiagonalSignSubsampleBackend,
    CountSketchBackend,
    RandomProjectionBackend,
    SRHTBackend,
    TridiagonalSignBackend,
    make_sketch_backend,
)


def _mk_sketch(
    request_id: str,
    block_id: int,
    logical_block_idx: int,
) -> KivoVDBlockSketch:
    return KivoVDBlockSketch(
        request_id=request_id,
        block_id=block_id,
        logical_block_idx=logical_block_idx,
        kv_group_id=0,
        layer_id=None,
        sketch_dim=64,
        metadata={},
    )


def test_backend_factory_creates_expected_backends() -> None:
    assert isinstance(
        make_sketch_backend(KivoVDSketchType.COUNT_SKETCH),
        CountSketchBackend,
    )
    assert isinstance(
        make_sketch_backend(KivoVDSketchType.RANDOM_PROJECTION),
        RandomProjectionBackend,
    )
    assert isinstance(make_sketch_backend(KivoVDSketchType.SRHT), SRHTBackend)
    assert isinstance(
        make_sketch_backend(KivoVDSketchType.BIDIAGONAL_SIGN),
        BidiagonalSignBackend,
    )
    assert isinstance(
        make_sketch_backend(KivoVDSketchType.BIDIAGONAL_SIGN_SUBSAMPLE),
        BidiagonalSignSubsampleBackend,
    )
    assert isinstance(
        make_sketch_backend(KivoVDSketchType.TRIDIAGONAL_SIGN),
        TridiagonalSignBackend,
    )


def test_backend_params_deterministic_for_same_seed() -> None:
    count = CountSketchBackend()
    spec1 = count.make_params(input_dim=8, sketch_dim=4, seed=3)
    spec2 = count.make_params(input_dim=8, sketch_dim=4, seed=3)
    assert np.array_equal(spec1.bucket_index, spec2.bucket_index)
    assert np.array_equal(spec1.bucket_sign, spec2.bucket_sign)

    rp = RandomProjectionBackend()
    p1 = rp.make_params(input_dim=8, sketch_dim=4, seed=5)
    p2 = rp.make_params(input_dim=8, sketch_dim=4, seed=5)
    assert np.allclose(p1, p2)

    srht = SRHTBackend()
    s1 = srht.make_params(input_dim=10, sketch_dim=4, seed=11)
    s2 = srht.make_params(input_dim=10, sketch_dim=4, seed=11)
    assert np.array_equal(s1.signs, s2.signs)
    assert np.array_equal(s1.sampled_indices, s2.sampled_indices)

    bidiag = BidiagonalSignBackend()
    b1 = bidiag.make_params(input_dim=10, sketch_dim=4, seed=13)
    b2 = bidiag.make_params(input_dim=10, sketch_dim=4, seed=13)
    assert np.array_equal(b1.signs, b2.signs)
    assert np.array_equal(b1.sampled_indices, b2.sampled_indices)

    tri = TridiagonalSignBackend()
    t1 = tri.make_params(input_dim=10, sketch_dim=4, seed=15)
    t2 = tri.make_params(input_dim=10, sketch_dim=4, seed=15)
    assert np.array_equal(t1.signs, t2.signs)
    assert np.array_equal(t1.sampled_indices, t2.sampled_indices)


def test_candidate_selector_includes_recent_blocks() -> None:
    index = KivoVDSketchIndex()
    for i in range(10):
        index.add_or_update_block_sketch(_mk_sketch("r1", 100 + i, i))

    selector = KivoVDCandidateSelector(
        KivoVDCandidateSelectorConfig(
            recent_window_blocks=2,
            candidate_budget_blocks=4,
            min_candidate_blocks=2,
            include_recent_blocks=True,
        )
    )
    decision = selector.select_candidates("r1", None, index)
    assert len(decision.selected_block_ids) == 4
    assert 108 in decision.selected_block_ids
    assert 109 in decision.selected_block_ids


def test_candidate_selector_respects_budget() -> None:
    index = KivoVDSketchIndex()
    for i in range(12):
        index.add_or_update_block_sketch(_mk_sketch("r2", i, i))

    selector = KivoVDCandidateSelector(
        KivoVDCandidateSelectorConfig(
            candidate_budget_blocks=3,
            min_candidate_blocks=1,
            include_recent_blocks=False,
        )
    )
    decision = selector.select_candidates("r2", None, index)
    assert len(decision.selected_block_ids) == 3


def test_candidate_selector_fallback_empty_request() -> None:
    index = KivoVDSketchIndex()
    selector = KivoVDCandidateSelector(
        KivoVDCandidateSelectorConfig(
            fallback_to_all_on_empty=True,
        )
    )
    decision = selector.select_candidates("missing", None, index)
    assert decision.selected_block_ids == []
    assert decision.reason == "empty_request_blocks"


def test_observer_dry_run_select_candidates() -> None:
    index = KivoVDSketchIndex()
    index.add_or_update_block_sketch(_mk_sketch("r3", 1, 0))
    index.add_or_update_block_sketch(_mk_sketch("r3", 2, 1))

    selector = KivoVDCandidateSelector(
        KivoVDCandidateSelectorConfig(
            candidate_budget_blocks=1,
            min_candidate_blocks=1,
        )
    )
    observer = KivoVDObserver(
        enabled=True,
        sketch_index=index,
        candidate_selector=selector,
    )
    decision = observer.dry_run_select_candidates("r3", source="unit")
    assert decision is not None
    assert len(decision.selected_block_ids) == 1
    assert observer.get_counters()["num_dry_run_select_calls"] == 1

    events = observer.get_recent_events()
    assert events[-1]["event_type"] == "dry_run_routing_decision"
    assert events[-1]["request_id"] == "r3"
    assert events[-1]["selected_block_count"] == 1
    assert events[-1]["recent_block_count"] == 1
    assert events[-1]["skipped_block_count"] == 1
    assert events[-1]["candidate_budget_blocks"] == 1
    assert events[-1]["source"] == "unit"
    assert events[-1]["full_block_ids_exported"] is False
    assert "selected_block_ids_full" not in events[-1]


def test_observer_dry_run_can_export_full_block_ids() -> None:
    index = KivoVDSketchIndex()
    index.add_or_update_block_sketch(_mk_sketch("full", 1, 0))
    index.add_or_update_block_sketch(_mk_sketch("full", 2, 1))
    observer = KivoVDObserver(
        enabled=True,
        sketch_index=index,
        candidate_selector=KivoVDCandidateSelector(),
        export_full_block_ids=True,
    )

    decision = observer.dry_run_select_candidates("full", source="unit")

    assert decision is not None
    event = observer.get_recent_events()[-1]
    assert event["full_block_ids_exported"] is True
    assert event["selected_block_ids_full"] == decision.selected_block_ids
    assert event["recent_block_ids_full"] == decision.recent_block_ids
    assert event["skipped_block_ids_full"] == decision.skipped_block_ids
    assert event["selected_block_preview"] == decision.selected_block_ids[:8]


def test_observer_full_block_ids_can_be_enabled_by_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIVO_EXPORT_FULL_BLOCK_IDS", "1")

    observer = KivoVDObserver(
        enabled=True,
        sketch_index=KivoVDSketchIndex(),
        candidate_selector=KivoVDCandidateSelector(),
    )

    observer.dry_run_select_candidates("missing", source="env")
    event = observer.get_recent_events()[-1]
    assert event["full_block_ids_exported"] is True
    assert event["selected_block_ids_full"] == []


def test_observer_dry_run_empty_request_records_event() -> None:
    observer = KivoVDObserver(
        enabled=True,
        sketch_index=KivoVDSketchIndex(),
        candidate_selector=KivoVDCandidateSelector(),
    )
    decision = observer.dry_run_select_candidates("missing", source="empty")
    assert decision is not None
    assert decision.selected_block_ids == []

    events = observer.get_recent_events()
    assert events[-1]["event_type"] == "dry_run_routing_decision"
    assert events[-1]["selected_block_count"] == 0
    assert events[-1]["recent_block_count"] == 0
    assert events[-1]["skipped_block_count"] == 0
    assert events[-1]["source"] == "empty"
