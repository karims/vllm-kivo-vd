# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from vllm.v1.core.kivo_vd_observer import (
    KivoVDObserver,
    create_kivo_vd_observer,
)
from vllm.v1.core.kivo_vd_sketch import KivoVDSketchIndex


def test_kivo_vd_observer_instantiation() -> None:
    observer = KivoVDObserver(enabled=False)
    assert observer.enabled is False


def test_kivo_vd_observer_factory_disabled_by_default() -> None:
    assert create_kivo_vd_observer(False) is None


def test_kivo_vd_observer_factory_enabled_has_sketch_index() -> None:
    observer = create_kivo_vd_observer(True)
    assert observer is not None
    assert observer.sketch_index is not None


def test_kivo_vd_observer_counters_increment() -> None:
    observer = KivoVDObserver(enabled=True)
    observer.on_before_allocate_slots("req-1", 4, 0, source="running")
    observer.on_after_allocate_slots(
        "req-1",
        ([1, 2],),
        num_new_tokens=4,
        source="running",
    )
    observer.on_free_request("req-1", ([1, 2],), source="preempt")

    assert observer.num_before_allocate_calls == 1
    assert observer.num_after_allocate_calls == 1
    assert observer.num_free_request_calls == 1


def test_kivo_vd_observer_events_captured() -> None:
    observer = KivoVDObserver(enabled=True)
    observer.on_before_allocate_slots("req-2", 8, 0, source="waiting")
    observer.on_after_allocate_slots("req-2", ([3], [4]), num_new_tokens=8)
    observer.on_free_request("req-2", ([3], [4]))

    events = observer.get_recent_events()
    assert len(events) == 3
    assert events[0]["event_type"] == "before_allocate_slots"
    assert events[1]["event_type"] == "after_allocate_slots"
    assert events[2]["event_type"] == "free_request"
    assert events[1]["num_new_blocks"] == 2


def test_kivo_vd_observer_event_buffer_bounded() -> None:
    observer = KivoVDObserver(enabled=True, max_events=3)
    for i in range(5):
        observer.on_before_allocate_slots(f"req-{i}", i + 1, 0)

    events = observer.get_recent_events(limit=10)
    assert len(events) == 3
    assert events[0]["request_id"] == "req-2"
    assert events[2]["request_id"] == "req-4"


def test_kivo_vd_observer_reset_clears_state() -> None:
    observer = KivoVDObserver(enabled=True)
    observer.on_before_allocate_slots("req-a", 1, 0)
    observer.on_after_allocate_slots("req-a", ([1],))
    observer.on_free_request("req-a", ([1],))
    assert observer.get_counters()["num_events"] == 3

    observer.reset()

    assert observer.get_counters() == {
        "num_before_allocate_calls": 0,
        "num_after_allocate_calls": 0,
        "num_free_request_calls": 0,
        "num_dry_run_select_calls": 0,
        "num_events": 0,
    }
    assert observer.get_recent_events() == []


def test_observer_updates_sketch_index_from_after_allocate() -> None:
    observer = KivoVDObserver(enabled=True, sketch_index=KivoVDSketchIndex())
    observer.on_after_allocate_slots(
        request_id="req-s1",
        block_ids_by_group=([10, 11], [21]),
        num_new_tokens=6,
        source="running",
    )

    sketches = observer.sketch_index.get_request_block_sketches("req-s1")
    assert len(sketches) == 3
    assert {(s.kv_group_id, s.block_id) for s in sketches} == {
        (0, 10),
        (0, 11),
        (1, 21),
    }
    assert all(s.metadata.get("source") == "running" for s in sketches)


def test_observer_free_request_removes_sketch_entries() -> None:
    observer = KivoVDObserver(enabled=True, sketch_index=KivoVDSketchIndex())
    observer.on_after_allocate_slots(
        request_id="req-s2",
        block_ids_by_group=([1, 2],),
        num_new_tokens=2,
        source="waiting",
    )
    assert observer.sketch_index.get_request_block_sketches("req-s2")

    observer.on_free_request("req-s2", ([1, 2],), source="free_blocks")
    assert observer.sketch_index.get_request_block_sketches("req-s2") == []


def test_observer_without_sketch_index_still_works() -> None:
    observer = KivoVDObserver(enabled=True, sketch_index=None)
    observer.on_after_allocate_slots(
        request_id="req-ns",
        block_ids_by_group=([7],),
        num_new_tokens=1,
        source="running",
    )
    observer.on_free_request("req-ns", ([7],), source="preempt")
    counters = observer.get_counters()
    assert counters["num_after_allocate_calls"] == 1
    assert counters["num_free_request_calls"] == 1


def test_observer_dry_run_without_sketch_index_returns_none() -> None:
    observer = KivoVDObserver(enabled=True, sketch_index=None)
    decision = observer.dry_run_select_candidates("req-empty", source="test")
    assert decision is None
    assert observer.get_counters()["num_dry_run_select_calls"] == 1


def test_observer_export_events_writes_jsonl(tmp_path: Path) -> None:
    export_path = tmp_path / "events.jsonl"
    observer = KivoVDObserver(enabled=True, event_export_path=str(export_path))
    observer.on_before_allocate_slots("req-export", 2, 0, source="unit")

    written = observer.export_events()

    assert written == 1
    row = json.loads(export_path.read_text(encoding="utf-8").strip())
    assert row["event_type"] == "before_allocate_slots"
    assert row["request_id"] == "req-export"


def test_observer_export_events_respects_limit(tmp_path: Path) -> None:
    export_path = tmp_path / "limited.jsonl"
    observer = KivoVDObserver(enabled=True, event_export_path=str(export_path))
    for i in range(5):
        observer.on_before_allocate_slots(f"req-{i}", 1, 0)

    written = observer.export_events(limit=2)

    rows = [
        json.loads(line)
        for line in export_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written == 2
    assert [row["request_id"] for row in rows] == ["req-3", "req-4"]


def test_observer_export_events_creates_parent_directory(tmp_path: Path) -> None:
    export_path = tmp_path / "nested" / "kivo" / "events.jsonl"
    observer = KivoVDObserver(enabled=True, event_export_path=str(export_path))
    observer.on_before_allocate_slots("req-parent", 1, 0)

    assert observer.export_events() == 1
    assert export_path.exists()


def test_observer_export_events_sanitizes_tensor_like_data(tmp_path: Path) -> None:
    class FakeTensor:
        shape = (1024, 1024)
        dtype = "float16"

    export_path = tmp_path / "sanitized.jsonl"
    observer = KivoVDObserver(enabled=True, event_export_path=str(export_path))
    observer._record_event("fake_tensor_event", payload=FakeTensor())

    observer.export_events()

    row = json.loads(export_path.read_text(encoding="utf-8").strip())
    assert row["event_type"] == "fake_tensor_event"
    assert row["payload"] == {"type": "FakeTensor"}
