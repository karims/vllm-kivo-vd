# SPDX-License-Identifier: Apache-2.0

from vllm.config.vllm import VllmConfig
from vllm.v1.core.kivo_vd_observer import (
    KivoVDObserver,
    create_kivo_vd_observer,
)


def test_kivo_vd_config_default_disabled() -> None:
    cfg = VllmConfig()
    assert cfg.enable_kivo_vd is False


def test_kivo_vd_observer_instantiation() -> None:
    observer = KivoVDObserver(enabled=False)
    assert observer.enabled is False


def test_kivo_vd_observer_factory_disabled_by_default() -> None:
    cfg = VllmConfig()
    assert create_kivo_vd_observer(cfg.enable_kivo_vd) is None


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
        "num_events": 0,
    }
    assert observer.get_recent_events() == []
