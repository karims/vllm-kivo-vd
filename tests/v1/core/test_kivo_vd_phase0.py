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
    observer.on_before_allocate_slots("req-1", 4, 0)
    observer.on_after_allocate_slots("req-1", ([1, 2],))
    observer.on_free_request("req-1", ([1, 2],))

    assert observer.num_before_allocate_calls == 1
    assert observer.num_after_allocate_calls == 1
    assert observer.num_free_request_calls == 1
