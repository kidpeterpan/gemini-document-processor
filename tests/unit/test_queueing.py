"""Bounded concurrency (feature 001 US4 / SC-004)."""

from __future__ import annotations

import time
from concurrent.futures import wait

from booksum.queueing import BoundedExecutor


def test_never_exceeds_max_workers():
    with BoundedExecutor(3) as executor:
        futures = [executor.submit(time.sleep, 0.05) for _ in range(10)]
        wait(futures)
    assert executor.max_observed == 3
    assert executor.max_observed <= executor.max_workers


def test_returns_results():
    with BoundedExecutor(2) as executor:
        futures = [executor.submit(lambda value=index: value * 2) for index in range(4)]
    assert sorted(future.result() for future in futures) == [0, 2, 4, 6]


def test_rejects_invalid_max_workers():
    try:
        BoundedExecutor(0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_in_flight_returns_to_zero():
    with BoundedExecutor(2) as executor:
        futures = [executor.submit(time.sleep, 0.01) for _ in range(4)]
        wait(futures)
        assert executor.in_flight == 0
