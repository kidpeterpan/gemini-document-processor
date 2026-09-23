"""Per-job log isolation (feature 003 US3)."""

from __future__ import annotations

import logging

from web.logging_channel import JobLogHandler, JobLogRegistry


def test_channels_are_isolated():
    registry = JobLogRegistry()
    registry.sink("job-a", "alpha only")
    registry.sink("job-b", "beta only")

    assert registry.tail("job-a") == registry.tail("job-a")
    assert any("alpha only" in line for line in registry.tail("job-a"))
    assert not any("beta only" in line for line in registry.tail("job-a"))
    assert not any("alpha only" in line for line in registry.tail("job-b"))


def test_channel_is_bounded():
    registry = JobLogRegistry(max_entries=3)
    for index in range(10):
        registry.sink("job", f"line {index}")
    tail = registry.tail("job", 100)
    assert len(tail) == 3
    assert "line 9" in tail[-1]


def test_tail_default_is_fifteen():
    registry = JobLogRegistry()
    for index in range(30):
        registry.sink("job", f"line {index}")
    assert len(registry.tail("job")) == 15


def test_handler_routes_by_job_id():
    registry = JobLogRegistry()
    handler = JobLogHandler(registry)
    logger = logging.getLogger("booksum.test.logging")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("routed message", extra={"job_id": "job-x"})
        logger.info("unrouted message")
    finally:
        logger.removeHandler(handler)

    lines = registry.tail("job-x")
    assert any("routed message" in line for line in lines)
    assert not any("unrouted message" in line for line in lines)


def test_clear_empties_channel():
    registry = JobLogRegistry()
    registry.sink("job", "something")
    registry.clear("job")
    assert registry.tail("job") == []
