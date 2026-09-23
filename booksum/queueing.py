"""Bounded concurrency.

One shared executor bounds *all* in-flight unit work across every job, which is
the invariant the spec requires (feature 001 FR-005 / SC-004). A per-job pool
would break the global bound as soon as a second job starts.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


class BoundedExecutor:
    """A thread pool with an observable, enforced in-flight ceiling."""

    def __init__(self, max_workers: int, *, thread_name_prefix: str = "booksum") -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = int(max_workers)
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix=thread_name_prefix
        )
        self._lock = threading.Lock()
        self._in_flight = 0
        self._max_observed = 0

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def max_observed(self) -> int:
        with self._lock:
            return self._max_observed

    def _run(self, fn: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
        with self._lock:
            self._in_flight += 1
            self._max_observed = max(self._max_observed, self._in_flight)
        try:
            return fn(*args, **kwargs)
        finally:
            with self._lock:
                self._in_flight -= 1

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        return self._pool.submit(self._run, fn, args, kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> BoundedExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown(wait=True)
