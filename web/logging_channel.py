"""Per-job log channels.

Replaces the previous approach of reassigning ``sys.stdout`` globally, which
interleaved concurrent jobs' logs (feature 003 US3).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

DEFAULT_MAX_ENTRIES = 500


class JobLogChannel:
    """Bounded, thread-safe log buffer for one job."""

    def __init__(self, job_id: str, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.job_id = job_id
        self._entries: deque[str] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def append(self, message: str, level: str = "info") -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        with self._lock:
            self._entries.append(entry)

    def tail(self, n: int = 15) -> list[str]:
        with self._lock:
            entries = list(self._entries)
        return entries[-n:] if n > 0 else entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class JobLogRegistry:
    """Maps job ids to channels and provides the log sink."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._channels: dict[str, JobLogChannel] = {}
        self._lock = threading.Lock()

    def channel(self, job_id: str) -> JobLogChannel:
        with self._lock:
            channel = self._channels.get(job_id)
            if channel is None:
                channel = JobLogChannel(job_id, self._max_entries)
                self._channels[job_id] = channel
            return channel

    def sink(self, job_id: str, message: str) -> None:
        self.channel(job_id).append(message)

    def tail(self, job_id: str, n: int = 15) -> list[str]:
        return self.channel(job_id).tail(n)

    def clear(self, job_id: str) -> None:
        self.channel(job_id).clear()


class JobLogHandler(logging.Handler):
    """Routes records carrying ``job_id`` in ``extra`` to that job's channel."""

    def __init__(self, registry: JobLogRegistry, attribute: str = "job_id") -> None:
        super().__init__()
        self._registry = registry
        self._attribute = attribute

    def emit(self, record: logging.LogRecord) -> None:
        job_id = getattr(record, self._attribute, None)
        if job_id is None:
            return
        try:
            self._registry.channel(str(job_id)).append(
                record.getMessage(), record.levelname.lower()
            )
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)
