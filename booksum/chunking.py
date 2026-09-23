"""Content hashing and unit-range helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator


def sha256_text(text: str) -> str:
    """Stable hex digest of text, used as the idempotency component.

    The digest is combined with prompt version and model to form the result
    cache key (see ``specs/001-durable-pipeline/research.md`` R3), so it must be
    deterministic across runs and platforms.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_ranges(total_pages: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield 1-based inclusive ``(start, end)`` page ranges.

    >>> list(page_ranges(10, 4))
    [(1, 4), (5, 8), (9, 10)]
    """
    if total_pages <= 0:
        return
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    start = 1
    while start <= total_pages:
        end = min(start + chunk_size - 1, total_pages)
        yield start, end
        start = end + 1
