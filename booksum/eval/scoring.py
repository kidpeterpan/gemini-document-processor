"""Deterministic, offline scoring of a summary against a curated checklist.

This is intentionally lexical: it measures whether a fact is *present*, which
is exactly the product promise, and it is reproducible in CI. An LLM judge is
deliberately not used on the gating path (feature 002 R5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .dataset import ChecklistItem


@dataclass(frozen=True)
class ScoreResult:
    captured: int
    total: int
    missing: tuple[str, ...]

    @property
    def capture_rate(self) -> float:
        return self.captured / self.total if self.total else 0.0


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace so matching is stable."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def score_summary(summary: str, items: list[ChecklistItem]) -> ScoreResult:
    haystack = normalize(summary)
    captured = 0
    missing: list[str] = []
    for item in items:
        if any(normalize(variant) in haystack for variant in item.variants if variant.strip()):
            captured += 1
        else:
            missing.append(item.item_id)
    return ScoreResult(captured=captured, total=len(items), missing=tuple(missing))
