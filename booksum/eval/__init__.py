"""Output-quality measurement: dataset, scoring, and harness.

Deliberately does not import :mod:`booksum.eval.harness` here: the harness is
also a CLI entry point (``python -m booksum.eval.harness``), and importing it
eagerly makes that invocation emit a runpy warning. Import the submodules
directly instead.
"""

from .dataset import (
    ChecklistItem,
    EvalDocument,
    load_checklist,
    load_dataset,
    validate_dataset,
)
from .scoring import ScoreResult, score_summary

__all__ = [
    "ChecklistItem",
    "EvalDocument",
    "ScoreResult",
    "load_checklist",
    "load_dataset",
    "validate_dataset",
    "score_summary",
]
