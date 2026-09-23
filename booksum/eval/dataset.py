"""Evaluation dataset and checklist loading.

Checklist line format::

    - <human label> :: <variant one> | <variant two>

An item counts as captured when ANY of its variants is found in the summary.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistItem:
    item_id: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class EvalDocument:
    id: str
    source: str
    doc_type: str
    checklist: str
    min_capture: float | None = None


def load_dataset(path: str | os.PathLike) -> list[EvalDocument]:
    documents: list[EvalDocument] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            for key in ("id", "source", "doc_type", "checklist"):
                if key not in payload:
                    raise ValueError(f"{path}:{line_number}: missing '{key}'")
            documents.append(
                EvalDocument(
                    id=str(payload["id"]),
                    source=str(payload["source"]),
                    doc_type=str(payload["doc_type"]),
                    checklist=str(payload["checklist"]),
                    min_capture=(
                        float(payload["min_capture"])
                        if payload.get("min_capture") is not None
                        else None
                    ),
                )
            )
    return documents


def load_checklist(path: str | os.PathLike) -> list[ChecklistItem]:
    items: list[ChecklistItem] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("- ") or "::" not in stripped:
                continue
            label, _, variants_blob = stripped[2:].partition("::")
            variants = tuple(
                variant.strip() for variant in variants_blob.split("|") if variant.strip()
            )
            if variants:
                items.append(ChecklistItem(item_id=label.strip(), variants=variants))
    return items


def validate_dataset(documents: Iterable[EvalDocument], repo_root: str | os.PathLike) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    issues: list[str] = []
    for document in documents:
        source = (
            document.source
            if os.path.isabs(document.source)
            else os.path.join(repo_root, document.source)
        )
        checklist = (
            document.checklist
            if os.path.isabs(document.checklist)
            else os.path.join(repo_root, document.checklist)
        )
        if not os.path.exists(source):
            issues.append(f"{document.id}: source not found: {document.source}")
        if not os.path.exists(checklist):
            issues.append(f"{document.id}: checklist not found: {document.checklist}")
            continue
        if not load_checklist(checklist):
            issues.append(f"{document.id}: checklist has no items: {document.checklist}")
    return issues
