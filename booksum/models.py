"""Domain status enums and value objects.

No persistence or framework imports belong here so every layer can depend on
these types safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DocumentType(StrEnum):
    PDF = "pdf"
    EPUB = "epub"

    @classmethod
    def from_path(cls, path: str) -> DocumentType:
        lowered = path.lower()
        if lowered.endswith(".epub"):
            return cls.EPUB
        if lowered.endswith(".pdf"):
            return cls.PDF
        raise ValueError(f"unsupported document type: {path}")


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED)


class UnitStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ArtifactStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactKind(StrEnum):
    SYNTHESIS = "synthesis"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class UnitSeed:
    """A unit to be persisted before processing begins."""

    ordinal: int
    label: str
    content_hash: str
    start_ref: int | None = None
    end_ref: int | None = None


@dataclass(frozen=True)
class Unit:
    id: int
    job_id: str
    ordinal: int
    label: str
    content_hash: str
    start_ref: int | None
    end_ref: int | None
    status: UnitStatus
    attempts: int
    result_path: str | None
    error: str | None


@dataclass(frozen=True)
class Job:
    id: str
    source_path: str
    source_name: str
    doc_type: DocumentType
    status: JobStatus
    settings: dict[str, Any]
    output_path: str | None
    metadata: dict[str, Any]
    error: str | None
    owner_token: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Artifact:
    job_id: str
    kind: str
    status: str
    model: str | None
    prompt_version: str | None
    content: str | None
    created_at: str


@dataclass
class ExtractedImage:
    filename: str
    path: str
    alt: str = ""
    width: int = 0
    height: int = 0
    page: int | None = None
    chapter: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "alt": self.alt,
            "width": self.width,
            "height": self.height,
            "page": self.page,
            "chapter": self.chapter,
        }


@dataclass
class ExtractedUnit:
    ordinal: int
    label: str
    text: str
    start_ref: int | None = None
    end_ref: int | None = None
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    doc_type: DocumentType
    name: str
    metadata: dict[str, Any]
    units: list[ExtractedUnit]
    image_dir: str | None = None
