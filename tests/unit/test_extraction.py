"""Real PDF/EPUB extraction, including the EPUB image-bytes fix."""

from __future__ import annotations

import os
from typing import Any

import pytest

from booksum.config import Settings
from booksum.errors import ExtractionError
from booksum.extraction import extract_document

from ..fixtures import make_epub, make_pdf, make_png_bytes


def settings(tmp_path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(data_dir=tmp_path / "runtime", chunk_size=1, extract_images=True)
    base.update(overrides)
    return Settings(**base)


def test_pdf_units_and_text(tmp_path):
    path = make_pdf(
        tmp_path / "book.pdf",
        ["Alpha sentence here.", "Beta sentence here.", "Gamma sentence here."],
    )
    doc = extract_document(path, settings(tmp_path))
    assert doc.doc_type.value == "pdf"
    assert len(doc.units) == 3
    assert doc.units[0].label == "Chunk 1"
    assert "Alpha sentence" in doc.units[0].text
    assert "Gamma sentence" in doc.units[2].text
    assert doc.units[2].start_ref == 3


def test_pdf_chunk_size_groups_pages(tmp_path):
    path = make_pdf(tmp_path / "book.pdf", [f"Page {i} content." for i in range(1, 6)])
    doc = extract_document(path, settings(tmp_path, chunk_size=2))
    assert len(doc.units) == 3
    assert doc.units[0].start_ref == 1 and doc.units[0].end_ref == 2
    assert doc.units[2].start_ref == 5 and doc.units[2].end_ref == 5


def test_epub_chapters_and_metadata(tmp_path):
    path = make_epub(
        tmp_path / "book.epub",
        [
            ("Chapter One", "Quantum mechanics describes small scales."),
            ("Chapter Two", "Entanglement links distant particles."),
        ],
        title="Physics Primer",
        author="A. Writer",
    )
    doc = extract_document(path, settings(tmp_path))
    assert doc.doc_type.value == "epub"
    assert len(doc.units) == 2
    assert doc.units[0].label == "Chapter 1"
    assert "Quantum mechanics" in doc.units[0].text
    assert doc.metadata.get("title") == "Physics Primer"


def test_epub_images_are_written_and_resolvable(tmp_path):
    path = make_epub(
        tmp_path / "book.epub",
        [("Chapter One", "A chapter long enough to be considered content." * 4)],
        image_name="figure.png",
        image_bytes=make_png_bytes(),
    )
    image_dir = tmp_path / "out_images"
    doc = extract_document(path, settings(tmp_path), image_dir=str(image_dir))

    images = doc.units[0].images
    assert len(images) == 1
    written = images[0].path
    assert os.path.exists(written), "EPUB image bytes must actually be written"
    assert os.path.getsize(written) > 0
    assert written.startswith(str(image_dir))


def test_epub_unresolved_image_is_skipped_not_linked(tmp_path):
    path = make_epub(
        tmp_path / "book.epub",
        [("Chapter One", "A chapter long enough to be considered content." * 4)],
        image_name="ghost.png",  # referenced but never added to the manifest
        image_bytes=None,
    )
    doc = extract_document(path, settings(tmp_path), image_dir=str(tmp_path / "imgs"))
    assert doc.units[0].images == []


def test_epub_without_images(tmp_path):
    path = make_epub(
        tmp_path / "book.epub",
        [("Chapter One", "A chapter long enough to be considered content." * 4)],
    )
    doc = extract_document(path, settings(tmp_path), image_dir=str(tmp_path / "imgs"))
    assert doc.units[0].images == []


def test_missing_file_raises():
    with pytest.raises(ExtractionError):
        extract_document("/nonexistent/book.pdf", Settings())


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_document(str(path), settings(tmp_path))


def test_empty_pdf_is_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with pytest.raises(ExtractionError):
        extract_document(str(path), settings(tmp_path))


def test_images_disabled_writes_nothing(tmp_path):
    path = make_epub(
        tmp_path / "book.epub",
        [("Chapter One", "A chapter long enough to be considered content." * 4)],
        image_name="figure.png",
        image_bytes=make_png_bytes(),
    )
    doc = extract_document(path, settings(tmp_path, extract_images=False))
    assert doc.image_dir is None
    assert doc.units[0].images == []
