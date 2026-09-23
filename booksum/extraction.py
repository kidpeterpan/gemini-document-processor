"""Document extraction: PDF and EPUB text, images and metadata.

Images are written to disk as a side effect, and the recorded path is the path
that was actually written. EPUB image references are resolved against the
manifest bytes; an image whose bytes cannot be resolved is skipped and counted
rather than linked (feature 002 FR-007 / R6).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import html2text
import pypdf

from .chunking import page_ranges
from .config import Settings
from .errors import ExtractionError
from .models import (
    DocumentType,
    ExtractedDocument,
    ExtractedImage,
    ExtractedUnit,
)

logger = logging.getLogger("booksum.extraction")


# ---------------------------------------------------------------------- #
# Public entry point
# ---------------------------------------------------------------------- #
def extract_document(
    path: str | os.PathLike,
    settings: Settings,
    image_dir: str | os.PathLike | None = None,
) -> ExtractedDocument:
    """Extract a document into units of text plus written image files."""
    path = str(path)
    if not os.path.exists(path):
        raise ExtractionError(f"document not found: {path}")

    doc_type = DocumentType.from_path(path)
    name = os.path.splitext(os.path.basename(path))[0]

    resolved_image_dir: str | None = None
    if settings.extract_images:
        resolved_image_dir = (
            str(image_dir) if image_dir else os.path.join(os.path.dirname(path), f"{name}_images")
        )
        os.makedirs(resolved_image_dir, exist_ok=True)

    if doc_type is DocumentType.PDF:
        return _extract_pdf(path, name, settings, resolved_image_dir)
    return _extract_epub(path, name, settings, resolved_image_dir)


# ---------------------------------------------------------------------- #
# PDF
# ---------------------------------------------------------------------- #
def _pdf_page_count(pdf_path: str) -> int:
    try:
        with open(pdf_path, "rb") as handle:
            return len(pypdf.PdfReader(handle).pages)
    except Exception as exc:  # noqa: BLE001 - surface as ExtractionError
        raise ExtractionError(f"cannot read PDF: {exc}") from exc


def _extract_pdf(
    path: str, name: str, settings: Settings, image_dir: str | None
) -> ExtractedDocument:
    metadata = _pdf_metadata(path)
    total_pages = _pdf_page_count(path)
    units: list[ExtractedUnit] = []

    for ordinal, (start, end) in enumerate(page_ranges(total_pages, settings.chunk_size), start=1):
        text = _pdf_text(path, start, end)
        images: list[ExtractedImage] = []
        if settings.extract_images and image_dir:
            images = _pdf_images(path, start, end, image_dir, settings)
        units.append(
            ExtractedUnit(
                ordinal=ordinal,
                label=f"Chunk {ordinal}",
                text=text,
                start_ref=start,
                end_ref=end,
                images=images,
            )
        )

    if not units:
        raise ExtractionError("PDF contains no pages")

    return ExtractedDocument(
        doc_type=DocumentType.PDF,
        name=name,
        metadata=metadata,
        units=units,
        image_dir=image_dir,
    )


def _pdf_text(pdf_path: str, start_page: int, end_page: int) -> str:
    try:
        with open(pdf_path, "rb") as handle:
            reader = pypdf.PdfReader(handle)
            last = min(end_page, len(reader.pages))
            parts: list[str] = []
            for index in range(start_page - 1, last):
                try:
                    page_text = reader.pages[index].extract_text() or ""
                except Exception as exc:  # noqa: BLE001
                    logger.warning("text extraction failed on page %s: %s", index + 1, exc)
                    page_text = f"[Error extracting text: {exc}]"
                parts.append(f"--- Page {index + 1} ---\n{page_text}")
            return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"cannot read PDF text: {exc}") from exc


def _pdf_images(
    pdf_path: str,
    start_page: int,
    end_page: int,
    output_dir: str,
    settings: Settings,
) -> list[ExtractedImage]:
    extracted: list[ExtractedImage] = []
    try:
        with open(pdf_path, "rb") as handle:
            reader = pypdf.PdfReader(handle)
            last = min(end_page, len(reader.pages))
            for page_index in range(start_page - 1, last):
                page_number = page_index + 1
                try:
                    images = list(getattr(reader.pages[page_index], "images", []) or [])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("image access failed on page %s: %s", page_number, exc)
                    continue

                for position, image in enumerate(images, start=1):
                    record = _write_pdf_image(image, page_number, position, output_dir, settings)
                    if record is not None:
                        extracted.append(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF image extraction failed: %s", exc)
    return extracted


def _write_pdf_image(
    image: object,
    page_number: int,
    position: int,
    output_dir: str,
    settings: Settings,
) -> ExtractedImage | None:
    pil_image = getattr(image, "image", None)
    width = int(getattr(pil_image, "width", 0) or 0)
    height = int(getattr(pil_image, "height", 0) or 0)
    if width and height and (width < settings.min_img_width or height < settings.min_img_height):
        return None

    filename = f"page{page_number:03d}_img{position:03d}.{settings.img_format}"
    target = os.path.join(output_dir, filename)
    try:
        if pil_image is not None:
            fmt = "JPEG" if settings.img_format.lower() in {"jpg", "jpeg"} else "PNG"
            pil_image.save(target, format=fmt)
        else:
            data = getattr(image, "data", None)
            if not data:
                return None
            with open(target, "wb") as handle:
                handle.write(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write image on page %s: %s", page_number, exc)
        return None

    if not os.path.exists(target) or os.path.getsize(target) == 0:
        return None

    return ExtractedImage(
        filename=filename,
        path=target,
        alt=f"Image {position} from page {page_number}",
        width=width,
        height=height,
        page=page_number,
    )


def _pdf_metadata(pdf_path: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    try:
        with open(pdf_path, "rb") as handle:
            raw = pypdf.PdfReader(handle).metadata
        if raw:
            for key in raw:
                clean = str(key).lower().replace("/", "").strip()
                if raw[key]:
                    metadata[clean] = str(raw[key])
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF metadata extraction failed: %s", exc)
    metadata.setdefault("title", os.path.splitext(os.path.basename(pdf_path))[0])
    return metadata


# ---------------------------------------------------------------------- #
# EPUB
# ---------------------------------------------------------------------- #
def _extract_epub(
    path: str, name: str, settings: Settings, image_dir: str | None
) -> ExtractedDocument:
    try:
        import ebooklib
        from bs4 import BeautifulSoup
        from ebooklib import epub
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ExtractionError(f"EPUB support unavailable: {exc}") from exc

    try:
        book = epub.read_epub(path)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"cannot read EPUB: {exc}") from exc

    metadata = _epub_metadata(book, name)
    converter = _make_converter()

    # Map image items by basename so `<img src>` can resolve to real bytes.
    image_items: dict[str, Any] = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            key = os.path.basename(item.get_name() or "")
            if key:
                image_items[key] = item

    units: list[ExtractedUnit] = []
    chapter_index = 0
    unresolved = 0

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        try:
            content = item.get_content().decode("utf-8", errors="replace")
            soup = BeautifulSoup(content, "html.parser")
            text = converter.handle(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping unreadable chapter item: %s", exc)
            continue

        if not _has_content(text):
            continue

        chapter_index += 1
        images: list[ExtractedImage] = []
        if settings.extract_images and image_dir:
            images, skipped = _write_epub_images(
                soup, image_items, image_dir, name, chapter_index, settings
            )
            unresolved += skipped

        units.append(
            ExtractedUnit(
                ordinal=chapter_index,
                label=f"Chapter {chapter_index}",
                text=text,
                start_ref=chapter_index,
                end_ref=chapter_index,
                images=images,
            )
        )

    if unresolved:
        logger.info("skipped %s unresolved EPUB image reference(s)", unresolved)

    if not units:
        raise ExtractionError("EPUB contains no readable content")

    return ExtractedDocument(
        doc_type=DocumentType.EPUB,
        name=name,
        metadata=metadata,
        units=units,
        image_dir=image_dir,
    )


def _write_epub_images(
    soup: Any,
    image_items: dict[str, Any],
    output_dir: str,
    epub_name: str,
    chapter_index: int,
    settings: Settings,
) -> tuple[list[ExtractedImage], int]:
    """Write bytes for each resolvable ``<img>``; return (written, skipped)."""
    written: list[ExtractedImage] = []
    skipped = 0

    for position, tag in enumerate(soup.find_all("img"), start=1):
        src = (tag.get("src") or "").strip()
        if not src:
            continue
        key = os.path.basename(src)
        item = image_items.get(key)
        if item is None:
            skipped += 1
            logger.debug("unresolved EPUB image reference: %s", src)
            continue

        try:
            data = item.get_content()
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.warning("could not read EPUB image %s: %s", src, exc)
            continue
        if not data:
            skipped += 1
            continue

        ext = os.path.splitext(key)[1].lstrip(".").lower() or settings.img_format
        filename = f"{epub_name}_chapter{chapter_index:03d}_img{position:03d}.{ext}"
        target = os.path.join(output_dir, filename)
        try:
            with open(target, "wb") as handle:
                handle.write(data)
        except OSError as exc:
            skipped += 1
            logger.warning("could not write EPUB image %s: %s", src, exc)
            continue

        if not os.path.exists(target) or os.path.getsize(target) == 0:
            skipped += 1
            continue

        written.append(
            ExtractedImage(
                filename=filename,
                path=target,
                alt=(tag.get("alt") or "").strip() or f"Figure {position}",
                chapter=chapter_index,
            )
        )

    return written, skipped


_DC_FIELDS = ("title", "creator", "language", "publisher", "date", "description", "identifier")


def _epub_metadata(book: Any, fallback_title: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    try:
        for field in _DC_FIELDS:
            values = book.get_metadata("DC", field)
            if not values:
                continue
            entry = values[0]
            metadata[field] = entry[0] if isinstance(entry, (tuple, list)) else str(entry)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EPUB metadata extraction failed: %s", exc)
    if metadata.get("creator"):
        metadata.setdefault("author", metadata["creator"])
    metadata.setdefault("title", fallback_title)
    return metadata


def _make_converter() -> html2text.HTML2Text:
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = False
    converter.ignore_tables = False
    converter.body_width = 0
    return converter


def _has_content(text: str | None, threshold: int = 100) -> bool:
    """True when text has substantial content (not just navigation chrome)."""
    if not text:
        return False
    return len(re.sub(r"\s+|[^\w]", "", text)) > threshold
