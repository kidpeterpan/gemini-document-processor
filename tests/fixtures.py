"""Real, minimal document fixtures built with declared dependencies.

``make_pdf`` writes a valid PDF by hand (no extra dependency); ``make_epub``
uses ``ebooklib``. Both produce files the production extractors can read, so
integration tests exercise real parsing rather than mocks.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from pathlib import Path


def make_pdf(path: str | Path, page_texts: Sequence[str]) -> str:
    """Write a valid multi-page PDF containing one line of text per page."""
    n = len(page_texts)
    if n == 0:
        raise ValueError("need at least one page")

    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, text in enumerate(page_texts):
        page_id = 4 + 2 * index
        content_id = page_id + 1
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()

        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 16 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        objects[content_id] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    total = max(objects) + 1
    for object_id in range(1, total):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"

    xref_position = len(out)
    out += f"xref\n0 {total}\n".encode()
    out += b"0000000000 65535 f \n"
    for object_id in range(1, total):
        out += f"{offsets[object_id]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n"
    ).encode()

    Path(path).write_bytes(bytes(out))
    return str(path)


def make_png_bytes(
    width: int = 120, height: int = 120, colour: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _pad(text: str, minimum: int = 220) -> str:
    filler = (
        " This paragraph intentionally carries enough words to be considered real book content."
    )
    while len(text) < minimum:
        text += filler
    return text


def make_epub(
    path: str | Path,
    chapters: Iterable[tuple[str, str]],
    *,
    image_name: str | None = None,
    image_bytes: bytes | None = None,
    title: str = "Test Book",
    author: str = "Test Author",
) -> str:
    """Write a real EPUB with the given chapters and an optional image."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("booksum-test")
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    items = []
    for index, (chapter_title, body) in enumerate(chapters, start=1):
        item = epub.EpubHtml(title=chapter_title, file_name=f"chap{index}.xhtml", lang="en")
        image_tag = ""
        if image_name and index == 1:
            image_tag = f'<img src="images/{image_name}" alt="figure"/>'
        item.content = f"<h1>{chapter_title}</h1><p>{_pad(body)}</p>{image_tag}"
        book.add_item(item)
        items.append(item)

    if image_name and image_bytes:
        book.add_item(
            epub.EpubItem(
                uid="figure-1",
                file_name=f"images/{image_name}",
                media_type="image/png",
                content=image_bytes,
            )
        )

    book.toc = items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]

    epub.write_epub(str(path), book)
    return str(path)
