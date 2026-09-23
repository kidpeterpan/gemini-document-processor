"""Markdown artifact contract (feature 002)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from booksum.assembly import missing_image_paths, render_markdown, write_markdown
from booksum.models import ExtractedImage

PROMPT_VERSIONS = {"unit": "unit.v1", "synthesis": "synth.v1", "coverage": "coverage.v1"}


def build(tmp_path: Path, **overrides: Any):
    output = tmp_path / "out" / "book_summary.md"
    kwargs: dict[str, Any] = dict(
        summaries={"Chunk 1": "first summary", "Chunk 2": "second summary"},
        images_by_unit={},
        doc_name="book",
        doc_type="pdf",
        metadata={"title": "The Book", "author": "A. Writer"},
        model="gemini-2.0-flash",
        prompt_versions=PROMPT_VERSIONS,
        output_path=str(output),
    )
    kwargs.update(overrides)
    return render_markdown(**kwargs)


def test_frontmatter_has_model_and_all_prompt_versions(tmp_path: Path):
    content = build(tmp_path)
    assert "model: gemini-2.0-flash" in content
    assert "prompt_versions:" in content
    assert "unit: unit.v1" in content
    assert "synthesis: synth.v1" in content
    assert "coverage: coverage.v1" in content


def test_all_units_appear_in_summary(tmp_path: Path):
    content = build(tmp_path)
    assert "### Chunk 1" in content
    assert "first summary" in content
    assert "### Chunk 2" in content
    assert "second summary" in content


def test_synthesis_sections_are_rendered(tmp_path: Path):
    content = build(
        tmp_path,
        synthesis={
            "overview": "Whole book overview",
            "key_ideas": ["idea A"],
            "glossary": [{"term": "T", "definition": "D"}],
        },
        synthesis_status="complete",
    )
    assert "## Synopsis" in content
    assert "Whole book overview" in content
    assert "### Key Ideas" in content
    assert "- idea A" in content
    assert "- **T** — D" in content


def test_failed_synthesis_is_visible_not_silent(tmp_path: Path):
    content = build(tmp_path, synthesis=None, synthesis_status="failed")
    assert "## Synopsis" in content
    assert "status: failed" in content


def test_coverage_notes_are_rendered(tmp_path: Path):
    content = build(
        tmp_path,
        coverage={
            "evaluated": True,
            "score": 0.75,
            "units": [{"label": "Chunk 1", "missing": ["fact x"], "score": 0.75}],
        },
        coverage_status="complete",
    )
    assert "## Coverage Notes" in content
    assert "Chunk 1" in content
    assert "fact x" in content


def test_not_evaluated_coverage_is_labelled(tmp_path: Path):
    content = build(
        tmp_path,
        coverage={"evaluated": False, "score": 0.0, "units": []},
        coverage_status="complete",
    )
    assert "not evaluated" in content


def test_existing_image_is_embedded_with_relative_path(tmp_path: Path):
    image_path = tmp_path / "images" / "figure.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png-bytes")

    content = build(
        tmp_path,
        images_by_unit={"Chunk 1": [ExtractedImage("figure.png", str(image_path), alt="Figure")]},
    )
    assert "![Figure](../images/figure.png)" in content
    assert missing_image_paths([ExtractedImage("figure.png", str(image_path))]) == []


def test_missing_image_is_omitted_and_noted(tmp_path: Path):
    ghost = tmp_path / "images" / "ghost.png"  # never written
    content = build(
        tmp_path,
        images_by_unit={"Chunk 1": [ExtractedImage("ghost.png", str(ghost), alt="Ghost")]},
    )
    assert "ghost.png" not in content
    assert "omitted" in content
    assert missing_image_paths([ExtractedImage("ghost.png", str(ghost))]) == [str(ghost)]


def test_usage_and_footer(tmp_path: Path):
    content = build(tmp_path, usage={"calls": 5, "prompt_tokens": 100, "output_tokens": 50})
    assert "5 calls" in content
    assert "150 tokens" in content
    assert "Summary generated using gemini-2.0-flash" in content


def test_write_markdown_creates_parent_dirs(tmp_path: Path):
    target = tmp_path / "deep" / "nested" / "out.md"
    write_markdown(str(target), "content")
    assert target.read_text(encoding="utf-8") == "content"
