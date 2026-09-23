"""Markdown assembly.

The generated document is the product; its structure is a contract (see
``specs/002-summary-quality/contracts/artifact-contract.md``). Two invariants
matter most:

* version traceability — frontmatter always carries ``model`` and every prompt
  version;
* no dangling links — an image is only embedded if its file exists, otherwise
  it is omitted and the omission is noted.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

from .models import ExtractedImage

logger = logging.getLogger("booksum.assembly")

_YAML_SPECIAL = set(":#{}[],&*?|<>=!%@`\"'")
_YAML_KEYWORDS = {"true", "false", "null", "yes", "no", "on", "off", "~"}


def _yaml_scalar(value: Any) -> str:
    text = "" if value is None else str(value)
    needs_quotes = (
        text == ""
        or text != text.strip()
        or "\n" in text
        or any(ch in _YAML_SPECIAL for ch in text)
        or text.startswith("-")
        or text.lower() in _YAML_KEYWORDS
    )
    if needs_quotes:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return f'"{escaped}"'
    return text


def _anchor(label: str) -> str:
    return label.lower().replace(" ", "-")


def _image_path(image: ExtractedImage, output_dir: str | None) -> str | None:
    path = image.path
    if not path or not os.path.exists(path):
        return None
    if output_dir:
        return os.path.relpath(path, output_dir)
    return path


def render_markdown(
    *,
    summaries: Mapping[str, str],
    images_by_unit: Mapping[str, list[ExtractedImage]],
    doc_name: str,
    doc_type: str,
    metadata: Mapping[str, Any] | None = None,
    obsidian_metadata: Mapping[str, Any] | None = None,
    synthesis: Mapping[str, Any] | None = None,
    synthesis_status: str | None = None,
    coverage: Mapping[str, Any] | None = None,
    coverage_status: str | None = None,
    model: str = "",
    prompt_versions: Mapping[str, str] | None = None,
    usage: Mapping[str, Any] | None = None,
    output_path: str | None = None,
) -> str:
    """Render the final Markdown document."""
    metadata = dict(metadata or {})
    obsidian_metadata = dict(obsidian_metadata or {})
    prompt_versions = dict(prompt_versions or {})
    output_dir = os.path.dirname(output_path) if output_path else None

    lines: list[str] = []

    # ---- frontmatter -------------------------------------------------- #
    lines.append("---")
    tags = [tag.strip() for tag in str(obsidian_metadata.get("tags", "")).split(",") if tag.strip()]
    if tags:
        lines.append("tags:")
        lines.extend(f"  - {_yaml_scalar(tag)}" for tag in tags)
    lines.append(f"model: {_yaml_scalar(model)}")
    if prompt_versions:
        lines.append("prompt_versions:")
        for key in sorted(prompt_versions):
            lines.append(f"  {key}: {_yaml_scalar(prompt_versions[key])}")
    for key, value in obsidian_metadata.items():
        if key == "tags" or not value:
            continue
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    lines.append("")

    # ---- title and document info -------------------------------------- #
    title = metadata.get("title") or doc_name
    lines.append(f"# {title}")
    lines.append("")
    if metadata.get("author"):
        lines.append(f"**Author:** {metadata['author']}")
        lines.append("")

    lines.append("## Document Information")
    lines.append("")
    lines.append(f"- **Type:** {doc_type.upper()}")
    for key, value in metadata.items():
        if key in {"title", "author"} or not value:
            continue
        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    lines.append("")

    # ---- synthesis ---------------------------------------------------- #
    if synthesis:
        lines.append("## Synopsis")
        lines.append("")
        overview = str(synthesis.get("overview", "")).strip()
        if overview:
            lines.append(overview)
            lines.append("")

        key_ideas = synthesis.get("key_ideas") or []
        if key_ideas:
            lines.append("### Key Ideas")
            lines.append("")
            lines.extend(f"- {idea}" for idea in key_ideas)
            lines.append("")

        glossary = synthesis.get("glossary") or []
        if glossary:
            lines.append("### Glossary")
            lines.append("")
            for entry in glossary:
                term = entry.get("term", "")
                definition = entry.get("definition", "")
                lines.append(f"- **{term}** — {definition}")
            lines.append("")
    elif synthesis_status:
        lines.append("## Synopsis")
        lines.append("")
        lines.append(
            f"_Synthesis was not available (status: {synthesis_status}). "
            "The section summaries below are complete._"
        )
        lines.append("")

    # ---- table of contents -------------------------------------------- #
    lines.append("## Table of Contents")
    lines.append("")
    for index, label in enumerate(summaries.keys(), start=1):
        lines.append(f"{index}. [{label}](#{_anchor(label)})")
    lines.append("")

    # ---- per-unit detail ---------------------------------------------- #
    lines.append("## Summary")
    lines.append("")
    for label, summary in summaries.items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append(summary.strip() if summary else "_(no content)_")
        lines.append("")

        images = images_by_unit.get(label) or []
        if images:
            embedded: list[str] = []
            omitted = 0
            for image in images:
                rel = _image_path(image, output_dir)
                if rel is None:
                    omitted += 1
                    logger.warning("omitting missing image: %s", image.path)
                    continue
                alt = image.alt or f"Image from {label}"
                embedded.append(f"![{alt}]({rel})")
            if embedded:
                lines.append("#### Images")
                lines.append("")
                lines.extend(embedded)
                lines.append("")
            if omitted:
                lines.append(f"_{omitted} image(s) omitted because the source file was missing._")
                lines.append("")

    # ---- coverage ----------------------------------------------------- #
    if coverage:
        lines.append("## Coverage Notes")
        lines.append("")
        if coverage.get("evaluated") is False:
            lines.append("_Coverage was not evaluated for this document._")
            lines.append("")
        else:
            for unit in coverage.get("units") or []:
                label = unit.get("label", "?")
                score = unit.get("score")
                missing = unit.get("missing") or []
                score_text = (
                    f"coverage {float(score):.2f}"
                    if isinstance(score, (int, float))
                    else "coverage n/a"
                )
                if missing:
                    lines.append(f"- **{label}** — {score_text}; missing: {', '.join(missing)}")
                else:
                    lines.append(f"- **{label}** — {score_text}; nothing salient missing")
            lines.append("")
    elif coverage_status:
        lines.append("## Coverage Notes")
        lines.append("")
        lines.append(f"_Coverage check was not available (status: {coverage_status})._")
        lines.append("")

    # ---- footer ------------------------------------------------------- #
    lines.append("---")
    footer = f"*Summary generated using {model or 'unknown model'}"
    if prompt_versions.get("unit"):
        footer += f" ({prompt_versions['unit']})"
    if usage:
        calls = usage.get("calls")
        tokens = (usage.get("prompt_tokens") or 0) + (usage.get("output_tokens") or 0)
        parts = []
        if calls is not None:
            parts.append(f"{calls} calls")
        if tokens:
            parts.append(f"{tokens} tokens")
        if parts:
            footer += " · " + " · ".join(parts)
    footer += "*"
    lines.append(footer)
    lines.append("")

    return "\n".join(lines)


def write_markdown(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def missing_image_paths(images: Iterable[ExtractedImage]) -> list[str]:
    """Return paths of images that are referenced but do not exist."""
    return [image.path for image in images if not image.path or not os.path.exists(image.path)]
