"""Summarization passes: per-unit (map), synthesis (reduce), coverage check.

Synthesis and coverage are *optional*: a failure returns an outcome with
``status="failed"`` and never raises, so the per-unit detail is always delivered
(feature 002 FR-004). ``LLMCancelled`` is the one exception — it propagates so a
stop request can actually halt work.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import LLMCancelled, LLMError
from .llm import LLMClient, LLMResult, extract_json_object
from .prompts import PromptRegistry

logger = logging.getLogger("booksum.summarization")


@dataclass(frozen=True)
class SynthesisInput:
    label: str
    summary: str


@dataclass(frozen=True)
class CoverageInput:
    label: str
    source_text: str
    summary: str


@dataclass(frozen=True)
class SynthesisOutcome:
    status: str
    synthesis: dict | None
    model: str | None
    prompt_version: str
    error: str | None = None


@dataclass(frozen=True)
class CoverageOutcome:
    status: str
    report: dict | None
    model: str | None
    prompt_version: str
    error: str | None = None


# ---------------------------------------------------------------------- #
# Map: per-unit summary
# ---------------------------------------------------------------------- #
def summarize_unit(
    client: LLMClient,
    prompts: PromptRegistry,
    *,
    text: str,
    doc_type: str,
    doc_filename: str,
    start_ref: int | None,
    end_ref: int | None,
    models: Sequence[str],
    temperature: float = 0.1,
    max_output_tokens: int = 8192,
    timeout: float | None = None,
    cancel: threading.Event | None = None,
) -> LLMResult:
    page_or_chapter = "หน้า" if doc_type == "pdf" else "บท"
    prompt = prompts.render(
        "unit",
        doc_type=doc_type,
        doc_filename=doc_filename,
        page_or_chapter=page_or_chapter,
        start_ref=start_ref if start_ref is not None else "?",
        end_ref=end_ref if end_ref is not None else "?",
        text=text,
    )
    return client.generate(
        prompt,
        models=models,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        cancel=cancel,
    )


# ---------------------------------------------------------------------- #
# Reduce: whole-document synthesis
# ---------------------------------------------------------------------- #
def _render_block(items: Sequence[SynthesisInput]) -> str:
    return "\n\n".join(f"### {item.label}\n{item.summary}" for item in items)


def _group(items: Sequence[SynthesisInput], budget: int) -> list[list[SynthesisInput]]:
    groups: list[list[SynthesisInput]] = []
    current: list[SynthesisInput] = []
    size = 0
    for item in items:
        piece = len(item.summary) + len(item.label) + 8
        if current and size + piece > budget:
            groups.append(current)
            current = []
            size = 0
        current.append(item)
        size += piece
    if current:
        groups.append(current)
    return groups


def _validate_synthesis(payload: dict, *, grouped: bool) -> dict:
    overview = str(payload.get("overview", "")).strip()
    if not overview:
        raise LLMError("synthesis response has no 'overview'")

    raw_ideas = payload.get("key_ideas") or []
    if not isinstance(raw_ideas, list):
        raw_ideas = [raw_ideas]
    key_ideas = [str(item).strip() for item in raw_ideas if str(item).strip()]

    glossary: list[dict[str, str]] = []
    for entry in payload.get("glossary") or []:
        if isinstance(entry, dict) and entry.get("term") and entry.get("definition"):
            glossary.append(
                {"term": str(entry["term"]).strip(), "definition": str(entry["definition"]).strip()}
            )

    return {
        "overview": overview,
        "key_ideas": key_ideas,
        "glossary": glossary,
        "grouped": grouped,
    }


def synthesize(
    client: LLMClient,
    prompts: PromptRegistry,
    *,
    summaries: Sequence[SynthesisInput],
    doc_title: str,
    models: Sequence[str],
    budget_chars: int = 200_000,
    timeout: float | None = None,
    cancel: threading.Event | None = None,
) -> SynthesisOutcome:
    """Reduce unit summaries into a whole-document synthesis."""
    version = prompts.get("synthesis").version
    material = [item for item in summaries if item.summary and item.summary.strip()]
    if not material:
        return SynthesisOutcome("skipped", None, None, version, "no summaries to synthesize")

    grouped = False
    try:
        block = _render_block(material)
        if len(block) > budget_chars:
            grouped = True
            logger.info("synthesis input exceeds budget; using grouped reduce")
            reduced: list[SynthesisInput] = []
            for group in _group(material, max(budget_chars // 2, 1_000)):
                group_prompt = prompts.render(
                    "synthesis", doc_title=doc_title, summaries=_render_block(group)
                )
                result = client.generate(
                    group_prompt, models=models, timeout=timeout, cancel=cancel
                )
                partial = _validate_synthesis(extract_json_object(result.text), grouped=True)
                reduced.append(
                    SynthesisInput(
                        label=str(len(reduced) + 1),
                        summary=json.dumps(partial, ensure_ascii=False),
                    )
                )
            block = _render_block(reduced)

        prompt = prompts.render("synthesis", doc_title=doc_title, summaries=block)
        result = client.generate(prompt, models=models, timeout=timeout, cancel=cancel)
        payload = _validate_synthesis(extract_json_object(result.text), grouped=grouped)
        payload["source_units"] = [item.label for item in material]
        return SynthesisOutcome("complete", payload, result.model, version, None)
    except LLMCancelled:
        raise
    except LLMError as exc:
        logger.warning("synthesis failed: %s", exc)
        return SynthesisOutcome("failed", None, None, version, str(exc))
    except Exception as exc:  # noqa: BLE001 - optional pass must not fail the job
        logger.warning("synthesis errored: %s", exc)
        return SynthesisOutcome("failed", None, None, version, str(exc))


# ---------------------------------------------------------------------- #
# Reduce: coverage check
# ---------------------------------------------------------------------- #
def _validate_coverage(payload: dict) -> dict:
    units: list[dict] = []
    for entry in payload.get("units") or []:
        if not isinstance(entry, dict):
            continue
        missing = entry.get("missing") or []
        if not isinstance(missing, list):
            missing = [str(missing)]
        raw_score = entry.get("score")
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        units.append(
            {
                "label": str(entry.get("label", "?")),
                "missing": [str(item) for item in missing if str(item).strip()],
                "score": max(0.0, min(1.0, score)),
            }
        )
    if not units:
        raise LLMError("coverage response contained no units")
    score = sum(unit["score"] for unit in units) / len(units)
    return {"units": units, "score": score, "evaluated": True}


def check_coverage(
    client: LLMClient,
    prompts: PromptRegistry,
    *,
    units: Sequence[CoverageInput],
    models: Sequence[str],
    timeout: float | None = None,
    cancel: threading.Event | None = None,
) -> CoverageOutcome:
    """Report salient source facts missing from each unit's summary."""
    version = prompts.get("coverage").version
    evaluated = [unit for unit in units if unit.source_text and unit.source_text.strip()]
    if not evaluated:
        return CoverageOutcome(
            "skipped",
            {"units": [], "score": 0.0, "evaluated": False},
            None,
            version,
            "no source text available for coverage",
        )

    blocks = [
        f"## {unit.label}\n\n### SOURCE\n{unit.source_text}\n\n### SUMMARY\n{unit.summary}"
        for unit in evaluated
    ]
    try:
        prompt = prompts.render("coverage", units="\n\n".join(blocks))
        result = client.generate(prompt, models=models, timeout=timeout, cancel=cancel)
        report = _validate_coverage(extract_json_object(result.text))
        return CoverageOutcome("complete", report, result.model, version, None)
    except LLMCancelled:
        raise
    except LLMError as exc:
        logger.warning("coverage check failed: %s", exc)
        return CoverageOutcome("failed", None, None, version, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("coverage check errored: %s", exc)
        return CoverageOutcome("failed", None, None, version, str(exc))
