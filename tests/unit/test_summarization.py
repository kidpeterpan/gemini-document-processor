"""Prompt registry, synthesis (reduce), and coverage checking."""

from __future__ import annotations

import json

import pytest

from booksum.llm import FakeLLMClient
from booksum.prompts import PromptRegistry
from booksum.summarization import (
    CoverageInput,
    SynthesisInput,
    check_coverage,
    summarize_unit,
    synthesize,
)

PROMPTS = PromptRegistry()
MODELS = ("gemini-2.0-flash",)


# ---------------------------------------------------------------------- #
# Prompts
# ---------------------------------------------------------------------- #
def test_prompt_versions_are_stable():
    assert PROMPTS.versions() == {
        "unit": "unit.v1",
        "synthesis": "synth.v1",
        "coverage": "coverage.v1",
    }


def test_render_missing_placeholder_raises_value_error():
    with pytest.raises(ValueError):
        PROMPTS.render("unit", doc_type="pdf")


def test_unknown_prompt_raises_key_error():
    with pytest.raises(KeyError):
        PROMPTS.get("nope")


# ---------------------------------------------------------------------- #
# Map
# ---------------------------------------------------------------------- #
def test_summarize_unit_sends_source_text():
    client = FakeLLMClient()
    result = summarize_unit(
        client,
        PROMPTS,
        text="UNIQUE-SOURCE-MARKER",
        doc_type="pdf",
        doc_filename="book.pdf",
        start_ref=1,
        end_ref=7,
        models=MODELS,
    )
    assert result.text
    assert "UNIQUE-SOURCE-MARKER" in client.calls[0].prompt


# ---------------------------------------------------------------------- #
# Reduce: synthesis
# ---------------------------------------------------------------------- #
def test_synthesize_returns_validated_payload():
    payload = {
        "overview": "The book argues X.",
        "key_ideas": ["idea one", "idea two"],
        "glossary": [{"term": "X", "definition": "the thing"}],
    }
    client = FakeLLMClient(responder=lambda prompt, model: json.dumps(payload))
    outcome = synthesize(
        client,
        PROMPTS,
        summaries=[SynthesisInput("Chunk 1", "a"), SynthesisInput("Chunk 2", "b")],
        doc_title="Book",
        models=MODELS,
    )
    assert outcome.status == "complete"
    assert outcome.synthesis["overview"] == "The book argues X."
    assert outcome.synthesis["key_ideas"] == ["idea one", "idea two"]
    assert outcome.synthesis["glossary"][0]["term"] == "X"
    assert outcome.synthesis["source_units"] == ["Chunk 1", "Chunk 2"]
    assert outcome.prompt_version == "synth.v1"


def test_synthesize_requires_overview():
    client = FakeLLMClient(responder=lambda prompt, model: '{"key_ideas": []}')
    outcome = synthesize(
        client,
        PROMPTS,
        summaries=[SynthesisInput("Chunk 1", "a")],
        doc_title="Book",
        models=MODELS,
    )
    assert outcome.status == "failed"
    assert "overview" in (outcome.error or "")


def test_synthesize_degrades_instead_of_raising():
    client = FakeLLMClient(fail_models=MODELS)
    outcome = synthesize(
        client,
        PROMPTS,
        summaries=[SynthesisInput("Chunk 1", "a")],
        doc_title="Book",
        models=MODELS,
    )
    assert outcome.status == "failed"
    assert outcome.synthesis is None


def test_synthesize_skips_when_no_summaries():
    outcome = synthesize(FakeLLMClient(), PROMPTS, summaries=[], doc_title="Book", models=MODELS)
    assert outcome.status == "skipped"


def test_synthesize_uses_grouped_reduce_over_budget():
    calls = {"n": 0}

    def responder(prompt: str, model: str) -> str:
        calls["n"] += 1
        return json.dumps({"overview": f"synthesis {calls['n']}", "key_ideas": [], "glossary": []})

    summaries = [SynthesisInput(f"Chunk {i}", "x" * 400) for i in range(1, 6)]
    client = FakeLLMClient(responder=responder)
    outcome = synthesize(
        client,
        PROMPTS,
        summaries=summaries,
        doc_title="Book",
        models=MODELS,
        budget_chars=500,
    )
    assert outcome.status == "complete"
    assert outcome.synthesis["grouped"] is True
    assert calls["n"] > 1


# ---------------------------------------------------------------------- #
# Reduce: coverage
# ---------------------------------------------------------------------- #
def test_coverage_reports_seeded_omission():
    report = {"units": [{"label": "Chunk 1", "missing": ["the second law"], "score": 0.5}]}
    client = FakeLLMClient(responder=lambda prompt, model: json.dumps(report))
    outcome = check_coverage(
        client,
        PROMPTS,
        units=[CoverageInput("Chunk 1", "source text", "short summary")],
        models=MODELS,
    )
    assert outcome.status == "complete"
    assert outcome.report["evaluated"] is True
    assert outcome.report["units"][0]["missing"] == ["the second law"]
    assert outcome.report["score"] == 0.5


def test_coverage_complete_summary_has_no_missing():
    report = {"units": [{"label": "Chunk 1", "missing": [], "score": 1.0}]}
    client = FakeLLMClient(responder=lambda prompt, model: json.dumps(report))
    outcome = check_coverage(
        client,
        PROMPTS,
        units=[CoverageInput("Chunk 1", "source text", "thorough summary")],
        models=MODELS,
    )
    assert outcome.report["units"][0]["missing"] == []


def test_coverage_without_source_is_not_evaluated():
    outcome = check_coverage(
        FakeLLMClient(),
        PROMPTS,
        units=[CoverageInput("Chunk 1", "", "summary")],
        models=MODELS,
    )
    assert outcome.report["evaluated"] is False
    assert outcome.report["units"] == []
    assert outcome.status == "skipped"


def test_coverage_degrades_instead_of_raising():
    client = FakeLLMClient(fail_models=MODELS)
    outcome = check_coverage(
        client,
        PROMPTS,
        units=[CoverageInput("Chunk 1", "source", "summary")],
        models=MODELS,
    )
    assert outcome.status == "failed"
    assert outcome.report is None


def test_coverage_clamps_scores_and_rejects_empty_units():
    client = FakeLLMClient(
        responder=lambda prompt, model: json.dumps(
            {"units": [{"label": "C1", "missing": ["x"], "score": 5}]}
        )
    )
    outcome = check_coverage(
        client, PROMPTS, units=[CoverageInput("C1", "s", "sum")], models=MODELS
    )
    assert outcome.report["units"][0]["score"] == 1.0
