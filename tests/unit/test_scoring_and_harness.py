"""Checklist scoring and the evaluation harness gate."""

from __future__ import annotations

import json
from pathlib import Path

from booksum.config import Settings
from booksum.eval.dataset import (
    ChecklistItem,
    load_checklist,
    load_dataset,
    validate_dataset,
)
from booksum.eval.harness import run_evaluation
from booksum.eval.scoring import normalize, score_summary
from booksum.llm import FakeLLMClient

from ..fixtures import make_pdf


def test_normalize_collapses_whitespace_and_case():
    assert normalize("Hello   WORLD\n") == "hello world"


def test_score_summary_counts_captured_and_missing():
    items = [
        ChecklistItem("alpha", ("alpha fact", "α fact")),
        ChecklistItem("beta", ("beta fact",)),
    ]
    result = score_summary("The Alpha Fact is true.", items)
    assert result.captured == 1
    assert result.missing == ("beta",)
    assert result.capture_rate == 0.5


def test_score_summary_matches_any_variant():
    items = [ChecklistItem("alpha", ("english variant", "thai variant"))]
    assert score_summary("thai variant appears", items).captured == 1


def test_score_summary_empty_checklist():
    result = score_summary("anything", [])
    assert result.total == 0
    assert result.capture_rate == 0.0


def test_load_checklist_parses_variants(tmp_path: Path):
    path = tmp_path / "checklist.md"
    path.write_text(
        "# Checklist\n- first fact :: variant one | variant two\n- second :: only\nignored line\n",
        encoding="utf-8",
    )
    items = load_checklist(path)
    assert len(items) == 2
    assert items[0].item_id == "first fact"
    assert items[0].variants == ("variant one", "variant two")


def build_eval_workspace(tmp_path: Path) -> tuple[Path, Path]:
    source = make_pdf(
        tmp_path / "source.pdf",
        ["Alpha fact appears here.", "Beta fact appears here."],
    )
    checklist = tmp_path / "checklist.md"
    checklist.write_text("- alpha :: Alpha fact\n- beta :: Beta fact\n", encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "source": source,
                "doc_type": "pdf",
                "checklist": str(checklist),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset, checklist


def test_dataset_validation_reports_missing_files(tmp_path: Path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        json.dumps({"id": "x", "source": "nope.pdf", "doc_type": "pdf", "checklist": "nope.md"})
        + "\n",
        encoding="utf-8",
    )
    issues = validate_dataset(load_dataset(broken), tmp_path)
    assert len(issues) == 2


def test_harness_passes_when_facts_are_present(tmp_path: Path):
    dataset, _ = build_eval_workspace(tmp_path)
    settings = Settings(data_dir=tmp_path / "runtime", chunk_size=1)
    report = run_evaluation(
        load_dataset(dataset),
        settings=settings,
        client_factory=lambda: FakeLLMClient(responder=lambda prompt, model: prompt),
        repo_root=str(tmp_path),
        report_path=str(tmp_path / "report.json"),
    )
    assert report["passed"] is True
    assert report["aggregate_capture_rate"] == 1.0
    assert (tmp_path / "report.json").exists()


def test_harness_fails_on_seeded_regression(tmp_path: Path):
    dataset, _ = build_eval_workspace(tmp_path)
    settings = Settings(data_dir=tmp_path / "runtime", chunk_size=1, coverage_threshold=0.9)

    def regressed(prompt: str, model: str) -> str:
        return prompt.replace("Beta fact", "")

    report = run_evaluation(
        load_dataset(dataset),
        settings=settings,
        client_factory=lambda: FakeLLMClient(responder=regressed),
        repo_root=str(tmp_path),
    )
    assert report["passed"] is False
    document = report["documents"][0]
    assert document["status"] == "evaluated"
    assert "beta" in document["missing"]
    assert document["capture_rate"] < 0.9


def test_harness_reports_missing_source_as_error(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {"id": "ghost", "source": "ghost.pdf", "doc_type": "pdf", "checklist": "ghost.md"}
        )
        + "\n",
        encoding="utf-8",
    )
    report = run_evaluation(
        load_dataset(dataset),
        settings=Settings(data_dir=tmp_path / "runtime"),
        client_factory=FakeLLMClient,
        repo_root=str(tmp_path),
    )
    assert report["documents"][0]["status"] == "error"
    assert report["passed"] is False


def test_harness_records_usage(tmp_path: Path):
    dataset, _ = build_eval_workspace(tmp_path)
    report = run_evaluation(
        load_dataset(dataset),
        settings=Settings(data_dir=tmp_path / "runtime", chunk_size=1),
        client_factory=lambda: FakeLLMClient(responder=lambda prompt, model: prompt),
        repo_root=str(tmp_path),
    )
    document = report["documents"][0]
    # 2 units + 1 synthesis call + 1 coverage call
    assert document["calls"] == 4
