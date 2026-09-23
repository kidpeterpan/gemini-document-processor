"""The evaluation report must satisfy its published JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from booksum.config import Settings
from booksum.eval.dataset import load_dataset
from booksum.eval.harness import run_evaluation
from booksum.llm import FakeLLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "specs" / "002-summary-quality" / "contracts" / "eval-report.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_published_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_emitted_report_matches_schema(tmp_path, pdf_path):
    checklist = tmp_path / "checklist.md"
    checklist.write_text("- alpha :: Alpha fact\n- beta :: Beta fact\n", encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "source": pdf_path,
                "doc_type": "pdf",
                "checklist": str(checklist),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_evaluation(
        load_dataset(dataset),
        settings=Settings(data_dir=tmp_path / "runtime", chunk_size=1),
        client_factory=lambda: FakeLLMClient(responder=lambda prompt, model: prompt),
        repo_root=str(tmp_path),
    )

    jsonschema.validate(report, _schema())


def test_schema_rejects_malformed_report():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"passed": True}, _schema())
