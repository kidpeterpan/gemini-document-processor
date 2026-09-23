"""The evaluation report must satisfy its published JSON Schema.

The schema ships with the package (``booksum/eval/eval_report.schema.json``) so
this contract is verifiable in CI, where the local Spec Kit artifacts under
``specs/`` are not committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from booksum.config import Settings
from booksum.eval.dataset import load_dataset
from booksum.eval.harness import run_evaluation
from booksum.eval.schema import SCHEMA_PATH, load_schema
from booksum.llm import FakeLLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
# Local Spec Kit copy of the same contract; kept in sync when present.
SPECS_SCHEMA_PATH = (
    REPO_ROOT / "specs" / "002-summary-quality" / "contracts" / "eval-report.schema.json"
)


def _schema() -> dict:
    return load_schema()


def test_schema_ships_with_the_package():
    assert SCHEMA_PATH.exists(), f"schema must ship with the package: {SCHEMA_PATH}"


def test_published_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_specs_copy_matches_package_copy_when_present():
    """Keep the local Spec Kit artifact in sync without requiring it in CI."""
    if not SPECS_SCHEMA_PATH.exists():
        pytest.skip("local Spec Kit schema copy is not present (expected in CI)")
    assert json.loads(SPECS_SCHEMA_PATH.read_text(encoding="utf-8")) == _schema()


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
