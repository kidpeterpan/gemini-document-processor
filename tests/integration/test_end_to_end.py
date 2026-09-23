"""End-to-end: real documents through the pure core (no server, no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from booksum.config import Settings
from booksum.llm import FakeLLMClient
from booksum.models import JobStatus
from booksum.queueing import BoundedExecutor
from booksum.service import JobService
from booksum.store import JobStore

IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def responder(prompt: str, model: str) -> str:
    if "Section summaries:" in prompt:
        return json.dumps(
            {
                "overview": "OVERVIEW",
                "key_ideas": ["IDEA"],
                "glossary": [{"term": "T", "definition": "D"}],
            }
        )
    if "Sections:" in prompt and '"score"' in prompt:
        return json.dumps({"units": [{"label": "Chunk 1", "missing": [], "score": 1.0}]})
    return "UNIT-SUMMARY"


def run_job(tmp_settings: Settings, path: str) -> tuple[JobService, JobStore, str]:
    store = JobStore(tmp_settings.db_path)
    service = JobService(
        store,
        tmp_settings,
        client_factory=lambda _settings: FakeLLMClient(responder=responder),
        executor=BoundedExecutor(1),
    )
    job_id = service.submit(path, tmp_settings, schedule=False)
    service.run(job_id)
    return service, store, job_id


def test_pdf_end_to_end(tmp_settings, pdf_path):
    service, store, job_id = run_job(tmp_settings, pdf_path)

    job = store.get_job(job_id)
    assert job.status is JobStatus.COMPLETED
    content = Path(job.output_path).read_text(encoding="utf-8")

    assert content.startswith("---")
    assert "## Synopsis" in content
    assert "OVERVIEW" in content
    assert "UNIT-SUMMARY" in content
    assert "prompt_versions:" in content
    assert store.unit_counts(job_id)["complete"] == 3
    service.shutdown()


def test_epub_end_to_end_images_resolve(tmp_settings, epub_path):
    service, store, job_id = run_job(tmp_settings, epub_path)

    job = store.get_job(job_id)
    assert job.status is JobStatus.COMPLETED
    output = Path(job.output_path)
    content = output.read_text(encoding="utf-8")

    links = IMAGE_LINK.findall(content)
    assert links, "the EPUB fixture contains an image that must be embedded"
    for link in links:
        target = (output.parent / link).resolve()
        assert target.exists(), f"dangling image link: {link}"

    assert store.unit_counts(job_id)["complete"] == 2
    service.shutdown()


def test_epub_without_images_has_no_dangling_links(tmp_settings, tmp_path):
    from tests.fixtures import make_epub

    path = make_epub(
        tmp_path / "plain.epub",
        [("Chapter One", "A chapter long enough to be considered content." * 4)],
    )
    service, store, job_id = run_job(tmp_settings, path)

    content = Path(store.get_job(job_id).output_path).read_text(encoding="utf-8")
    assert IMAGE_LINK.findall(content) == []
    service.shutdown()
