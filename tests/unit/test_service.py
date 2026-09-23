"""JobService: durability, resume, stop, retry, and startup recovery."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from booksum.chunking import sha256_text
from booksum.config import Settings
from booksum.errors import BooksumError
from booksum.extraction import extract_document
from booksum.llm import FakeLLMClient
from booksum.models import JobStatus, UnitSeed, UnitStatus
from booksum.queueing import BoundedExecutor
from booksum.service import JobService
from booksum.store import JobStore


def structured_responder(prompt: str, model: str) -> str:
    if "Section summaries:" in prompt:
        return json.dumps(
            {
                "overview": "OVERVIEW-TEXT",
                "key_ideas": ["KEY-IDEA"],
                "glossary": [{"term": "TERM", "definition": "DEF"}],
            }
        )
    if "Sections:" in prompt and '"score"' in prompt:
        return json.dumps({"units": [{"label": "Chunk 1", "missing": [], "score": 1.0}]})
    return "SUMMARY-FOR-UNIT"


def build_service(
    store: JobStore, settings: Settings, client: FakeLLMClient, *, concurrency: int = 1
) -> JobService:
    return JobService(
        store,
        settings,
        client_factory=lambda _settings: client,
        executor=BoundedExecutor(concurrency),
    )


def offline_settings(tmp_settings: Settings, **overrides) -> Settings:
    snapshot = tmp_settings.to_snapshot()
    snapshot.pop("models", None)
    snapshot.update(
        {"synthesis_enabled": False, "coverage_enabled": False, "extract_images": False}
    )
    api_key = overrides.pop("api_key", None)
    snapshot.update(overrides)
    settings = Settings.from_snapshot(snapshot)
    if api_key is not None:
        settings.api_key = api_key
    return settings


# ---------------------------------------------------------------------- #
# Happy path
# ---------------------------------------------------------------------- #
def test_full_run_completes_and_exports(store, tmp_settings, pdf_path):
    client = FakeLLMClient(responder=structured_responder)
    service = build_service(store, tmp_settings, client)

    job_id = service.submit(pdf_path, tmp_settings, schedule=False)
    service.run(job_id)

    job = store.get_job(job_id)
    assert job.status is JobStatus.COMPLETED
    assert job.output_path and Path(job.output_path).exists()

    content = Path(job.output_path).read_text(encoding="utf-8")
    assert "## Synopsis" in content
    assert "OVERVIEW-TEXT" in content
    assert "SUMMARY-FOR-UNIT" in content
    assert "unit: unit.v1" in content

    counts = store.unit_counts(job_id)
    assert counts == {"pending": 0, "running": 0, "complete": 3, "failed": 0, "total": 3}

    assert (Path(tmp_settings.exports_dir) / Path(job.output_path).name).exists()
    service.shutdown()


def test_usage_counts_calls_and_artifacts_are_complete(store, tmp_settings, pdf_path):
    client = FakeLLMClient(responder=structured_responder)
    service = build_service(store, tmp_settings, client)
    job_id = service.submit(pdf_path, tmp_settings, schedule=False)
    service.run(job_id)

    usage = service.usage_totals(job_id)
    assert usage["calls"] == 5  # 3 units + synthesis + coverage
    assert store.get_artifact(job_id, "synthesis").status == "complete"
    assert store.get_artifact(job_id, "coverage").status == "complete"
    service.shutdown()


def test_api_key_is_not_persisted(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings)
    settings.api_key = "top-secret"
    service = build_service(store, settings, FakeLLMClient())
    job_id = service.submit(pdf_path, settings, schedule=False)

    stored = store.get_job(job_id).settings
    assert "api_key" not in stored
    assert "top-secret" not in json.dumps(stored)
    service.shutdown()


# ---------------------------------------------------------------------- #
# Resume / idempotency
# ---------------------------------------------------------------------- #
def test_second_run_reuses_cached_results(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings)
    client = FakeLLMClient()
    service = build_service(store, settings, client)

    first = service.submit(pdf_path, settings, schedule=False)
    service.run(first)
    assert client.call_count == 3

    client.calls.clear()
    second = service.submit(pdf_path, settings, schedule=False)
    service.run(second)

    assert client.call_count == 0, "identical content must not be paid for twice"
    assert store.unit_counts(second)["complete"] == 3
    service.shutdown()


def test_resume_processes_only_pending_units(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings)
    client = FakeLLMClient()
    service = build_service(store, settings, client)

    # Reference run: everything processed from scratch.
    reference = service.submit(pdf_path, settings, schedule=False)
    service.run(reference)
    reference_content = Path(store.get_job(reference).output_path).read_text(encoding="utf-8")

    # Interrupted run: pre-complete the first two units using the reference output.
    interrupted = service.submit(pdf_path, settings, schedule=False)
    doc = extract_document(pdf_path, settings)
    seeds = [
        UnitSeed(
            ordinal=unit.ordinal,
            label=unit.label,
            content_hash=sha256_text(unit.text),
            start_ref=unit.start_ref,
            end_ref=unit.end_ref,
        )
        for unit in doc.units
    ]
    store.add_units(interrupted, seeds)

    for ordinal in (1, 2):
        source_file = Path(tmp_settings.artifacts_dir) / reference / "units" / f"{ordinal:04d}.md"
        target_file = Path(tmp_settings.artifacts_dir) / interrupted / "units" / f"{ordinal:04d}.md"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
        store.save_unit_result(
            interrupted,
            ordinal,
            content_hash=seeds[ordinal - 1].content_hash,
            prompt_version="unit.v1",
            model=settings.model,
            summary=target_file.read_text(encoding="utf-8"),
            result_path=str(target_file),
        )

    client.calls.clear()
    store.clear_results()  # force unit 3 to be recomputed rather than cache-served
    service.run(interrupted)

    assert client.call_count == 1, "only the remaining unit should be re-sent"
    assert store.get_job(interrupted).status is JobStatus.COMPLETED

    resumed_content = Path(store.get_job(interrupted).output_path).read_text(encoding="utf-8")
    strip = lambda text: text.split("---\n*Summary generated", 1)[0]  # noqa: E731
    assert strip(resumed_content) == strip(reference_content)
    service.shutdown()


def test_resume_requires_api_key(store, tmp_settings, pdf_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = offline_settings(tmp_settings, api_key="")
    service = build_service(store, settings, FakeLLMClient())
    job_id = service.submit(pdf_path, settings, schedule=False)
    store.set_job_status(job_id, JobStatus.STOPPED.value)

    with pytest.raises(BooksumError):
        service.resume(job_id, schedule=False)
    service.shutdown()


def test_resume_refuses_running_job(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings)
    service = build_service(store, settings, FakeLLMClient())
    job_id = service.submit(pdf_path, settings, schedule=False)
    store.acquire_job(job_id, "someone")

    assert service.resume(job_id, schedule=False) is False
    service.shutdown()


def test_startup_recovery_resumes_orphans(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings, api_key="env-key")
    service = build_service(store, settings, FakeLLMClient())

    job_id = service.submit(pdf_path, settings, schedule=False)
    doc = extract_document(pdf_path, settings)
    store.add_units(
        job_id,
        [
            UnitSeed(unit.ordinal, unit.label, sha256_text(unit.text), unit.start_ref, unit.end_ref)
            for unit in doc.units
        ],
    )
    store.acquire_job(job_id, "dead-process")

    resumable = service.resume_on_startup(schedule=False)

    assert job_id in resumable
    assert store.get_job(job_id).status is JobStatus.STOPPED
    service.shutdown()


# ---------------------------------------------------------------------- #
# Stop
# ---------------------------------------------------------------------- #
def test_stop_halts_calls_and_preserves_state(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings)
    client = FakeLLMClient(block_until_cancel=True)
    service = build_service(store, settings, client)

    job_id = service.submit(pdf_path, settings, schedule=False)

    worker = threading.Thread(target=service.run, args=(job_id,), daemon=True)
    worker.start()
    time.sleep(0.3)
    assert service.request_stop(job_id) is True
    worker.join(timeout=10)
    service.shutdown()

    assert not worker.is_alive(), "stop must take effect promptly"
    assert store.get_job(job_id).status is JobStatus.STOPPED
    assert client.call_count == 0
    counts = store.unit_counts(job_id)
    assert counts["complete"] == 0
    assert counts["total"] == 3


def test_request_stop_unknown_job_returns_false(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings)
    service = build_service(store, settings, FakeLLMClient())
    assert service.request_stop("nope") is False
    service.shutdown()


# ---------------------------------------------------------------------- #
# Retry
# ---------------------------------------------------------------------- #
def test_retry_units_resets_failures(store, tmp_settings, pdf_path):
    settings = offline_settings(tmp_settings)
    service = build_service(store, settings, FakeLLMClient())

    job_id = service.submit(pdf_path, settings, schedule=False)
    doc = extract_document(pdf_path, settings)
    store.add_units(
        job_id,
        [
            UnitSeed(unit.ordinal, unit.label, sha256_text(unit.text), unit.start_ref, unit.end_ref)
            for unit in doc.units
        ],
    )
    store.mark_unit_failed(job_id, 1, "boom")

    assert service.retry_units(job_id, schedule=False) == 1
    assert store.get_unit(job_id, 1).status is UnitStatus.PENDING
    service.shutdown()


# ---------------------------------------------------------------------- #
# Degradation
# ---------------------------------------------------------------------- #
def test_synthesis_failure_degrades_but_job_completes(store, tmp_settings, pdf_path):
    # Default fake responder is not JSON, so synthesis/coverage fail.
    client = FakeLLMClient()
    service = build_service(store, tmp_settings, client)
    job_id = service.submit(pdf_path, tmp_settings, schedule=False)
    service.run(job_id)

    assert store.get_job(job_id).status is JobStatus.COMPLETED
    assert store.get_artifact(job_id, "synthesis").status == "failed"
    content = Path(store.get_job(job_id).output_path).read_text(encoding="utf-8")
    assert "SUMMARY-FOR-UNIT" not in content  # default responder output differs
    assert "status: failed" in content
    service.shutdown()
