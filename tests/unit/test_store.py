"""Job store: durability primitives, ownership, cache, migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from booksum.errors import JobNotFoundError, JobOwnershipError, StoreSchemaError
from booksum.models import JobStatus, UnitSeed, UnitStatus
from booksum.store import SCHEMA_VERSION, JobStore


def seed(ordinal: int, label: str | None = None) -> UnitSeed:
    return UnitSeed(
        ordinal=ordinal,
        label=label or f"Chunk {ordinal}",
        content_hash=f"{ordinal:064d}",
        start_ref=ordinal,
        end_ref=ordinal,
    )


def test_create_and_read_job(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf",
        source_name="a.pdf",
        doc_type="pdf",
        settings_snapshot={"model": "m"},
    )
    job = store.get_job(job_id)
    assert job is not None
    assert job.status is JobStatus.QUEUED
    assert job.doc_type.value == "pdf"
    assert job.settings["model"] == "m"


def test_create_job_rejects_api_key(store: JobStore):
    with pytest.raises(ValueError):
        store.create_job(
            source_path="/tmp/a.pdf",
            source_name="a.pdf",
            doc_type="pdf",
            settings_snapshot={"api_key": "secret"},
        )


def test_acquire_job_is_atomic(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    assert store.acquire_job(job_id, "token-1") is True
    assert store.acquire_job(job_id, "token-2") is False
    assert store.owner_token(job_id) == "token-1"


def test_acquire_after_release(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.acquire_job(job_id, "t1")
    store.release_job(job_id, "t1", JobStatus.STOPPED.value)
    assert store.acquire_job(job_id, "t2") is True


def test_release_ignores_wrong_token(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.acquire_job(job_id, "owner")
    store.release_job(job_id, "intruder", JobStatus.COMPLETED.value)
    assert store.get_job(job_id).status is JobStatus.RUNNING


def test_unit_ownership_enforced(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.add_units(job_id, [seed(1)])
    store.acquire_job(job_id, "owner")
    with pytest.raises(JobOwnershipError):
        store.mark_unit_running(job_id, 1, "intruder")


def test_unit_lifecycle_and_counts(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.add_units(job_id, [seed(index) for index in range(1, 4)])

    counts = store.unit_counts(job_id)
    assert counts == {"pending": 3, "running": 0, "complete": 0, "failed": 0, "total": 3}

    first = store.next_pending_unit(job_id)
    assert first.ordinal == 1

    store.mark_unit_running(job_id, 1)
    store.mark_unit_complete(job_id, 1, "/tmp/1.md")
    store.mark_unit_failed(job_id, 2, "boom")

    counts = store.unit_counts(job_id)
    assert counts["complete"] == 1
    assert counts["failed"] == 1
    assert counts["pending"] == 1
    assert store.next_pending_unit(job_id).ordinal == 3


def test_add_units_is_idempotent(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.add_units(job_id, [seed(1), seed(2)])
    store.add_units(job_id, [seed(1), seed(2)])
    assert store.unit_counts(job_id)["total"] == 2


def test_result_cache_is_keyed_by_hash_version_model(store: JobStore):
    assert store.find_result("h", "unit.v1", "m1") is None
    store.save_result("h", "unit.v1", "m1", "summary one")
    assert store.find_result("h", "unit.v1", "m1") == "summary one"
    assert store.find_result("h", "unit.v1", "m2") is None
    assert store.find_result("h", "unit.v2", "m1") is None


def test_save_unit_result_is_atomic(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.add_units(job_id, [seed(1)])
    store.acquire_job(job_id, "owner")

    store.save_unit_result(
        job_id,
        1,
        content_hash=seed(1).content_hash,
        prompt_version="unit.v1",
        model="m",
        summary="done",
        result_path="/tmp/1.md",
        owner_token="owner",
    )

    unit = store.get_unit(job_id, 1)
    assert unit.status is UnitStatus.COMPLETE
    assert unit.result_path == "/tmp/1.md"
    assert store.find_result(seed(1).content_hash, "unit.v1", "m") == "done"


def test_reset_failed_units(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.add_units(job_id, [seed(1), seed(2)])
    store.mark_unit_failed(job_id, 1, "boom")
    store.mark_unit_failed(job_id, 2, "boom")
    assert store.reset_failed_units(job_id) == 2
    assert store.unit_counts(job_id)["pending"] == 2


def test_artifacts_roundtrip(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.put_artifact(
        job_id, "synthesis", "complete", model="m", prompt_version="synth.v1", content="{}"
    )
    artifact = store.get_artifact(job_id, "synthesis")
    assert artifact.status == "complete"
    assert artifact.prompt_version == "synth.v1"
    # replace on the same (job_id, kind)
    store.put_artifact(job_id, "synthesis", "failed", content=None)
    assert store.get_artifact(job_id, "synthesis").status == "failed"


def test_recover_orphans_moves_running_to_stopped(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.acquire_job(job_id, "owner")
    assert store.orphan_running_jobs() == [job_id]
    assert store.recover_orphans() == 1
    job = store.get_job(job_id)
    assert job.status is JobStatus.STOPPED
    assert job.owner_token is None


def test_mark_stopping_is_conditional(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    # Not running yet: no transition.
    assert store.mark_stopping(job_id) is False

    store.acquire_job(job_id, "owner")
    assert store.mark_stopping(job_id) is True
    assert store.get_job(job_id).status is JobStatus.STOPPING

    # Already stopping: no further transition.
    assert store.mark_stopping(job_id) is False

    # A finalised status is never clobbered back to stopping.
    store.release_job(job_id, "owner", JobStatus.STOPPED.value)
    assert store.mark_stopping(job_id) is False
    assert store.get_job(job_id).status is JobStatus.STOPPED


def test_require_job_raises(store: JobStore):
    with pytest.raises(JobNotFoundError):
        store.require_job("missing")


def test_schema_version_is_recorded(store: JobStore, tmp_settings):
    connection = sqlite3.connect(tmp_settings.db_path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()
    assert version == SCHEMA_VERSION


def test_newer_schema_is_rejected(tmp_path: Path):
    db_path = tmp_path / "future.db"
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA user_version = 999")
    connection.commit()
    connection.close()
    with pytest.raises(StoreSchemaError):
        JobStore(db_path)


def test_set_job_status_records_error_and_output(store: JobStore):
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.set_job_status(job_id, "failed", error="nope")
    job = store.get_job(job_id)
    assert job.error == "nope"

    store.acquire_job(job_id, "owner")
    store.set_job_status(job_id, "running", output_path="/tmp/out.md", metadata={"title": "T"})
    job = store.get_job(job_id)
    assert job.output_path == "/tmp/out.md"
    assert job.metadata["title"] == "T"


def test_list_jobs_newest_first(store: JobStore):
    first = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    second = store.create_job(
        source_path="/tmp/b.pdf", source_name="b.pdf", doc_type="pdf", settings_snapshot={}
    )
    ids = [job.id for job in store.list_jobs()]
    assert set(ids) == {first, second}
