"""Durable job store backed by SQLite.

Implements the contract in
``specs/001-durable-pipeline/contracts/store-interface.md``:

* one atomic ``acquire_job`` compare-and-set (research R4);
* content-addressed result cache (research R3);
* unit result + status written in a single transaction (FR-011);
* no method accepts or stores an API key (FR-012).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .errors import JobNotFoundError, JobOwnershipError, StoreSchemaError
from .models import (
    Artifact,
    DocumentType,
    Job,
    JobStatus,
    Unit,
    UnitSeed,
    UnitStatus,
)

SCHEMA_VERSION = 1

_DDL_V1 = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    source_name   TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    status        TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    output_path   TEXT,
    metadata_json TEXT,
    error         TEXT,
    owner_token   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS units (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    label        TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    start_ref    INTEGER,
    end_ref      INTEGER,
    status       TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    result_path  TEXT,
    error        TEXT,
    UNIQUE (job_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_units_job_status ON units (job_id, status);

CREATE TABLE IF NOT EXISTS results (
    content_hash   TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    summary        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (content_hash, prompt_version, model)
);

CREATE TABLE IF NOT EXISTS artifacts (
    job_id         TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,
    status         TEXT NOT NULL,
    model          TEXT,
    prompt_version TEXT,
    content        TEXT,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (job_id, kind)
);
"""

_ACQUIRABLE = (JobStatus.QUEUED.value, JobStatus.STOPPED.value, JobStatus.FAILED.value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobStore:
    """Thread-safe SQLite repository for jobs, units, results and artifacts."""

    def __init__(self, db_path: str | os.PathLike) -> None:
        self._db_path = str(db_path)
        parent = os.path.dirname(os.path.abspath(self._db_path))
        os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # ------------------------------------------------------------------ #
    # Migrations
    # ------------------------------------------------------------------ #
    def _migrate(self) -> None:
        with self._lock:
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if current > SCHEMA_VERSION:
                raise StoreSchemaError(
                    f"store schema v{current} is newer than supported v{SCHEMA_VERSION}"
                )
            if current == 0:
                with self._conn:
                    self._conn.executescript(_DDL_V1)
                    self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> JobStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Jobs
    # ------------------------------------------------------------------ #
    def create_job(
        self,
        *,
        source_path: str,
        source_name: str,
        doc_type: str,
        settings_snapshot: Mapping[str, Any],
    ) -> str:
        import uuid

        if "api_key" in settings_snapshot:
            raise ValueError("settings snapshot must not contain 'api_key'")
        job_id = str(uuid.uuid4())
        now = _utc_now()
        payload = json.dumps(dict(settings_snapshot), ensure_ascii=False, sort_keys=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs (id, source_path, source_name, doc_type, status,"
                " settings_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    source_path,
                    source_name,
                    doc_type,
                    JobStatus.QUEUED.value,
                    payload,
                    now,
                    now,
                ),
            )
        return job_id

    def create_job_with_units(
        self,
        *,
        source_path: str,
        source_name: str,
        doc_type: str,
        settings_snapshot: Mapping[str, Any],
        units: Sequence[UnitSeed],
    ) -> str:
        """Create a job and all its units in one transaction."""
        if any("api_key" == key for key in settings_snapshot):
            raise ValueError("settings snapshot must not contain 'api_key'")
        import uuid

        job_id = str(uuid.uuid4())
        now = _utc_now()
        payload = json.dumps(dict(settings_snapshot), ensure_ascii=False, sort_keys=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs (id, source_path, source_name, doc_type, status,"
                " settings_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    source_path,
                    source_name,
                    doc_type,
                    JobStatus.QUEUED.value,
                    payload,
                    now,
                    now,
                ),
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO units (job_id, ordinal, label, content_hash,"
                " start_ref, end_ref, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        job_id,
                        unit.ordinal,
                        unit.label,
                        unit.content_hash,
                        unit.start_ref,
                        unit.end_ref,
                        UnitStatus.PENDING.value,
                    )
                    for unit in units
                ],
            )
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def require_job(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def list_jobs(self, *, limit: int = 100) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def set_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        output_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        JobStatus(status)  # validate
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, _utc_now()]
        if error is not None:
            assignments.append("error = ?")
            params.append(error)
        if output_path is not None:
            assignments.append("output_path = ?")
            params.append(output_path)
        if metadata is not None:
            assignments.append("metadata_json = ?")
            params.append(json.dumps(dict(metadata), ensure_ascii=False))
        params.append(job_id)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", params)

    def acquire_job(self, job_id: str, owner_token: str) -> bool:
        """Atomically claim a job for processing (feature 001 R4)."""
        placeholders = ", ".join("?" for _ in _ACQUIRABLE)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"UPDATE jobs SET status = ?, owner_token = ?, updated_at = ?"
                f" WHERE id = ? AND status IN ({placeholders})",
                (JobStatus.RUNNING.value, owner_token, _utc_now(), job_id, *_ACQUIRABLE),
            )
        return cursor.rowcount == 1

    def release_job(self, job_id: str, owner_token: str, status: str) -> None:
        JobStatus(status)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET status = ?, owner_token = NULL, updated_at = ?"
                " WHERE id = ? AND owner_token = ?",
                (status, _utc_now(), job_id, owner_token),
            )

    def mark_stopping(self, job_id: str) -> bool:
        """Atomically move a *running* job to ``stopping``.

        Returns False when the job is no longer running, so a stop request can
        never overwrite a status the worker has already finalised (e.g. back
        from ``stopped``).
        """
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (JobStatus.STOPPING.value, _utc_now(), job_id, JobStatus.RUNNING.value),
            )
        return cursor.rowcount == 1

    def owner_token(self, job_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT owner_token FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return row["owner_token"] if row else None

    # ------------------------------------------------------------------ #
    # Units
    # ------------------------------------------------------------------ #
    def add_units(self, job_id: str, units: Iterable[UnitSeed]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO units (job_id, ordinal, label, content_hash,"
                " start_ref, end_ref, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        job_id,
                        unit.ordinal,
                        unit.label,
                        unit.content_hash,
                        unit.start_ref,
                        unit.end_ref,
                        UnitStatus.PENDING.value,
                    )
                    for unit in units
                ],
            )

    def list_units(self, job_id: str) -> list[Unit]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM units WHERE job_id = ? ORDER BY ordinal", (job_id,)
            ).fetchall()
        return [_row_to_unit(row) for row in rows]

    def get_unit(self, job_id: str, ordinal: int) -> Unit | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM units WHERE job_id = ? AND ordinal = ?", (job_id, ordinal)
            ).fetchone()
        return _row_to_unit(row) if row else None

    def next_pending_unit(self, job_id: str) -> Unit | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM units WHERE job_id = ? AND status = ? ORDER BY ordinal LIMIT 1",
                (job_id, UnitStatus.PENDING.value),
            ).fetchone()
        return _row_to_unit(row) if row else None

    def mark_unit_running(self, job_id: str, ordinal: int, owner_token: str | None = None) -> None:
        self._assert_owner(job_id, owner_token)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE units SET status = ?, attempts = attempts + 1"
                " WHERE job_id = ? AND ordinal = ?",
                (UnitStatus.RUNNING.value, job_id, ordinal),
            )

    def mark_unit_complete(
        self, job_id: str, ordinal: int, result_path: str, owner_token: str | None = None
    ) -> None:
        self._assert_owner(job_id, owner_token)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE units SET status = ?, result_path = ?, error = NULL"
                " WHERE job_id = ? AND ordinal = ?",
                (UnitStatus.COMPLETE.value, result_path, job_id, ordinal),
            )

    def mark_unit_failed(
        self, job_id: str, ordinal: int, error: str, owner_token: str | None = None
    ) -> None:
        self._assert_owner(job_id, owner_token)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE units SET status = ?, error = ? WHERE job_id = ? AND ordinal = ?",
                (UnitStatus.FAILED.value, error, job_id, ordinal),
            )

    def unit_counts(self, job_id: str) -> dict[str, int]:
        counts = {status.value: 0 for status in UnitStatus}
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM units WHERE job_id = ? GROUP BY status",
                (job_id,),
            ).fetchall()
        total = 0
        for row in rows:
            counts[row["status"]] = int(row["n"])
            total += int(row["n"])
        counts["total"] = total
        return counts

    # ------------------------------------------------------------------ #
    # Results (content-addressed cache)
    # ------------------------------------------------------------------ #
    def find_result(self, content_hash: str, prompt_version: str, model: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM results WHERE content_hash = ? AND prompt_version = ? AND model = ?",
                (content_hash, prompt_version, model),
            ).fetchone()
        return row["summary"] if row else None

    def save_result(self, content_hash: str, prompt_version: str, model: str, summary: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO results"
                " (content_hash, prompt_version, model, summary, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (content_hash, prompt_version, model, summary, _utc_now()),
            )

    def clear_results(self) -> int:
        """Empty the result cache.

        Maintenance operation: forces regeneration of every unit on the next
        run. Returns the number of cached results removed.
        """
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM results")
        return cursor.rowcount

    def save_unit_result(
        self,
        job_id: str,
        ordinal: int,
        *,
        content_hash: str,
        prompt_version: str,
        model: str,
        summary: str,
        result_path: str,
        owner_token: str | None = None,
    ) -> None:
        """Persist the result and flip the unit status in one transaction (FR-011)."""
        self._assert_owner(job_id, owner_token)
        now = _utc_now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO results"
                " (content_hash, prompt_version, model, summary, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (content_hash, prompt_version, model, summary, now),
            )
            self._conn.execute(
                "UPDATE units SET status = ?, result_path = ?, error = NULL"
                " WHERE job_id = ? AND ordinal = ?",
                (UnitStatus.COMPLETE.value, result_path, job_id, ordinal),
            )

    def reset_failed_units(self, job_id: str) -> int:
        """Move failed units back to pending; return how many were reset (FR-007)."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE units SET status = ?, error = NULL WHERE job_id = ? AND status = ?",
                (UnitStatus.PENDING.value, job_id, UnitStatus.FAILED.value),
            )
        return cursor.rowcount

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #
    def put_artifact(
        self,
        job_id: str,
        kind: str,
        status: str,
        *,
        model: str | None = None,
        prompt_version: str | None = None,
        content: str | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts"
                " (job_id, kind, status, model, prompt_version, content, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, kind, status, model, prompt_version, content, _utc_now()),
            )

    def get_artifact(self, job_id: str, kind: str) -> Artifact | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE job_id = ? AND kind = ?", (job_id, kind)
            ).fetchone()
        if row is None:
            return None
        return Artifact(
            job_id=row["job_id"],
            kind=row["kind"],
            status=row["status"],
            model=row["model"],
            prompt_version=row["prompt_version"],
            content=row["content"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------ #
    # Recovery
    # ------------------------------------------------------------------ #
    def orphan_running_jobs(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status IN (?, ?)",
                (JobStatus.RUNNING.value, JobStatus.STOPPING.value),
            ).fetchall()
        return [row["id"] for row in rows]

    def recover_orphans(self) -> int:
        """Treat any running/stopping job as stopped (research R8)."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE jobs SET status = ?, owner_token = NULL, updated_at = ?"
                " WHERE status IN (?, ?)",
                (
                    JobStatus.STOPPED.value,
                    _utc_now(),
                    JobStatus.RUNNING.value,
                    JobStatus.STOPPING.value,
                ),
            )
        return cursor.rowcount

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _assert_owner(self, job_id: str, owner_token: str | None) -> None:
        if owner_token is None:
            return
        with self._lock:
            row = self._conn.execute(
                "SELECT owner_token FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        if row["owner_token"] not in (None, owner_token):
            raise JobOwnershipError(job_id)


# ---------------------------------------------------------------------- #
# Row mapping
# ---------------------------------------------------------------------- #
def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        source_path=row["source_path"],
        source_name=row["source_name"],
        doc_type=DocumentType(row["doc_type"]),
        status=JobStatus(row["status"]),
        settings=json.loads(row["settings_json"] or "{}"),
        output_path=row["output_path"],
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        error=row["error"],
        owner_token=row["owner_token"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_unit(row: sqlite3.Row) -> Unit:
    return Unit(
        id=row["id"],
        job_id=row["job_id"],
        ordinal=row["ordinal"],
        label=row["label"],
        content_hash=row["content_hash"],
        start_ref=row["start_ref"],
        end_ref=row["end_ref"],
        status=UnitStatus(row["status"]),
        attempts=row["attempts"],
        result_path=row["result_path"],
        error=row["error"],
    )
