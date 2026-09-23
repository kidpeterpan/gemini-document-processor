"""HTTP contract: preserved routes and safe job-lifecycle semantics."""

from __future__ import annotations

import io
import json
import time

from booksum.llm import FakeLLMClient
from booksum.queueing import BoundedExecutor
from booksum.service import JobService
from web.app import create_app


def build_app(tmp_settings, client=None):
    from booksum.store import JobStore

    client = client or FakeLLMClient(responder=lambda prompt, model: "UNIT-SUMMARY")
    store = JobStore(tmp_settings.db_path)
    service = JobService(
        store,
        tmp_settings,
        client_factory=lambda _settings: client,
        executor=BoundedExecutor(1),
    )
    app = create_app(tmp_settings, service)
    app.config.update(TESTING=True)
    return app, service, store


def test_index_renders(tmp_settings):
    app, _, _ = build_app(tmp_settings)
    with app.test_client() as http:
        response = http.get("/")
    assert response.status_code == 200
    assert "gemini-2.0-flash" in response.get_data(as_text=True)


def test_upload_requires_file(tmp_settings):
    app, _, _ = build_app(tmp_settings)
    with app.test_client() as http:
        assert http.post("/upload", data={}).status_code == 400


def test_upload_rejects_wrong_extension(tmp_settings):
    app, _, _ = build_app(tmp_settings)
    with app.test_client() as http:
        response = http.post(
            "/upload",
            data={"file": (io.BytesIO(b"hello"), "notes.txt")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 400


def test_upload_accepts_pdf_and_reports_job(tmp_settings, pdf_path):
    app, service, store = build_app(tmp_settings)
    with open(pdf_path, "rb") as handle:
        payload = handle.read()

    with app.test_client() as http:
        response = http.post(
            "/upload",
            data={"file": (io.BytesIO(payload), "sample.pdf")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    body = response.get_json()
    assert "job_id" in body
    assert body["redirect"].startswith("/job/")


def test_api_job_unknown_is_404(tmp_settings):
    app, _, _ = build_app(tmp_settings)
    with app.test_client() as http:
        assert http.get("/api/job/nope").status_code == 404


def test_api_job_payload_shape_and_no_secrets(tmp_settings):
    app, service, store = build_app(tmp_settings)
    job_id = store.create_job(
        source_path="/tmp/a.pdf",
        source_name="a.pdf",
        doc_type="pdf",
        settings_snapshot={"model": "gemini-2.0-flash"},
    )
    store.acquire_job(job_id, "owner-token")

    with app.test_client() as http:
        response = http.get(f"/api/job/{job_id}")
    assert response.status_code == 200
    body = response.get_json()
    for key in ("job_id", "status", "progress", "unit_counts", "log", "usage", "result_files"):
        assert key in body
    raw = json.dumps(body)
    assert "owner-token" not in raw
    assert "api_key" not in raw


def test_stop_requires_running_job(tmp_settings):
    app, service, store = build_app(tmp_settings)
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    with app.test_client() as http:
        response = http.post(f"/jobs/{job_id}/stop")
    assert response.status_code == 409


def test_resume_unknown_job_is_404(tmp_settings):
    app, _, _ = build_app(tmp_settings)
    with app.test_client() as http:
        assert http.post("/jobs/nope/resume").status_code == 404


def test_resume_running_job_is_409(tmp_settings):
    app, service, store = build_app(tmp_settings)
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    store.acquire_job(job_id, "owner")
    with app.test_client() as http:
        response = http.post(f"/jobs/{job_id}/resume")
    assert response.status_code == 409


def test_legacy_retry_route_delegates(tmp_settings):
    app, service, store = build_app(tmp_settings)
    job_id = store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    with app.test_client() as http:
        response = http.post(f"/retry_chunks/{job_id}")
    assert response.status_code == 200
    assert response.get_json()["retrying"] == 0


def test_obsidian_check_rejects_non_vault(tmp_settings, tmp_path):
    app, _, _ = build_app(tmp_settings)
    with app.test_client() as http:
        response = http.post("/obsidian_check", json={"path": str(tmp_path)})
    assert response.get_json()["valid"] is False


def test_obsidian_check_accepts_vault(tmp_settings, tmp_path):
    (tmp_path / ".obsidian").mkdir()
    app, _, _ = build_app(tmp_settings)
    with app.test_client() as http:
        response = http.post("/obsidian_check", json={"path": str(tmp_path)})
    assert response.get_json()["valid"] is True


def test_jobs_listing(tmp_settings):
    app, service, store = build_app(tmp_settings)
    store.create_job(
        source_path="/tmp/a.pdf", source_name="a.pdf", doc_type="pdf", settings_snapshot={}
    )
    with app.test_client() as http:
        response = http.get("/jobs")
    assert response.status_code == 200
    assert len(response.get_json()["jobs"]) == 1


def test_upload_flow_completes_and_downloads(tmp_settings, pdf_path):
    app, service, store = build_app(tmp_settings)
    with open(pdf_path, "rb") as handle:
        payload = handle.read()

    with app.test_client() as http:
        upload = http.post(
            "/upload",
            data={"file": (io.BytesIO(payload), "sample.pdf")},
            content_type="multipart/form-data",
        )
        job_id = upload.get_json()["job_id"]

        status = {}
        for _ in range(100):
            status = http.get(f"/api/job/{job_id}").get_json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)

    assert status["status"] == "completed"
    assert status["result_files"], "a completed job must expose result files"

    with app.test_client() as http:
        download = http.get(f"/download/{status['result_files'][0]['path']}")
    assert download.status_code == 200
    assert b"## Summary" in download.data
