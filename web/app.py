"""Flask application shell.

Routes only. All domain logic lives in ``booksum`` (feature 003 contract
``core-boundary.md``). Templates are authored files under ``web/templates`` and
are never written by the application.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from booksum.config import MODEL_CHOICES, Settings
from booksum.obsidian import is_vault
from booksum.service import JobService
from booksum.store import JobStore

from .logging_channel import JobLogRegistry

logger = logging.getLogger("booksum.web")

SETTINGS_FILENAME = "settings.json"


def settings_file_path() -> Path:
    return Path(os.getcwd()) / SETTINGS_FILENAME


def load_saved_settings() -> dict:
    path = settings_file_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}


def save_settings(values: dict) -> None:
    existing = load_saved_settings()
    existing.update(values)
    try:
        with open(settings_file_path(), "w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.warning("could not write settings: %s", exc)


def create_app(
    settings: Settings | None = None,
    service: JobService | None = None,
    log_registry: JobLogRegistry | None = None,
) -> Flask:
    settings = settings or Settings.from_env()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    registry = log_registry or JobLogRegistry()
    job_service = service or JobService(
        JobStore(settings.db_path), settings, log_sink=registry.sink
    )

    app = Flask(__name__, template_folder="templates")
    app.config["BOOKSUM_SETTINGS"] = settings
    app.config["BOOKSUM_SERVICE"] = job_service
    app.config["BOOKSUM_LOGS"] = registry

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #
    @app.route("/")
    def index():
        saved = load_saved_settings()
        vault = saved.get("obsidian_vault_path") or settings.obsidian_vault_path
        return render_template(
            "index.html",
            obsidian_dir=vault,
            models=MODEL_CHOICES,
            default_model=settings.model,
            chunk_size=settings.chunk_size,
        )

    @app.route("/job/<job_id>")
    def job_status(job_id: str):
        if job_service.store.get_job(job_id) is None:
            return "Job not found", 404
        return render_template("job_status.html", job_id=job_id)

    # ------------------------------------------------------------------ #
    # Upload
    # ------------------------------------------------------------------ #
    @app.route("/upload", methods=["POST"])
    def upload_file():
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        upload = request.files["file"]
        if not upload.filename:
            return jsonify({"error": "No selected file"}), 400
        if not (
            upload.filename.lower().endswith(".pdf") or upload.filename.lower().endswith(".epub")
        ):
            return jsonify({"error": "File must be a PDF or EPUB"}), 400

        values = {key: request.form.get(key) for key in request.form}
        values["obsidian_vault_path"] = (
            values.get("obsidian_vault_path") or settings.obsidian_vault_path
        )
        try:
            run_settings = Settings.from_mapping(values)
        except ValueError as exc:
            return jsonify({"error": f"invalid settings: {exc}"}), 400

        if run_settings.obsidian_vault_path:
            save_settings({"obsidian_vault_path": run_settings.obsidian_vault_path})

        safe_name = os.path.basename(upload.filename)
        stored = settings.uploads_dir / f"{uuid.uuid4().hex}_{safe_name}"
        upload.save(stored)

        try:
            job_id = job_service.submit(str(stored), run_settings)
        except Exception as exc:  # noqa: BLE001
            logger.exception("could not submit job")
            return jsonify({"error": str(exc)}), 500

        return jsonify({"job_id": job_id, "redirect": url_for("job_status", job_id=job_id)})

    # ------------------------------------------------------------------ #
    # Job API
    # ------------------------------------------------------------------ #
    @app.route("/api/job/<job_id>")
    def api_job_status(job_id: str):
        job = job_service.store.get_job(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(_job_payload(job_id, job_service, registry))

    @app.route("/jobs")
    def list_jobs():
        return jsonify({"jobs": [_job_summary(job) for job in job_service.store.list_jobs()]})

    @app.route("/jobs/<job_id>/stop", methods=["POST"])
    def stop_job(job_id: str):
        job = job_service.store.get_job(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        if not job_service.request_stop(job_id):
            return jsonify({"error": "job is not stoppable", "status": job.status.value}), 409
        return jsonify({"job_id": job_id, "status": "stopping"})

    @app.route("/jobs/<job_id>/resume", methods=["POST"])
    def resume_job(job_id: str):
        job = job_service.store.get_job(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        try:
            started = job_service.resume(job_id)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 409
        if not started:
            return jsonify({"error": "job is already running", "status": job.status.value}), 409
        return jsonify({"job_id": job_id, "status": "running"})

    @app.route("/jobs/<job_id>/retry_units", methods=["POST"])
    def retry_units(job_id: str):
        job = job_service.store.get_job(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        count = job_service.retry_units(job_id)
        return jsonify({"job_id": job_id, "retrying": count})

    @app.route("/retry_chunks/<job_id>", methods=["POST"])
    def retry_chunks_legacy(job_id: str):
        return retry_units(job_id)

    # ------------------------------------------------------------------ #
    # Files
    # ------------------------------------------------------------------ #
    @app.route("/download/<path:filename>")
    def download_file(filename: str):
        return send_from_directory(str(settings.exports_dir), filename, as_attachment=True)

    @app.route("/view/<path:filename>")
    def view_file(filename: str):
        return send_from_directory(str(settings.exports_dir), filename)

    # ------------------------------------------------------------------ #
    # Obsidian
    # ------------------------------------------------------------------ #
    @app.route("/obsidian_check", methods=["POST"])
    def obsidian_check():
        payload = request.get_json(silent=True) or {}
        path = str(payload.get("path", ""))
        if not path:
            return jsonify({"valid": False, "message": "Path is empty"})
        if not os.path.exists(path):
            return jsonify({"valid": False, "message": "Path does not exist"})
        if not os.path.isdir(path):
            return jsonify({"valid": False, "message": "Path is not a directory"})
        if not is_vault(path):
            return jsonify(
                {"valid": False, "message": "Not an Obsidian vault (no .obsidian folder)"}
            )
        return jsonify({"valid": True, "message": "Valid Obsidian vault"})

    # ------------------------------------------------------------------ #
    # Startup recovery
    # ------------------------------------------------------------------ #
    job_service.resume_on_startup(schedule=False)

    return app


def _job_summary(job) -> dict:
    return {
        "job_id": job.id,
        "status": job.status.value,
        "source_name": job.source_name,
        "created_at": job.created_at,
    }


def _job_payload(job_id: str, service: JobService, registry: JobLogRegistry) -> dict:
    job = service.store.require_job(job_id)
    counts = service.store.unit_counts(job_id)
    total = counts.get("total", 0)
    complete = counts.get("complete", 0)
    progress = int(round((complete / total) * 100)) if total else 0
    if job.status.value == "completed":
        progress = 100

    result_files: list[dict] = []
    if job.output_path:
        name = os.path.basename(job.output_path)
        result_files.append({"type": "file", "name": name, "path": name})
        images_name = f"{os.path.splitext(job.source_name)[0]}_images"
        if (Path(service.settings.exports_dir) / images_name).is_dir():
            result_files.append({"type": "directory", "name": images_name, "path": images_name})

    obsidian = service.store.get_artifact(job_id, "obsidian")
    if obsidian and obsidian.content:
        result_files.append(
            {
                "type": "obsidian",
                "name": f"Obsidian: {os.path.basename(obsidian.content)}",
                "path": obsidian.content,
            }
        )

    failed = counts.get("failed", 0)
    synthesis = service.store.get_artifact(job_id, "synthesis")
    coverage = service.store.get_artifact(job_id, "coverage")
    return {
        "job_id": job_id,
        "status": job.status.value,
        "progress": progress,
        "message": registry.tail(job_id, 1)[0] if registry.tail(job_id, 1) else "Waiting…",
        "unit_counts": counts,
        "log": registry.tail(job_id, 15),
        "result_files": result_files,
        "error": job.error,
        "document_type": job.doc_type.value,
        "document_metadata": job.metadata,
        "failed_chunks": failed,
        "usage": service.usage_totals(job_id),
        "synthesis_status": synthesis.status if synthesis else None,
        "coverage_status": coverage.status if coverage else None,
    }


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    logger.info("starting on http://%s:%s (debug=%s)", settings.host, settings.port, settings.debug)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
