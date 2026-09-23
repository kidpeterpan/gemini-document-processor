"""Job orchestration: submit, run, stop, resume.

This is the seam where durability lives. Every unit result is persisted before
the next unit starts, and a restart resumes from persisted state
(``specs/001-durable-pipeline``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .assembly import render_markdown, write_markdown
from .chunking import sha256_text
from .config import Settings
from .errors import BooksumError, LLMCancelled, LLMError
from .extraction import extract_document
from .llm import GeminiRestClient, LLMClient
from .models import (
    ArtifactKind,
    ArtifactStatus,
    DocumentType,
    JobStatus,
    UnitSeed,
    UnitStatus,
)
from .prompts import PromptRegistry
from .queueing import BoundedExecutor
from .store import JobStore
from .summarization import (
    CoverageInput,
    CoverageOutcome,
    SynthesisInput,
    SynthesisOutcome,
    check_coverage,
    summarize_unit,
    synthesize,
)

logger = logging.getLogger("booksum.service")

LogSink = Callable[[str, str], None]
ClientFactory = Callable[[Settings], LLMClient]


class JobService:
    """Facade the web layer (and the eval harness) drives."""

    def __init__(
        self,
        store: JobStore,
        settings: Settings | None = None,
        *,
        client_factory: ClientFactory | None = None,
        executor: BoundedExecutor | None = None,
        prompts: PromptRegistry | None = None,
        log_sink: LogSink | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or Settings()
        self.prompts = prompts or PromptRegistry()
        self._client_factory = client_factory or self._default_client
        self._executor = executor or BoundedExecutor(self.settings.max_concurrency)
        self._owns_executor = executor is None
        self._log_sink = log_sink
        self._api_keys: dict[str, str] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Wiring helpers
    # ------------------------------------------------------------------ #
    def _default_client(self, settings: Settings) -> LLMClient:
        return GeminiRestClient(
            settings.api_key,
            max_retries=settings.max_retries,
            connect_timeout=settings.connect_timeout,
            request_timeout=settings.request_timeout,
            base_retry_delay=settings.base_retry_delay,
        )

    def _log(self, job_id: str, message: str) -> None:
        logger.info("job %s: %s", job_id, message)
        if self._log_sink is not None:
            self._log_sink(job_id, message)

    def _resolve_api_key(self, job_id: str) -> str:
        with self._lock:
            key = self._api_keys.get(job_id)
        return key or self.settings.api_key or os.environ.get("GEMINI_API_KEY", "")

    def _in_memory_key(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._api_keys

    # ------------------------------------------------------------------ #
    # Submission / lifecycle
    # ------------------------------------------------------------------ #
    def submit(
        self,
        source_path: str,
        settings: Settings | None = None,
        *,
        schedule: bool = True,
    ) -> str:
        settings = settings or self.settings
        doc_type = DocumentType.from_path(source_path)
        job_id = self.store.create_job(
            source_path=os.path.abspath(source_path),
            source_name=os.path.basename(source_path),
            doc_type=doc_type.value,
            settings_snapshot=settings.to_snapshot(),
        )
        if settings.api_key:
            with self._lock:
                self._api_keys[job_id] = settings.api_key
        self._log(
            job_id, f"Queued {doc_type.value.upper()} job for {os.path.basename(source_path)}"
        )
        if schedule:
            self._schedule(job_id)
        return job_id

    def _schedule(self, job_id: str) -> None:
        thread = threading.Thread(target=self.run, args=(job_id,), daemon=True)
        thread.start()

    def request_stop(self, job_id: str) -> bool:
        with self._lock:
            event = self._cancels.get(job_id)
        if event is None:
            return False
        event.set()
        # Atomic running -> stopping: never clobber a status the worker has
        # already finalised (see JobStore.mark_stopping).
        self.store.mark_stopping(job_id)
        self._log(job_id, "Stop requested")
        return True

    def cancel_event(self, job_id: str) -> threading.Event | None:
        with self._lock:
            return self._cancels.get(job_id)

    def resume(self, job_id: str, *, schedule: bool = True) -> bool:
        job = self.store.get_job(job_id)
        if job is None:
            raise BooksumError(f"job not found: {job_id}")
        if job.status in (JobStatus.RUNNING, JobStatus.STOPPING):
            return False
        if not self._resolve_api_key(job_id):
            raise BooksumError("api_key_required")
        if schedule:
            self._schedule(job_id)
        return True

    def retry_units(self, job_id: str, *, schedule: bool = True) -> int:
        """Reset failed units to pending and resume (FR-007)."""
        count = self.store.reset_failed_units(job_id)
        if count and schedule:
            self._schedule(job_id)
        return count

    def resume_on_startup(self, *, schedule: bool = True) -> list[str]:
        """Recover orphaned jobs and resume those that can make progress."""
        recovered = self.store.recover_orphans()
        if recovered:
            logger.info("recovered %s orphaned job(s)", recovered)
        resumable: list[str] = []
        for job in self.store.list_jobs():
            if job.status not in (JobStatus.STOPPED, JobStatus.FAILED):
                continue
            counts = self.store.unit_counts(job.id)
            if counts["total"] == 0 or counts["pending"] == 0:
                continue
            if not self._resolve_api_key(job.id):
                self._log(job.id, "Not resuming: no API key available")
                continue
            resumable.append(job.id)
            if schedule:
                self._schedule(job.id)
        return resumable

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #
    def run(self, job_id: str) -> None:
        owner_token = uuid.uuid4().hex
        if not self.store.acquire_job(job_id, owner_token):
            self._log(job_id, "Not started: job is already running or finished")
            return

        cancel = threading.Event()
        with self._lock:
            self._cancels[job_id] = cancel

        job = self.store.require_job(job_id)
        settings = Settings.from_snapshot(job.settings)
        settings.api_key = self._resolve_api_key(job_id)

        usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}
        self._log(job_id, f"Starting {job.doc_type.value.upper()} processing for {job.source_name}")

        try:
            doc = extract_document(
                job.source_path,
                settings,
                image_dir=os.path.join(self.settings.artifacts_dir, job_id, "images"),
            )
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
            self.store.add_units(job_id, seeds)
            counts = self.store.unit_counts(job_id)
            self._log(
                job_id,
                f"Document has {counts['total']} units ({counts['complete']} already complete)",
            )

            client = self._client_factory(settings)
            self._process_units(job_id, doc, settings, client, owner_token, cancel, usage)

            if cancel.is_set():
                self.store.put_artifact(
                    job_id, "usage", ArtifactStatus.COMPLETE.value, content=json.dumps(usage)
                )
                self.store.release_job(job_id, owner_token, JobStatus.STOPPED.value)
                self._log(job_id, "Stopped; completed units were preserved")
                return

            outcomes = self._quality_passes(
                job_id, doc, settings, client, owner_token, cancel, usage
            )
            output_path = self._assemble(job_id, doc, settings, outcomes, usage)

            self.store.put_artifact(
                job_id, "usage", ArtifactStatus.COMPLETE.value, content=json.dumps(usage)
            )
            self.store.set_job_status(
                job_id, JobStatus.RUNNING.value, output_path=output_path, metadata=doc.metadata
            )
            self.store.release_job(job_id, owner_token, JobStatus.COMPLETED.value)
            self._log(job_id, f"Completed: {os.path.basename(output_path)}")

        except LLMCancelled:
            self.store.put_artifact(
                job_id, "usage", ArtifactStatus.COMPLETE.value, content=json.dumps(usage)
            )
            self.store.release_job(job_id, owner_token, JobStatus.STOPPED.value)
            self._log(job_id, "Stopped by request")
        except Exception as exc:  # noqa: BLE001 - job failure must be recorded
            logger.exception("job %s failed", job_id)
            self.store.set_job_status(job_id, JobStatus.FAILED.value, error=str(exc))
            self.store.release_job(job_id, owner_token, JobStatus.FAILED.value)
            self._log(job_id, f"Failed: {exc}")
        finally:
            with self._lock:
                self._cancels.pop(job_id, None)

    # ------------------------------------------------------------------ #
    # Unit processing
    # ------------------------------------------------------------------ #
    def _process_units(
        self,
        job_id: str,
        doc: Any,
        settings: Settings,
        client: LLMClient,
        owner_token: str,
        cancel: threading.Event,
        usage: dict[str, int],
    ) -> None:
        units_by_ordinal = {unit.ordinal: unit for unit in doc.units}
        pending = [u for u in self.store.list_units(job_id) if u.status is UnitStatus.PENDING]

        futures = []
        for unit in pending:
            if cancel.is_set():
                break
            futures.append(
                self._executor.submit(
                    self._process_one_unit,
                    job_id,
                    unit.ordinal,
                    unit.content_hash,
                    units_by_ordinal.get(unit.ordinal),
                    doc.doc_type.value,
                    doc.name,
                    settings,
                    client,
                    owner_token,
                    cancel,
                    usage,
                )
            )

        for future in futures:
            try:
                future.result()
            except LLMCancelled:
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("unit failed for job %s: %s", job_id, exc)

    def _process_one_unit(
        self,
        job_id: str,
        ordinal: int,
        content_hash: str,
        extracted: Any,
        doc_type: str,
        doc_filename: str,
        settings: Settings,
        client: LLMClient,
        owner_token: str,
        cancel: threading.Event,
        usage: dict[str, int],
    ) -> None:
        if cancel.is_set():
            raise LLMCancelled("cancelled before unit start")
        if extracted is None:
            self.store.mark_unit_failed(job_id, ordinal, "no extracted content", owner_token)
            return

        self.store.mark_unit_running(job_id, ordinal, owner_token)
        prompt_version = self.prompts.get("unit").version

        cached = self.store.find_result(content_hash, prompt_version, settings.model)
        if cached is not None:
            summary_text = cached
            self._log(job_id, f"Reused cached summary for unit {ordinal}")
        else:
            try:
                result = summarize_unit(
                    client,
                    self.prompts,
                    text=extracted.text,
                    doc_type=doc_type,
                    doc_filename=doc_filename,
                    start_ref=extracted.start_ref,
                    end_ref=extracted.end_ref,
                    models=settings.models,
                    timeout=settings.request_timeout,
                    cancel=cancel,
                )
            except LLMCancelled:
                raise
            except LLMError as exc:
                self.store.mark_unit_failed(job_id, ordinal, str(exc), owner_token)
                self._log(job_id, f"Unit {ordinal} failed: {exc}")
                return
            summary_text = result.text
            usage["calls"] += 1
            usage["prompt_tokens"] += int(result.prompt_tokens or 0)
            usage["output_tokens"] += int(result.output_tokens or 0)

        result_path = os.path.join(
            self.settings.artifacts_dir, job_id, "units", f"{ordinal:04d}.md"
        )
        write_markdown(result_path, summary_text)
        self.store.save_unit_result(
            job_id,
            ordinal,
            content_hash=content_hash,
            prompt_version=prompt_version,
            model=settings.model,
            summary=summary_text,
            result_path=result_path,
            owner_token=owner_token,
        )

    # ------------------------------------------------------------------ #
    # Quality passes + assembly
    # ------------------------------------------------------------------ #
    def _quality_passes(
        self,
        job_id: str,
        doc: Any,
        settings: Settings,
        client: LLMClient,
        owner_token: str,
        cancel: threading.Event,
        usage: dict[str, int],
    ) -> dict[str, Any]:
        units = self.store.list_units(job_id)
        units_by_ordinal = {unit.ordinal: unit for unit in doc.units}

        summaries: list[SynthesisInput] = []
        coverage_inputs: list[CoverageInput] = []
        for unit in units:
            if unit.status is not UnitStatus.COMPLETE or not unit.result_path:
                continue
            text = _read_text(unit.result_path)
            summaries.append(SynthesisInput(label=unit.label, summary=text))
            extracted = units_by_ordinal.get(unit.ordinal)
            coverage_inputs.append(
                CoverageInput(
                    label=unit.label,
                    source_text=extracted.text if extracted else "",
                    summary=text,
                )
            )

        title = str((doc.metadata or {}).get("title") or doc.name)

        if settings.synthesis_enabled:
            synthesis = synthesize(
                client,
                self.prompts,
                summaries=summaries,
                doc_title=title,
                models=settings.models,
                budget_chars=settings.synthesis_budget_chars,
                timeout=settings.request_timeout,
                cancel=cancel,
            )
        else:
            synthesis = SynthesisOutcome(
                ArtifactStatus.SKIPPED.value,
                None,
                None,
                self.prompts.get("synthesis").version,
                "disabled",
            )
        if synthesis.status == ArtifactStatus.COMPLETE.value:
            usage["calls"] += 1
        self.store.put_artifact(
            job_id,
            ArtifactKind.SYNTHESIS.value,
            synthesis.status,
            model=synthesis.model,
            prompt_version=synthesis.prompt_version,
            content=json.dumps(synthesis.synthesis, ensure_ascii=False)
            if synthesis.synthesis
            else None,
        )
        if synthesis.status != ArtifactStatus.COMPLETE.value:
            self._log(job_id, f"Synthesis {synthesis.status}: {synthesis.error or ''}".strip())

        if settings.coverage_enabled:
            coverage = check_coverage(
                client,
                self.prompts,
                units=coverage_inputs,
                models=settings.models,
                timeout=settings.request_timeout,
                cancel=cancel,
            )
        else:
            coverage = CoverageOutcome(
                ArtifactStatus.SKIPPED.value,
                None,
                None,
                self.prompts.get("coverage").version,
                "disabled",
            )
        if coverage.status == ArtifactStatus.COMPLETE.value:
            usage["calls"] += 1
        self.store.put_artifact(
            job_id,
            ArtifactKind.COVERAGE.value,
            coverage.status,
            model=coverage.model,
            prompt_version=coverage.prompt_version,
            content=json.dumps(coverage.report, ensure_ascii=False) if coverage.report else None,
        )
        if coverage.status != ArtifactStatus.COMPLETE.value:
            self._log(job_id, f"Coverage {coverage.status}: {coverage.error or ''}".strip())

        return {"synthesis": synthesis, "coverage": coverage, "summaries": summaries}

    def _assemble(
        self,
        job_id: str,
        doc: Any,
        settings: Settings,
        outcomes: Mapping[str, Any],
        usage: Mapping[str, int],
    ) -> str:
        units = self.store.list_units(job_id)
        units_by_ordinal = {unit.ordinal: unit for unit in doc.units}

        summaries: OrderedDict[str, str] = OrderedDict()
        images_by_unit: dict[str, list] = {}
        for unit in units:
            if unit.status is UnitStatus.COMPLETE and unit.result_path:
                summaries[unit.label] = _read_text(unit.result_path)
            else:
                summaries[unit.label] = (
                    f"**Error processing {unit.label}:** {unit.error or 'not completed'}"
                )
            extracted = units_by_ordinal.get(unit.ordinal)
            images_by_unit[unit.label] = list(extracted.images) if extracted else []

        synthesis = outcomes["synthesis"]
        coverage = outcomes["coverage"]
        output_path = os.path.join(self.settings.artifacts_dir, job_id, f"{doc.name}_summary.md")

        obsidian_metadata = None
        if settings.use_obsidian:
            obsidian_metadata = {
                "tags": settings.obsidian_tags,
                "author": settings.obsidian_author or str((doc.metadata or {}).get("author", "")),
                "coverurl": settings.obsidian_cover_url,
                "review": settings.obsidian_review,
            }

        content = render_markdown(
            summaries=summaries,
            images_by_unit=images_by_unit,
            doc_name=doc.name,
            doc_type=doc.doc_type.value,
            metadata=doc.metadata,
            obsidian_metadata=obsidian_metadata,
            synthesis=synthesis.synthesis,
            synthesis_status=synthesis.status,
            coverage=coverage.report,
            coverage_status=coverage.status,
            model=settings.model,
            prompt_versions=self.prompts.versions(),
            usage=usage,
            output_path=output_path,
        )
        write_markdown(output_path, content)
        self._export_artifacts(job_id, doc, settings, output_path)
        return output_path

    def _export_artifacts(
        self, job_id: str, doc: Any, settings: Settings, output_path: str
    ) -> None:
        """Copy the rendered document into the flat exports dir for download."""
        try:
            exports = Path(self.settings.exports_dir)
            exports.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, exports / os.path.basename(output_path))
            if doc.image_dir and os.path.isdir(doc.image_dir):
                target_images = exports / f"{doc.name}_images"
                if target_images.exists():
                    shutil.rmtree(target_images)
                shutil.copytree(doc.image_dir, target_images)
        except OSError as exc:
            logger.warning("could not copy artifacts to exports dir: %s", exc)

        if not settings.use_obsidian or not settings.obsidian_vault_path:
            return
        try:
            from .obsidian import export_to_vault

            target = export_to_vault(
                output_path,
                settings.obsidian_vault_path,
                images_dir=doc.image_dir,
            )
            self.store.put_artifact(
                job_id, "obsidian", ArtifactStatus.COMPLETE.value, content=target
            )
            self._log(job_id, f"Exported to Obsidian: {target}")
        except Exception as exc:  # noqa: BLE001 - export must not fail the job
            logger.warning("Obsidian export failed: %s", exc)
            self.store.put_artifact(
                job_id, "obsidian", ArtifactStatus.FAILED.value, content=str(exc)
            )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def usage_totals(self, job_id: str) -> dict[str, int]:
        artifact = self.store.get_artifact(job_id, "usage")
        if artifact and artifact.content:
            try:
                return json.loads(artifact.content)
            except json.JSONDecodeError:
                pass
        return {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}

    def shutdown(self) -> None:
        with self._lock:
            for event in self._cancels.values():
                event.set()
        if self._owns_executor:
            self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------- #
# Small helpers (kept module-level so they are easy to test)
# ---------------------------------------------------------------------- #
def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return ""
