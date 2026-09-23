"""Evaluation harness.

Drives the real pipeline (extraction -> per-unit summary -> synthesis ->
coverage) over a dataset and emits a machine-readable report that CI can gate
on. The schema ships with the package at
``booksum/eval/eval_report.schema.json``.

``--offline`` swaps in a deterministic echo client so the harness itself can be
verified without network access (FR-009).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from ..config import Settings
from ..errors import BooksumError, LLMError
from ..extraction import extract_document
from ..llm import LLMClient, LLMResult
from ..prompts import PromptRegistry
from ..summarization import (
    CoverageInput,
    SynthesisInput,
    check_coverage,
    summarize_unit,
    synthesize,
)
from .dataset import EvalDocument, load_checklist, load_dataset, validate_dataset
from .scoring import score_summary


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class OfflineEchoClient:
    """Deterministic client used only for ``--offline`` runs."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def generate(
        self,
        prompt: str,
        *,
        models: Sequence[str],
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
    ) -> LLMResult:
        self.calls += 1
        model = models[0] if models else "offline"
        if "Section summaries:" in prompt:
            body = prompt.split("Section summaries:", 1)[1].strip()
            text = json.dumps(
                {"overview": body[:2000] or "offline", "key_ideas": [], "glossary": []},
                ensure_ascii=False,
            )
        elif "Sections:" in prompt and '"score"' in prompt:
            labels = re.findall(r"^## (.+)$", prompt, flags=re.MULTILINE)
            text = json.dumps(
                {
                    "units": [
                        {"label": label.strip(), "missing": [], "score": 1.0} for label in labels
                    ]
                }
            )
        else:
            text = prompt
        return LLMResult(text=text, model=model, prompt_tokens=0, output_tokens=len(text) // 4)


def _client_call_stats(client: LLMClient) -> tuple[int, int, int]:
    calls_attr = getattr(client, "calls", 0)
    if isinstance(calls_attr, list):
        calls = len(calls_attr)  # FakeLLMClient exposes a list of calls
    else:
        calls = int(calls_attr or 0)
    prompt_tokens = int(
        getattr(client, "total_prompt_tokens", None) or getattr(client, "prompt_tokens", 0) or 0
    )
    output_tokens = int(
        getattr(client, "total_output_tokens", None) or getattr(client, "output_tokens", 0) or 0
    )
    return calls, prompt_tokens, output_tokens


def evaluate_document(
    document: EvalDocument,
    *,
    settings: Settings,
    client: LLMClient,
    repo_root: str,
    workspace: str | None = None,
) -> dict:
    """Run the pipeline for one eval document and score its output."""
    source = (
        document.source
        if os.path.isabs(document.source)
        else os.path.join(repo_root, document.source)
    )
    checklist = (
        document.checklist
        if os.path.isabs(document.checklist)
        else os.path.join(repo_root, document.checklist)
    )
    items = load_checklist(checklist)

    workspace = workspace or tempfile.mkdtemp(prefix="booksum-eval-")
    doc = extract_document(source, settings, image_dir=os.path.join(workspace, "images"))

    prompts = PromptRegistry()
    summaries: list[str] = []
    coverage_inputs: list[CoverageInput] = []
    for unit in doc.units:
        result = summarize_unit(
            client,
            prompts,
            text=unit.text,
            doc_type=doc.doc_type.value,
            doc_filename=doc.name,
            start_ref=unit.start_ref,
            end_ref=unit.end_ref,
            models=settings.models,
            timeout=settings.request_timeout,
        )
        summaries.append(result.text)
        coverage_inputs.append(
            CoverageInput(label=unit.label, source_text=unit.text, summary=result.text)
        )

    summary_text = "\n\n".join(summaries)
    scored = score_summary(summary_text, items)

    synthesis_status = None
    if settings.synthesis_enabled:
        synthesis = synthesize(
            client,
            prompts,
            summaries=[
                SynthesisInput(label=unit.label, summary=result)
                for unit, result in zip(doc.units, summaries, strict=False)
            ],
            doc_title=doc.name,
            models=settings.models,
            timeout=settings.request_timeout,
        )
        synthesis_status = synthesis.status

    coverage_score = None
    if settings.coverage_enabled:
        outcome = check_coverage(client, prompts, units=coverage_inputs, models=settings.models)
        if outcome.report and outcome.status == "complete":
            coverage_score = float(outcome.report.get("score", 0.0))

    calls, prompt_tokens, output_tokens = _client_call_stats(client)
    return {
        "id": document.id,
        "status": "evaluated",
        "captured": scored.captured,
        "total": scored.total,
        "capture_rate": scored.capture_rate,
        "missing": list(scored.missing),
        "coverage_score": coverage_score,
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "message": None
        if synthesis_status in (None, "complete")
        else f"synthesis {synthesis_status}",
    }


def run_evaluation(
    documents: Sequence[EvalDocument],
    *,
    settings: Settings,
    client_factory: Callable[[], LLMClient],
    repo_root: str,
    report_path: str | None = None,
) -> dict:
    results: list[dict] = []
    threshold = settings.coverage_threshold

    for document in documents:
        client = client_factory()
        try:
            results.append(
                evaluate_document(document, settings=settings, client=client, repo_root=repo_root)
            )
        except (BooksumError, LLMError, OSError, ValueError) as exc:
            results.append(
                {
                    "id": document.id,
                    "status": "error",
                    "captured": 0,
                    "total": 0,
                    "capture_rate": 0.0,
                    "missing": [],
                    "coverage_score": None,
                    "calls": 0,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "message": str(exc),
                }
            )

    evaluated = [item for item in results if item["status"] == "evaluated"]
    aggregate = (
        sum(item["capture_rate"] for item in evaluated) / len(evaluated) if evaluated else 0.0
    )

    passed = bool(evaluated)
    for document, item in zip(documents, results, strict=False):
        if item["status"] != "evaluated":
            continue
        floor = document.min_capture if document.min_capture is not None else threshold
        if item["capture_rate"] < floor:
            passed = False
    if aggregate < threshold:
        passed = False

    report = {
        "generated_at": _utc_now(),
        "config": {
            "model": settings.model,
            "prompt_versions": PromptRegistry().versions(),
            "coverage_threshold": threshold,
        },
        "documents": results,
        "aggregate_capture_rate": aggregate,
        "passed": passed,
    }

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the booksum output-quality evaluation.")
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--report", default=None, help="Where to write the JSON report")
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--model", default=Settings().model)
    parser.add_argument("--coverage-threshold", type=float, default=Settings().coverage_threshold)
    parser.add_argument("--offline", action="store_true", help="Use the deterministic echo client")
    parser.add_argument("--validate-dataset", action="store_true", help="Only validate the dataset")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    documents = load_dataset(args.dataset)

    if args.validate_dataset:
        issues = validate_dataset(documents, args.repo_root)
        if issues:
            for issue in issues:
                print(f"INVALID: {issue}", file=sys.stderr)
            return 1
        print(f"OK: {len(documents)} document(s) valid")
        return 0

    settings = Settings(model=args.model, coverage_threshold=args.coverage_threshold)
    if args.offline:
        client_factory: Callable[[], LLMClient] = OfflineEchoClient
    else:  # pragma: no cover - network path
        from ..llm import GeminiRestClient

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("GEMINI_API_KEY is required unless --offline is used", file=sys.stderr)
            return 2
        settings.api_key = api_key
        client_factory = lambda: GeminiRestClient(api_key, max_retries=settings.max_retries)  # noqa: E731

    report = run_evaluation(
        documents,
        settings=settings,
        client_factory=client_factory,
        repo_root=args.repo_root,
        report_path=args.report,
    )
    print(json.dumps({k: report[k] for k in ("aggregate_capture_rate", "passed")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
