# Gemini Document Processor

A durable, resumable tool that turns PDF and EPUB books into detailed Thai
summaries with Google's Gemini models, extracting images and exporting to an
Obsidian vault.

## Why it is built this way

- **Durable work.** A book is hours of sequential model calls. Job state and
  every unit result are persisted in SQLite, so a crash or restart resumes
  exactly where it stopped and never pays for the same content twice.
- **Pure core, thin shell.** `booksum/` is framework-free and fully testable;
  `web/` is a thin Flask shell. Templates are authored files and are never
  written by the application.
- **Measurable output quality.** Output is synthesised whole-book-first, then
  detail-first. A coverage pass reports what might be missing, and an offline
  evaluation harness gates prompt/model changes against a curated fact
  checklist.

The engineering principles are recorded in
[`.specify/memory/constitution.md`](.specify/memory/constitution.md).

## Features

- AI summarization with configurable Gemini models and automatic fallback.
- PDF (page-range chunks) and EPUB (chapter) processing.
- **Whole-document synthesis**: overview, key ideas, and glossary in addition to
  detailed section notes.
- **Omission coverage report**: salient source facts missing from each section's
  summary, with a coverage score.
- Image extraction with size filtering; every embedded link is verified to exist
  before it is written.
- **Resumable jobs**: checkpointed per unit, with stop / resume / retry-failed.
- Bounded concurrency so a burst of uploads cannot overwhelm the provider.
- Obsidian export with YAML frontmatter (tags, author, cover, review).
- Per-job isolated logs (no cross-job interleaving).

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"        # lint + tests
```

Get a Google Gemini API key from <https://aistudio.google.com/>.

## Usage

```bash
export GEMINI_API_KEY="..."     # used for real runs and for resuming after restart
python document_gui.py          # or: python -m web.app
```

Open <http://127.0.0.1:8081/>.

The API key is used for the run only and is **never written to disk**. On
restart, jobs resume using `GEMINI_API_KEY`; if it is unavailable, the job stays
`stopped` with an actionable message.

### Interface

- **Basic**: file, model, chunk size, API key, extract images.
- **Obsidian**: enable export, vault path (validated), tags, author, cover, review.
- **Advanced**: retries, request timeout, max concurrent calls, image thresholds
  and format, synthesis and coverage toggles.
- **Job page**: live progress, unit counts, token/call usage, synthesis and
  coverage status, logs, and stop / resume / retry-failed controls.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — | Credentials (never persisted) |
| `BOOKSUM_MODEL` | `gemini-2.0-flash` | Default model |
| `BOOKSUM_DATA_DIR` | `runtime` | Store, uploads, artifacts, exports, logs |
| `BOOKSUM_CHUNK_SIZE` | `7` | Pages per PDF chunk |
| `BOOKSUM_MAX_CONCURRENCY` | `4` | Global in-flight upstream calls |
| `BOOKSUM_REQUEST_TIMEOUT` | `60` | Per-request read timeout (seconds) |
| `BOOKSUM_COVERAGE_THRESHOLD` | `0.9` | Evaluation gate capture floor |
| `BOOKSUM_HOST` / `BOOKSUM_PORT` | `127.0.0.1` / `8081` | Bind address |
| `BOOKSUM_DEBUG` | unset | Enable Flask debug (opt-in only) |

Copy [`.settings.example.json`](.settings.example.json) to `settings.json` for a
machine-specific vault path (ignored by git).

## How it works

1. **Extract** the document into units (page ranges or chapters) and write image
   bytes to disk.
2. **Map**: summarize each unit, consulting the content-addressed result cache
   first. Results are persisted before the next unit starts.
3. **Reduce**: synthesise a whole-document overview, key ideas, and glossary.
4. **Audit**: report salient facts missing from each unit's summary.
5. **Assemble** Markdown with version/model frontmatter and only resolvable
   image links, then export to the vault if requested.

## Verification

```bash
ruff check . && ruff format --check .
python -m pytest          # offline, with the coverage gate
```

Every check runs offline. Tests that require the network are marked `network`
and deselected by default.

### Output-quality evaluation

```bash
python -m booksum.eval.harness --dataset evaluation/dataset.jsonl --validate-dataset
python -m booksum.eval.harness --dataset evaluation/dataset.jsonl --offline \
    --report /tmp/report.json
```

The report is validated against
[`specs/002-summary-quality/contracts/eval-report.schema.json`](specs/002-summary-quality/contracts/eval-report.schema.json)
and exits non-zero when the capture rate falls below the threshold.

## Project layout

```
booksum/     pure core: extraction, chunking, prompts, llm, store, service,
             summarization, assembly, obsidian, eval
web/         Flask shell: app.py, logging_channel.py, templates/
specs/       Spec Kit artifacts (constitution, spec, plan, tasks per feature)
tests/       unit/, contract/, integration/
evaluation/  output-quality dataset and checklists
document_gui.py   backward-compatible entry point
```

## License

MIT — see [LICENSE](LICENSE).
