# AGENTS.md

Guidance for agents and humans working in this repository.

## Verify (run before every commit)

```bash
ruff check . && ruff format --check .
python -m pytest          # offline tests + coverage gate (network marked tests excluded)
```

`python -m pytest` is the canonical command: it reads `pyproject.toml`, deselects
`network`-marked tests, and enforces the coverage floor.

## Layout

- `booksum/` — the pure core. **Must never import `flask` or `web`.**
- `web/` — Flask shell: routes, `JobLogRegistry`, and `web/templates/*.html`.
- `tests/` — `unit/`, `contract/`, `integration/`.
- `evaluation/` — output-quality dataset and checklists.
- `specs/` — Spec Kit artifacts (spec, plan, tasks per feature). **Local only,
  gitignored** — do not commit them, and do not make tests depend on them.

## Hard rules

1. **Never write a template file from code.** `web/templates/` is authored by
   humans; a contract test enforces this.
2. **Never persist or log an API key.** `Settings.to_snapshot()` strips it; keep
   it that way.
3. **The core must stay framework-free.** `tests/contract/test_core_boundary.py`
   imports `booksum` in a subprocess and asserts no `flask`/`web` module loads.
4. **Persist results before starting the next unit.** Durability is the point of
   the store; don't batch writes at the end.
5. **A failed synthesis/coverage pass degrades the artifact, never fails the job.**
6. **Bump a prompt's version when you edit its text** — it is part of the cache
   key.

## Spec-driven workflow

Feature work follows Spec Kit: `/speckit.constitution` → `.specify` →
`spec.md` → `plan.md` → `tasks.md` → implement. These artifacts and the
constitution live in `.specify/` and `specs/`, which are **gitignored** (kept
local, not committed). A published contract that tests depend on must live in
the package (e.g. `booksum/eval/eval_report.schema.json`), never under
`specs/`.

## Environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export GEMINI_API_KEY=...      # only needed for real runs
```

Runtime data (durable store, uploads, artifacts, logs) lives under `runtime/`
and is gitignored.
