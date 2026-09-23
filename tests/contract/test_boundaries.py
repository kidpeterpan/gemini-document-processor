"""Architectural boundaries: pure core, and templates as authored artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "web" / "templates"


def test_core_imports_no_web_framework():
    code = (
        "import sys, importlib;"
        "importlib.import_module('booksum');"
        "importlib.import_module('booksum.eval');"
        "bad = sorted(m for m in sys.modules"
        " if m == 'flask' or m.startswith('web'));"
        "print(bad)"
    )
    output = subprocess.check_output([sys.executable, "-c", code], cwd=REPO_ROOT, text=True)
    assert output.strip() == "[]", f"core imported the web layer: {output}"


def test_templates_exist_as_authored_files():
    names = {path.name for path in TEMPLATES_DIR.glob("*.html")}
    assert {"index.html", "job_status.html"} <= names


def test_create_app_does_not_write_templates(tmp_settings):
    from web.app import create_app

    before = {path.name: path.read_bytes() for path in TEMPLATES_DIR.glob("*.html")}
    before_mtimes = {path.name: path.stat().st_mtime_ns for path in TEMPLATES_DIR.glob("*.html")}

    app = create_app(tmp_settings)
    with app.test_client() as client:
        assert client.get("/").status_code == 200

    after = {path.name: path.read_bytes() for path in TEMPLATES_DIR.glob("*.html")}
    after_mtimes = {path.name: path.stat().st_mtime_ns for path in TEMPLATES_DIR.glob("*.html")}

    assert before == after, "templates were modified by the application"
    assert before_mtimes == after_mtimes, "templates were rewritten by the application"


def test_source_does_not_open_templates_for_writing():
    """No module may open a template file in a write mode."""
    offenders = []
    for path in (REPO_ROOT / "booksum").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "templates" in text and ('"w"' in text or "'w'" in text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"modules that may write templates: {offenders}"
