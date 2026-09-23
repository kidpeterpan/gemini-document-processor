"""Access to the published evaluation-report JSON Schema.

The schema ships with the package (``eval_report.schema.json``) so it is
available wherever the code runs — including CI, where the local Spec Kit
artifacts under ``specs/`` are not present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("eval_report.schema.json")


def load_schema() -> dict[str, Any]:
    """Return the evaluation-report JSON Schema as a dict."""
    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        return json.load(handle)
