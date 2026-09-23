"""Backward-compatible entry point.

The application was originally a single ``document_gui`` module. It is now a
thin shell over the ``web`` package; this shim keeps ``python document_gui.py``
and existing shortcuts working (feature 003 FR-010). It is intentionally
temporary.
"""

from web.app import create_app, main

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
