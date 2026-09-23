"""Flask shell package. Depends on ``booksum``; ``booksum`` must not import this."""

from .app import create_app, main

__all__ = ["create_app", "main"]
