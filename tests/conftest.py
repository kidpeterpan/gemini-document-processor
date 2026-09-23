"""Shared pytest fixtures.

Everything here is offline: no fixture reaches the network (Constitution III).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from booksum.config import Settings
from booksum.llm import FakeLLMClient
from booksum.queueing import BoundedExecutor
from booksum.service import JobService
from booksum.store import JobStore

from .fixtures import make_epub, make_pdf, make_png_bytes


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a throwaway data dir, with fast retries."""
    return Settings(
        data_dir=tmp_path / "runtime",
        chunk_size=1,
        max_concurrency=1,
        request_timeout=5,
        base_retry_delay=0.01,
        model="gemini-2.0-flash",
    )


@pytest.fixture
def store(tmp_settings: Settings) -> JobStore:
    return JobStore(tmp_settings.db_path)


@pytest.fixture
def fake_client() -> FakeLLMClient:
    return FakeLLMClient()


def make_service(
    store: JobStore,
    settings: Settings,
    client: FakeLLMClient | None = None,
) -> JobService:
    client = client or FakeLLMClient()
    return JobService(
        store,
        settings,
        client_factory=lambda _settings: client,
        executor=BoundedExecutor(settings.max_concurrency),
    )


@pytest.fixture
def service(store: JobStore, tmp_settings: Settings) -> JobService:
    return make_service(store, tmp_settings)


@pytest.fixture
def pdf_path(tmp_path: Path) -> str:
    return make_pdf(
        tmp_path / "sample.pdf",
        [
            "Alpha fact: relativity explains gravity as geometry.",
            "Beta fact: the speed of light is constant for all observers.",
            "Gamma fact: mass and energy are equivalent.",
        ],
    )


@pytest.fixture
def epub_path(tmp_path: Path) -> str:
    return make_epub(
        tmp_path / "sample.epub",
        [
            ("Chapter One", "Quantum mechanics describes nature at small scales."),
            ("Chapter Two", "Entanglement links particles across distance."),
        ],
        image_name="figure.png",
        image_bytes=make_png_bytes(),
    )
