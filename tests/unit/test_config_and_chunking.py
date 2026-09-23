"""Configuration, snapshot/secret handling, and chunking math."""

from __future__ import annotations

from pathlib import Path

from booksum.chunking import page_ranges, sha256_text
from booksum.config import Settings


def test_defaults_are_safe():
    settings = Settings.from_env({})
    assert settings.host == "127.0.0.1"
    assert settings.debug is False


def test_snapshot_omits_api_key():
    settings = Settings(api_key="super-secret")
    snapshot = settings.to_snapshot()
    assert "api_key" not in snapshot
    assert "super-secret" not in str(snapshot)


def test_from_mapping_parses_form_values():
    settings = Settings.from_mapping(
        {
            "model_name": "gemini-2.5-pro",
            "chunk_size": "5",
            "api_key": "abc",
            "extract_images": "on",
            "api_timeout": "90",
            "max_workers": "8",
            "use_obsidian": "on",
            "obsidian_tags": "book,main",
        }
    )
    assert settings.model == "gemini-2.5-pro"
    assert settings.chunk_size == 5
    assert settings.api_key == "abc"
    assert settings.extract_images is True
    assert settings.request_timeout == 90
    assert settings.max_concurrency == 8
    assert settings.use_obsidian is True


def test_from_mapping_tolerates_garbage_numbers():
    settings = Settings.from_mapping({"chunk_size": "not-a-number", "api_timeout": ""})
    assert settings.chunk_size == Settings().chunk_size
    assert settings.request_timeout == Settings().request_timeout


def test_from_snapshot_roundtrips_without_key():
    original = Settings(model="gemini-1.5-pro", chunk_size=3, data_dir=Path("/tmp/x"))
    restored = Settings.from_snapshot(original.to_snapshot())
    assert restored.model == "gemini-1.5-pro"
    assert restored.chunk_size == 3
    assert restored.data_dir == Path("/tmp/x")
    assert restored.api_key == ""


def test_models_deduplicates_and_orders():
    settings = Settings(model="m1", fallback_models=("m1", "m2", "m2", "m3"))
    assert settings.models == ("m1", "m2", "m3")


def test_derived_paths():
    settings = Settings(data_dir=Path("/data"))
    assert settings.db_path == Path("/data/booksum.db")
    assert settings.exports_dir == Path("/data/results")


def test_effective_per_doc_floor_falls_back_to_threshold():
    assert Settings(coverage_threshold=0.8).effective_per_doc_floor == 0.8
    assert Settings(coverage_threshold=0.8, per_doc_floor=0.5).effective_per_doc_floor == 0.5


def test_sha256_is_deterministic_and_content_sensitive():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("hello ")
    assert len(sha256_text("hello")) == 64


def test_page_ranges_covers_all_pages_once():
    ranges = list(page_ranges(10, 4))
    assert ranges == [(1, 4), (5, 8), (9, 10)]
    covered = [page for start, end in ranges for page in range(start, end + 1)]
    assert covered == list(range(1, 11))


def test_page_ranges_handles_empty_and_invalid():
    assert list(page_ranges(0, 5)) == []
    try:
        list(page_ranges(5, 0))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for chunk_size=0")
