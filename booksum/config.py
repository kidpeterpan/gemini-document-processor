"""Runtime configuration.

Settings are the single place model identifiers, timeouts, prompt versions and
thresholds live. Nothing in the pipeline may hardcode a model id (feature 002
FR-006).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemini-2.0-flash"

# Ordered model preference. The first entry is the default; the rest are
# fallbacks tried on retryable failures. Selectable from the UI.
MODEL_CHOICES: tuple[str, ...] = (
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-pro",
)

DEFAULT_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-1.5-pro",
)

DEFAULT_PROMPT_VERSIONS: dict[str, str] = {
    "unit": "unit.v1",
    "synthesis": "synth.v1",
    "coverage": "coverage.v1",
}

_API_KEY_FIELD = "api_key"


@dataclass
class Settings:
    """All tunable behaviour for one processing run."""

    # Credentials — never persisted (Constitution V / feature 001 FR-012).
    api_key: str = ""

    # Model selection
    model: str = DEFAULT_MODEL
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS

    # Chunking
    chunk_size: int = 7

    # Upstream client
    max_retries: int = 3
    request_timeout: float = 60.0
    connect_timeout: float = 10.0
    base_retry_delay: float = 2.0

    # Images
    extract_images: bool = True
    min_img_width: int = 100
    min_img_height: int = 100
    img_format: str = "png"

    # Concurrency
    max_concurrency: int = 4

    # Quality passes (feature 002)
    synthesis_enabled: bool = True
    coverage_enabled: bool = True
    synthesis_budget_chars: int = 200_000
    coverage_threshold: float = 0.9
    per_doc_floor: float | None = None
    prompt_versions: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPT_VERSIONS))

    # Storage
    data_dir: Path = Path("runtime")

    # Obsidian export
    obsidian_vault_path: str = ""
    use_obsidian: bool = False
    obsidian_tags: str = "book,main"
    obsidian_author: str = ""
    obsidian_cover_url: str = ""
    obsidian_review: str = ""

    # Server
    host: str = "127.0.0.1"
    port: int = 8081
    debug: bool = False

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #
    @property
    def models(self) -> tuple[str, ...]:
        """Default model followed by any distinct fallbacks."""
        ordered: list[str] = []
        for name in (self.model, *self.fallback_models):
            if name and name not in ordered:
                ordered.append(name)
        return tuple(ordered)

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "booksum.db"

    @property
    def uploads_dir(self) -> Path:
        return Path(self.data_dir) / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.data_dir) / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return Path(self.data_dir) / "logs"

    @property
    def exports_dir(self) -> Path:
        return Path(self.data_dir) / "results"

    @property
    def effective_per_doc_floor(self) -> float:
        return self.per_doc_floor if self.per_doc_floor is not None else self.coverage_threshold

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot with no secrets.

        The API key is deliberately excluded so it can never be written to the
        store (feature 001 FR-012).
        """
        data = asdict(self)
        data.pop(_API_KEY_FIELD, None)
        data["data_dir"] = str(self.data_dir)
        data["fallback_models"] = list(self.fallback_models)
        data["models"] = list(self.models)
        return data

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> Settings:
        """Rebuild settings from a stored snapshot (no api_key present)."""
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in snapshot.items() if k in allowed and k != _API_KEY_FIELD}
        if "data_dir" in filtered:
            filtered["data_dir"] = Path(filtered["data_dir"])
        if "fallback_models" in filtered and filtered["fallback_models"] is not None:
            filtered["fallback_models"] = tuple(filtered["fallback_models"])
        return cls(**filtered)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> Settings:
        """Build from loosely-typed input (e.g. an HTTP form)."""

        def _bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "on", "yes"}

        def _int(value: Any, default: int) -> int:
            try:
                return int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        def _float(value: Any, default: float) -> float:
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        src = mapping or {}
        base = cls()
        return cls(
            api_key=str(src.get("api_key", "") or ""),
            model=str(src.get("model") or src.get("model_name") or base.model),
            fallback_models=base.fallback_models,
            chunk_size=_int(src.get("chunk_size"), base.chunk_size),
            max_retries=_int(src.get("max_retries"), base.max_retries),
            request_timeout=_float(
                src.get("request_timeout", src.get("api_timeout")), base.request_timeout
            ),
            connect_timeout=_float(src.get("connect_timeout"), base.connect_timeout),
            base_retry_delay=_float(src.get("base_retry_delay"), base.base_retry_delay),
            extract_images=_bool(src.get("extract_images"), base.extract_images),
            min_img_width=_int(src.get("min_img_width"), base.min_img_width),
            min_img_height=_int(src.get("min_img_height"), base.min_img_height),
            img_format=str(src.get("img_format") or base.img_format),
            max_concurrency=_int(
                src.get("max_concurrency", src.get("max_workers")), base.max_concurrency
            ),
            synthesis_enabled=_bool(src.get("synthesis_enabled"), base.synthesis_enabled),
            coverage_enabled=_bool(src.get("coverage_enabled"), base.coverage_enabled),
            synthesis_budget_chars=_int(
                src.get("synthesis_budget_chars"), base.synthesis_budget_chars
            ),
            coverage_threshold=_float(src.get("coverage_threshold"), base.coverage_threshold),
            obsidian_vault_path=str(src.get("obsidian_vault_path") or base.obsidian_vault_path),
            use_obsidian=_bool(src.get("use_obsidian"), base.use_obsidian),
            obsidian_tags=str(src.get("obsidian_tags") or base.obsidian_tags),
            obsidian_author=str(src.get("obsidian_author") or base.obsidian_author),
            obsidian_cover_url=str(src.get("obsidian_cover_url") or base.obsidian_cover_url),
            obsidian_review=str(src.get("obsidian_review") or base.obsidian_review),
            debug=_bool(src.get("debug"), base.debug),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build from ``BOOKSUM_*`` environment variables."""
        e = env if env is not None else os.environ
        base = cls()

        def _env(key: str) -> str | None:
            value = e.get(key)
            return value if value not in (None, "") else None

        data_dir = _env("BOOKSUM_DATA_DIR")
        return cls(
            api_key=_env("GEMINI_API_KEY") or "",
            model=_env("BOOKSUM_MODEL") or base.model,
            chunk_size=int(_env("BOOKSUM_CHUNK_SIZE") or base.chunk_size),
            max_retries=int(_env("BOOKSUM_MAX_RETRIES") or base.max_retries),
            request_timeout=float(_env("BOOKSUM_REQUEST_TIMEOUT") or base.request_timeout),
            data_dir=Path(data_dir) if data_dir else base.data_dir,
            max_concurrency=int(_env("BOOKSUM_MAX_CONCURRENCY") or base.max_concurrency),
            extract_images=(_env("BOOKSUM_EXTRACT_IMAGES") or "1").lower()
            in {"1", "true", "on", "yes"},
            coverage_threshold=float(_env("BOOKSUM_COVERAGE_THRESHOLD") or base.coverage_threshold),
            host=_env("BOOKSUM_HOST") or base.host,
            port=int(_env("BOOKSUM_PORT") or base.port),
            debug=(_env("BOOKSUM_DEBUG") or "").lower() in {"1", "true", "on", "yes"},
            obsidian_vault_path=_env("BOOKSUM_OBSIDIAN_VAULT") or base.obsidian_vault_path,
        )
