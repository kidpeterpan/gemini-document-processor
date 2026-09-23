"""Upstream model client.

The production client talks to the Gemini REST API through ``requests`` so that
``timeout=(connect, read)`` actually aborts a hung request. This replaces the
previous thread-join "timeout", which left the request running and leaked a
thread (see ``specs/001-durable-pipeline/research.md`` R2).

The API key is passed in a header and is never included in an exception message.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import requests

from .errors import (
    LLMAuthError,
    LLMCancelled,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger("booksum.llm")

DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        models: Sequence[str],
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
    ) -> LLMResult: ...


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise LLMCancelled("cancelled before request was issued")


class GeminiRestClient:
    """``requests``-based Gemini client with fallback and real timeouts."""

    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        max_retries: int = 3,
        connect_timeout: float = 10.0,
        request_timeout: float = 60.0,
        base_retry_delay: float = 2.0,
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
    ) -> None:
        self._api_key = api_key or ""
        self._session = session or requests.Session()
        self._endpoint = endpoint.rstrip("/")
        self._max_retries = max(1, int(max_retries))
        self._connect_timeout = float(connect_timeout)
        self._request_timeout = float(request_timeout)
        self._base_retry_delay = float(base_retry_delay)
        self._temperature = float(temperature)
        self._max_output_tokens = int(max_output_tokens)

    # ------------------------------------------------------------------ #
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
        if not self._api_key:
            raise LLMAuthError("no API key configured")
        if not models:
            raise LLMError("no models configured")

        read_timeout = float(timeout) if timeout is not None else self._request_timeout
        last_error: LLMError | None = None

        for attempt in range(1, self._max_retries + 1):
            for model in models:
                _check_cancel(cancel)
                try:
                    return self._call_once(
                        model, prompt, temperature, max_output_tokens, read_timeout
                    )
                except (LLMAuthError, LLMTimeoutError, LLMCancelled):
                    # Non-recoverable or already-abandoned: propagate.
                    raise
                except LLMError as exc:
                    last_error = exc
                    logger.warning("attempt %s model %s failed: %s", attempt, model, exc)

            if attempt < self._max_retries:
                delay = self._base_retry_delay * (2 ** (attempt - 1))
                logger.info("retrying in %.1fs", delay)
                time.sleep(delay)

        raise last_error or LLMError("all models failed")

    # ------------------------------------------------------------------ #
    def _call_once(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        read_timeout: float,
    ) -> LLMResult:
        url = f"{self._endpoint}/models/{model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.95,
                "topK": 50,
            },
        }
        try:
            response = self._session.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                json=body,
                timeout=(self._connect_timeout, read_timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"request timed out after {read_timeout:.0f}s for model {model}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            # Never surface the exception's request object (it carries headers).
            raise LLMError(f"connection error for model {model}: {type(exc).__name__}") from None

        status = response.status_code
        if status in (401, 403):
            raise LLMAuthError(f"credentials rejected (HTTP {status})")
        if status == 429:
            raise LLMRateLimitError(f"rate limited for model {model}")
        if status >= 500:
            raise LLMError(f"upstream error (HTTP {status}) for model {model}")
        if status != 200:
            raise LLMError(f"unexpected HTTP {status} for model {model}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"invalid JSON response for model {model}") from exc

        return self._parse(model, data)

    @staticmethod
    def _parse(model: str, data: dict) -> LLMResult:
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"no candidates returned (blockReason={feedback!r})")

        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text = "".join(part.get("text", "") for part in parts)

        usage = data.get("usageMetadata") or {}
        return LLMResult(
            text=text,
            model=model,
            prompt_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
        )


# ---------------------------------------------------------------------- #
# Test double
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakeCall:
    prompt: str
    model: str


@dataclass
class FakeLLMClient:
    """Deterministic in-process client for tests and the offline harness.

    ``responder(prompt, model) -> str`` lets a test script output. Failure
    injection is available per model so fallback and degradation paths can be
    exercised without a network.
    """

    responder: Callable[[str, str], str] | None = None
    fail_models: tuple[str, ...] = ()
    auth_fail_models: tuple[str, ...] = ()
    timeout_models: tuple[str, ...] = ()
    cancel_models: tuple[str, ...] = ()
    block_until_cancel: bool = False
    prompt_tokens: int = 100
    output_tokens: int = 50
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    calls: list[FakeCall] = field(default_factory=list)

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
        _check_cancel(cancel)
        if self.block_until_cancel and cancel is not None:
            # Simulate a hung upstream call that only ends when cancelled.
            cancel.wait(timeout=5.0)
            _check_cancel(cancel)
        last_error: LLMError | None = None
        for model in models:
            _check_cancel(cancel)
            self.calls.append(FakeCall(prompt=prompt, model=model))
            if model in self.auth_fail_models:
                raise LLMAuthError("fake: credentials rejected")
            if model in self.timeout_models:
                raise LLMTimeoutError("fake: timeout")
            if model in self.cancel_models:
                raise LLMCancelled("fake: cancelled")
            if model in self.fail_models:
                last_error = LLMError(f"fake: transient failure on {model}")
                continue
            text = (
                self.responder(prompt, model)
                if self.responder is not None
                else f"[fake summary from {model}]"
            )
            self.total_prompt_tokens += self.prompt_tokens
            self.total_output_tokens += self.output_tokens
            return LLMResult(
                text=text,
                model=model,
                prompt_tokens=self.prompt_tokens,
                output_tokens=self.output_tokens,
            )
        raise last_error or LLMError("fake: all models failed")

    @property
    def call_count(self) -> int:
        return len(self.calls)


def extract_json_object(text: str) -> dict:
    """Best-effort extraction of a single JSON object from model output.

    Models sometimes wrap JSON in ```json fences or add prose; be forgiving.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise LLMError("response did not contain a JSON object")
