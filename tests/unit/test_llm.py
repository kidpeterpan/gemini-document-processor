"""Upstream client: parsing, fallback, timeout, cancellation, secret hygiene."""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

import pytest
import requests

from booksum.errors import (
    LLMAuthError,
    LLMCancelled,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from booksum.llm import FakeLLMClient, GeminiRestClient, LLMResult, extract_json_object


class FakeResponse:
    def __init__(
        self, status_code: int, payload: dict | None = None, *, invalid_json: bool = False
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Records posts and returns scripted responses or raises."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def success_payload(text="hello"):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
    }


def make_client(session, **overrides: Any):
    kwargs: dict[str, Any] = dict(
        api_key="key-123",
        session=session,
        max_retries=2,
        base_retry_delay=0.0,
        request_timeout=30,
        connect_timeout=5,
    )
    kwargs.update(overrides)
    return GeminiRestClient(**kwargs)


def test_success_parses_text_and_usage():
    session = FakeSession([FakeResponse(200, success_payload("summary text"))])
    result = make_client(session).generate("prompt", models=["m1"])
    assert result.text == "summary text"
    assert result.model == "m1"
    assert result.prompt_tokens == 11
    assert result.output_tokens == 7


def test_timeout_tuple_is_passed_to_requests():
    session = FakeSession([FakeResponse(200, success_payload())])
    make_client(session).generate("prompt", models=["m1"], timeout=42)
    assert session.calls[0]["timeout"] == (5, 42)


def test_falls_back_to_next_model_on_transient_error():
    session = FakeSession([FakeResponse(500), FakeResponse(200, success_payload("ok"))])
    result = make_client(session).generate("prompt", models=["primary", "fallback"])
    assert result.model == "fallback"
    assert result.text == "ok"


def test_rate_limit_after_all_models_fails():
    session = FakeSession(
        [FakeResponse(429), FakeResponse(429), FakeResponse(429), FakeResponse(429)]
    )
    with pytest.raises((LLMRateLimitError, LLMError)):
        make_client(session).generate("prompt", models=["m1", "m2"])


def test_auth_error_is_not_retried_and_has_no_key():
    session = FakeSession([FakeResponse(401)])
    client = make_client(session, max_retries=5)
    with pytest.raises(LLMAuthError) as excinfo:
        client.generate("prompt", models=["m1", "m2"])
    assert "key-123" not in str(excinfo.value)
    assert len(session.calls) == 1  # no retry, no fallback


def test_read_timeout_raises_and_does_not_retry():
    session = FakeSession([requests.exceptions.Timeout("read timed out")])
    with pytest.raises(LLMTimeoutError):
        make_client(session).generate("prompt", models=["m1"])
    assert len(session.calls) == 1


def test_connection_error_is_retried_then_succeeds():
    session = FakeSession(
        [requests.exceptions.ConnectionError("boom"), FakeResponse(200, success_payload("ok"))]
    )
    result = make_client(session).generate("prompt", models=["m1"])
    assert result.text == "ok"
    assert len(session.calls) == 2


def test_invalid_json_becomes_llm_error():
    session = FakeSession(
        [FakeResponse(200, invalid_json=True), FakeResponse(200, invalid_json=True)]
    )
    with pytest.raises(LLMError):
        make_client(session).generate("prompt", models=["m1"])


def test_no_candidates_becomes_llm_error():
    session = FakeSession(
        [
            FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}}),
            FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}}),
        ]
    )
    with pytest.raises(LLMError):
        make_client(session).generate("prompt", models=["m1"])


def test_cancelled_before_request_makes_no_call():
    session = FakeSession([FakeResponse(200, success_payload())])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(LLMCancelled):
        make_client(session).generate("prompt", models=["m1"], cancel=cancel)
    assert session.calls == []


def test_missing_api_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        make_client(FakeSession([]), api_key="").generate("prompt", models=["m1"])


def test_no_models_is_an_error():
    with pytest.raises(LLMError):
        make_client(FakeSession([])).generate("prompt", models=[])


# ---------------------------------------------------------------------- #
# Fake client
# ---------------------------------------------------------------------- #
def test_fake_client_records_calls_and_falls_back():
    client = FakeLLMClient(fail_models=("primary",))
    result = client.generate("p", models=["primary", "secondary"])
    assert result.model == "secondary"
    assert [call.model for call in client.calls] == ["primary", "secondary"]
    assert client.call_count == 2


def test_fake_client_auth_failure_is_immediate():
    client = FakeLLMClient(auth_fail_models=("primary",))
    with pytest.raises(LLMAuthError):
        client.generate("p", models=["primary", "secondary"])


def test_fake_client_blocks_until_cancelled():
    client = FakeLLMClient(block_until_cancel=True)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(LLMCancelled):
        client.generate("p", models=["m"], cancel=cancel)


# ---------------------------------------------------------------------- #
# JSON extraction
# ---------------------------------------------------------------------- #
def test_extract_json_plain():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    assert extract_json_object('Sure! {"a": 1} done') == {"a": 1}


def test_extract_json_failure():
    with pytest.raises(LLMError):
        extract_json_object("no json here")


def test_llm_result_is_frozen():
    result = LLMResult(text="x", model="m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.text = "y"  # type: ignore[misc]
