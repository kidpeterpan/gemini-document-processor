"""Typed exception hierarchy shared across the core."""

from __future__ import annotations


class BooksumError(Exception):
    """Base class for all errors raised by the core."""


class ExtractionError(BooksumError):
    """Raised when a document cannot be read or parsed."""


class StoreSchemaError(BooksumError):
    """Raised when the persisted schema is newer than this code supports."""


class JobNotFoundError(BooksumError):
    """Raised when a job id does not exist."""


class JobOwnershipError(BooksumError):
    """Raised when a caller writes to a job it does not own."""


class LLMError(BooksumError):
    """Base class for upstream model errors."""


class LLMTimeoutError(LLMError):
    """The upstream call exceeded the configured timeout and was abandoned."""


class LLMAuthError(LLMError):
    """Credentials were rejected. Not retryable, no model fallback."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited the request after retries were exhausted."""


class LLMCancelled(LLMError):
    """The caller requested cancellation before a request was issued."""
