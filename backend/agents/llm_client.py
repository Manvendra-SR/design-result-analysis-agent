"""
backend/agents/llm_client.py
==============================
OllamaClient: the single HTTP adapter between the LLM agents (Planner,
Recommender) and a locally-running Ollama server.

Why this is a separate module
-----------------------------
Agent *logic* (prompt construction, response validation, adaptive reasoning
patterns) lives in ``planner.py`` / ``recommender.py``. Everything about
*how* an LLM call is made - the endpoint, the request shape, retry policy,
logging - lives here. Swapping models is a ``.env`` change
(``OLLAMA_MODEL``); swapping providers would be a change to this one file.
No model name is ever hardcoded here (Requirement 15.4).

Transport
---------
Ollama's chat endpoint is ``POST {base_url}/api/chat`` with
``{"model", "messages", "stream": false}`` and an optional ``"format": "json"``
that asks the model to emit strict JSON. The response body is
``{"message": {"role": "assistant", "content": "..."}, "done": true, ...}``.

Retry policy (Requirement 15.5)
-------------------------------
A ``tenacity.Retrying`` controller (built in ``__init__`` so tests can inject
``retry_wait=wait_none()``) retries **transient** failures up to 3 times with
exponential backoff (1s -> 2s -> 4s, capped at 10s):

- ``httpx.TimeoutException``  - server slow / overloaded
- ``httpx.ConnectError``      - server not running yet / wrong URL
- ``httpx.HTTPStatusError`` with status >= 500 - server-side error

A 4xx response is a client bug (bad payload, unknown model) and fails
immediately. Every failure path - transient-exhausted or permanent - is
surfaced to callers as a single exception type, ``LLMError`` (Requirement
15.8 / Non-Functional Reliability 3), the same way ``connection.py`` and
``state_manager.py`` wrap transient database errors.

Logging (Requirement 15.6)
--------------------------
Every call logs the model, message count, and prompt/response character
counts + wall-clock duration at INFO; the full prompt and full response
text at DEBUG.

Requirements
------------
15.1  Ollama HTTP API used for Experiment_Planner_Agent
15.2  Ollama HTTP API used for Recommender_Agent
15.4  Server URL and model name read from config, never hardcoded
15.5  Retry with exponential backoff on timeout / unreachable server
15.6  Log all LLM prompts and responses
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

import backend.config as config  # side-effect: loads .env + configures logging

logger = logging.getLogger(__name__)

_CHAT_PATH = "/api/chat"
_TAGS_PATH = "/api/tags"
_DEFAULT_TIMEOUT_SECONDS = 120.0
_MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    """Raised for any failure talking to the Ollama server.

    Covers connection failures and timeouts that survived all retries, HTTP
    error responses, and malformed response bodies. Callers (the agents, and
    later the FastAPI layer) catch this at their boundary and turn it into a
    user-facing message.
    """


def _is_transient(exc: BaseException) -> bool:
    """True for failures worth retrying (see module docstring)."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


class OllamaClient:
    """HTTP adapter for a local Ollama server.

    Parameters
    ----------
    base_url:
        Ollama server URL. Defaults to ``config.OLLAMA_BASE_URL``.
    model:
        Model tag (e.g. ``"qwen2.5:7b"``). Defaults to ``config.OLLAMA_MODEL``.
        Never hardcoded - passing ``None`` reads the configured value.
    timeout:
        Per-request timeout in seconds (LLM generation can be slow on CPU).
    client:
        Pre-built ``httpx.Client`` (test injection). When provided,
        ``base_url``/``timeout`` are assumed already configured on it.
    retry_wait:
        Override the exponential-backoff wait strategy (tests pass
        ``tenacity.wait_none()`` to keep retry tests fast).
    max_attempts:
        Total attempts per call (default 3).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.Client] = None,
        retry_wait: Optional[wait_base] = None,
        max_attempts: int = _MAX_ATTEMPTS,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

        self._retrying = Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_attempts),
            wait=retry_wait or wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        logger.debug(
            "OllamaClient ready: base_url=%s model=%s timeout=%.0fs",
            self.base_url, self.model, timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat request and return the assistant's message content.

        Parameters
        ----------
        messages:
            OpenAI-style ``[{"role": "system"|"user"|"assistant", "content": ...}]``.
        format:
            Pass ``"json"`` to ask Ollama to constrain output to valid JSON.
        options:
            Ollama generation options (e.g. ``{"temperature": 0.2}``).

        Returns
        -------
        str
            The raw assistant text (callers parse JSON from it themselves via
            ``backend.agents.parsing.parse_json_response``).

        Raises
        ------
        LLMError
            On connection failure / timeout after retries, an HTTP error
            response, or a response body without a message content string.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if format is not None:
            payload["format"] = format
        if options is not None:
            payload["options"] = options

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "LLM call -> model=%s messages=%d prompt_chars=%d",
            self.model, len(messages), prompt_chars,
        )
        for i, m in enumerate(messages):
            logger.debug("LLM prompt[%d] (%s): %s", i, m.get("role"), m.get("content", ""))

        start = time.perf_counter()
        try:
            response = self._retrying(self._post, _CHAT_PATH, payload)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise LLMError(
                f"Ollama returned HTTP {exc.response.status_code} for {self.model!r}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.base_url} after retries: {exc}"
            ) from exc
        duration = time.perf_counter() - start

        content = self._extract_content(response)
        logger.info(
            "LLM response <- model=%s response_chars=%d duration=%.2fs",
            self.model, len(content), duration,
        )
        logger.debug("LLM response text: %s", content)
        return content

    def health_check(self) -> bool:
        """Return True if the Ollama server answers ``/api/tags``.

        Never raises - intended for scripts/checkpoints that skip live
        checks when no server is running.
        """
        try:
            resp = self._client.get(_TAGS_PATH)
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Ollama health check failed (%s): %s", self.base_url, exc)
            return False

    def close(self) -> None:
        """Close the underlying HTTP client (only if this instance created it)."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One HTTP attempt. Raises httpx errors (classified by ``_is_transient``)."""
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_content(body: Dict[str, Any]) -> str:
        """Pull ``message.content`` out of an Ollama chat response body."""
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMError(
                f"Unexpected Ollama response shape (no message.content): {str(body)[:300]}"
            )
        return message["content"]
