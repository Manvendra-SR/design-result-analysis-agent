"""
backend/agents/llm.py
=======================
The one LLM integration: Groq's OpenAI-compatible chat API with **strict
JSON-schema structured output**, so a reply always has the requested shape.

``request_json`` adds the one thing a schema cannot express - business rules
(a level out of range, a factor that does not exist). The caller's ``build``
function validates the parsed reply; if it raises ``ValueError`` the error is
sent back to the model once to fix, then the call fails with ``LLMError``.

Transport retries: timeouts, connection errors and HTTP 429/5xx are retried
with backoff; a 429 honours Groq's ``Retry-After`` (free-tier token limits).

Token budget: every call is stateless (the caller sends a short, fixed set of
messages), the reply is capped at ``MAX_COMPLETION_TOKENS``, reasoning models
(gpt-oss) get an explicit ``reasoning_effort``, and each response's ``usage``
is logged - reasoning tokens count against the free tier's per-minute and
per-day limits, so they are worth seeing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

import backend.config as config

logger = logging.getLogger(__name__)

T = TypeVar("T")
Messages = List[Dict[str, str]]

_MAX_WAIT_SECONDS = 30.0
MAX_COMPLETION_TOKENS = 2048


class LLMError(RuntimeError):
    """The LLM could not be reached, or did not produce a usable reply."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429 or code >= 500:
            return True
        # Strict mode occasionally fails to complete the schema; a fresh generation usually does.
        return code == 400 and "json_validate_failed" in exc.response.text
    return False


def _wait(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        try:
            return min(float(exc.response.headers["retry-after"]) + 0.5, _MAX_WAIT_SECONDS)
        except (KeyError, ValueError):
            pass
    return wait_exponential(multiplier=2, min=2, max=_MAX_WAIT_SECONDS)(retry_state)


class GroqClient:
    """Minimal Groq chat client. ``client`` / ``max_attempts`` are for tests."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        max_attempts: int = 4,
        wait: Callable[[RetryCallState], float] = _wait,
    ) -> None:
        self.model = model or config.GROQ_MODEL
        if client is None:
            key = api_key if api_key is not None else config.GROQ_API_KEY
            if not key:
                raise LLMError("GROQ_API_KEY is not set. Add it to .env (free key at https://console.groq.com).")
            client = httpx.Client(
                base_url=(base_url or config.GROQ_BASE_URL).rstrip("/"),
                timeout=90.0,
                headers={"Authorization": f"Bearer {key}"},
            )
        self._client = client
        self._retrying = Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_attempts),
            wait=wait,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def chat_json(
        self,
        messages: Messages,
        schema: Dict[str, Any],
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
    ) -> str:
        """Send ``messages`` and return the reply text, constrained to ``schema``."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "agent_output", "strict": True, "schema": schema},
            },
        }
        if self.model.startswith("openai/gpt-oss"):  # other models reject the parameter
            payload["reasoning_effort"] = reasoning_effort
        logger.info("LLM call -> model=%s prompt_chars=%d", self.model, sum(len(m["content"]) for m in messages))
        try:
            body = self._retrying(self._post, payload)
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Groq returned HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach Groq: {exc}") from exc
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details") or {}
            logger.info("LLM usage <- prompt=%s completion=%s reasoning=%s total=%s",
                        usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        details.get("reasoning_tokens"), usage.get("total_tokens"))
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Groq response shape: {str(body)[:300]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Groq returned an empty message")
        return content

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()


def request_json(
    llm: Any,
    messages: Messages,
    schema: Dict[str, Any],
    build: Callable[[Dict[str, Any]], T],
    temperature: float = 0.2,
    attempts: int = 2,
    reasoning_effort: str = "medium",
) -> T:
    """Chat, parse, and ``build`` the reply; feed a validation error back once.

    ``build`` raises ``ValueError`` (pydantic's ``ValidationError`` is one) for
    a reply that breaks a business rule. Any other exception propagates.
    """
    convo = list(messages)
    error: Exception | None = None
    for _ in range(attempts):
        raw = llm.chat_json(convo, schema=schema, temperature=temperature, reasoning_effort=reasoning_effort)
        try:
            return build(json.loads(raw))
        except ValueError as exc:  # json.JSONDecodeError is a ValueError too
            error = exc
            logger.warning("LLM reply rejected: %s", exc)
            convo = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That reply was invalid: {exc}. Reply again with corrected JSON."},
            ]
    raise LLMError(f"LLM did not produce a valid reply after {attempts} attempts: {error}")
