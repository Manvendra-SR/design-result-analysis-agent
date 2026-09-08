"""
backend/agents/groq_client.py
===============================
GroqClient: the LLM adapter for the Planner and Recommender agents.

Groq Cloud's OpenAI-compatible chat API, ``openai/gpt-oss-120b`` (model
configurable via ``GROQ_MODEL``). Needs ``GROQ_API_KEY``.

Structured output
-----------------
When the caller passes a ``schema`` this client uses **strict structured
output** -
``response_format: {"type": "json_schema", "json_schema": {..., "strict": true}}`` -
so generation is constrained to exactly that shape (``action`` cannot be
null, required fields cannot be missing, types are enforced). The planner /
recommender schemas live in ``agents/output_schemas.py`` in the
strict-compatible subset. With no ``schema`` it falls back to plain
``{"type": "json_object"}`` JSON mode. The ``agents/_common.py``
validation-retry loop is a backstop for the business rules a JSON schema
cannot express (hyperparameter ranges, "run_more_experiments needs a
non-empty list", ...).

Transport / retry
-----------------
A ``tenacity.Retrying`` controller retries transient failures (timeout,
connect error, HTTP 429/5xx) up to 4 times. A 429 (Groq's tokens-per-minute
limit) waits for the ``Retry-After`` the response gives; everything else
uses exponential backoff. Other 4xx fails fast. Every failure surfaces as
``LLMError``.

Requirements
------------
15.1 / 15.2  LLM API used for the Planner / Recommender agents
15.4         Model + endpoint read from config, never hardcoded
15.5         Retry with backoff on transient failures
15.6         Prompts and responses logged
15.8         Malformed output is not silently accepted (caller's retry loop)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

import backend.config as config
from backend.agents.llm_client import LLMConfigError, LLMError

logger = logging.getLogger(__name__)

_CHAT_PATH = "/chat/completions"
_MODELS_PATH = "/models"
_DEFAULT_TIMEOUT_SECONDS = 90.0
_MAX_ATTEMPTS = 4
_MAX_RETRY_WAIT_SECONDS = 30.0


class _RateLimitAwareWait(wait_base):
    """Honour Groq's ``Retry-After`` on a 429 (it tells you exactly how long
    to wait for the tokens-per-minute window to refill); fall back to
    exponential backoff for everything else."""

    def __init__(self) -> None:
        self._fallback = wait_exponential(multiplier=2, min=2, max=_MAX_RETRY_WAIT_SECONDS)

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            header = exc.response.headers.get("retry-after")
            if header:
                try:
                    return min(float(header) + 0.5, _MAX_RETRY_WAIT_SECONDS)
                except ValueError:
                    pass
        return self._fallback(retry_state)


def _is_transient(exc: BaseException) -> bool:
    """True for failures worth retrying: timeouts, connect errors, 429/5xx,
    and a 400 ``json_validate_failed`` (strict structured output where the
    model's generation did not complete the schema - a fresh generation
    usually does)."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429 or code >= 500:
            return True
        if code == 400:
            try:
                return (
                    exc.response.json().get("error", {}).get("code")
                    == "json_validate_failed"
                )
            except (ValueError, AttributeError):
                return False
    return False


class GroqClient:
    """OpenAI-compatible HTTP adapter for Groq Cloud.

    Parameters
    ----------
    api_key:
        Groq API key. Defaults to ``config.GROQ_API_KEY``. Missing -> ``LLMConfigError``.
    base_url:
        Defaults to ``config.GROQ_BASE_URL``.
    model:
        Groq model id. Defaults to ``config.GROQ_MODEL``. Never hardcoded.
    timeout:
        Per-request timeout in seconds.
    client:
        Pre-built ``httpx.Client`` (test injection). When provided,
        ``base_url`` / ``timeout`` / auth header are assumed already set on it.
    retry_wait:
        Override the backoff wait strategy (tests pass ``wait_none()``).
    max_attempts:
        Total attempts per call (default 3).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.Client] = None,
        retry_wait: Optional[wait_base] = None,
        max_attempts: int = _MAX_ATTEMPTS,
    ) -> None:
        self.base_url = (base_url or config.GROQ_BASE_URL).rstrip("/")
        self.model = model or config.GROQ_MODEL
        key = api_key if api_key is not None else config.GROQ_API_KEY

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            if not key:
                raise LLMConfigError(
                    "GROQ_API_KEY is not set. Add it to .env "
                    "(get a free key at https://console.groq.com)."
                )
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                headers={"Authorization": f"Bearer {key}"},
            )

        self._retrying = Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_attempts),
            wait=retry_wait or _RateLimitAwareWait(),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        logger.debug(
            "GroqClient ready: base_url=%s model=%s timeout=%.0fs",
            self.base_url, self.model, timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat request in structured-output mode and return the text.

        ``schema`` (when given) becomes a strict ``json_schema``
        ``response_format`` so generation is constrained to that shape;
        otherwise plain ``json_object`` mode is used. ``options`` carries
        generation knobs: ``{"temperature": 0.2}`` maps to ``temperature``.
        """
        if schema is not None:
            response_format: Dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": "agent_output", "strict": True, "schema": schema},
            }
        else:
            response_format = {"type": "json_object"}

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
            "stream": False,
        }
        if options and "temperature" in options:
            payload["temperature"] = options["temperature"]

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "LLM call -> provider=groq model=%s messages=%d prompt_chars=%d",
            self.model, len(messages), prompt_chars,
        )
        for i, m in enumerate(messages):
            logger.debug("LLM prompt[%d] (%s): %s", i, m.get("role"), m.get("content", ""))

        start = time.perf_counter()
        try:
            body = self._retrying(self._post, _CHAT_PATH, payload)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise LLMError(
                f"Groq returned HTTP {exc.response.status_code} for {self.model!r}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Groq at {self.base_url} after retries: {exc}"
            ) from exc
        duration = time.perf_counter() - start

        content = self._extract_content(body)
        logger.info(
            "LLM response <- provider=groq model=%s response_chars=%d duration=%.2fs",
            self.model, len(content), duration,
        )
        logger.debug("LLM response text: %s", content)
        return content

    def health_check(self) -> bool:
        """Return True if Groq answers ``/models``. Never raises."""
        try:
            resp = self._client.get(_MODELS_PATH)
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Groq health check failed (%s): %s", self.base_url, exc)
            return False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "GroqClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_content(body: Dict[str, Any]) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Unexpected Groq response shape (no choices[0].message.content): "
                f"{str(body)[:300]}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Groq returned an empty message: {str(body)[:300]}")
        return content
