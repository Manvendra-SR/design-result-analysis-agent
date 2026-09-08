"""
backend/agents/gemini_client.py
=================================
GeminiClient: a second ``LLMClient`` implementation, for Google's Gemini API.

Opt-in via ``LLM_PROVIDER=gemini``. Groq stays the default and is untouched -
this client mirrors ``GroqClient``'s surface (``chat_json`` / ``health_check``
/ ``close`` / context manager) so the Planner and Recommender agents use it
without any change.

Google's Generative Language REST API (``:generateContent``), no SDK - the
same "one HTTP adapter" approach as ``GroqClient``. Needs ``GEMINI_API_KEY``;
model configurable via ``GEMINI_MODEL``.

Structured output
-----------------
When the caller passes a ``schema`` (the strict JSON Schemas in
``agents/output_schemas.py``) this client sends it as
``generationConfig.responseSchema`` with ``responseMimeType:
application/json``, so generation is constrained to that shape. Gemini's
schema dialect is a subset of OpenAPI, so ``_to_gemini_schema`` adapts the
strict schema: ``additionalProperties`` is dropped (unsupported), a
``["string", "null"]`` type union becomes ``type: "string"`` +
``nullable: true``, and ``propertyOrdering`` is filled in. With no ``schema``
it falls back to plain JSON mode (``responseMimeType`` only). The
``agents/_common.py`` validation-retry loop is the same backstop it is for
Groq.

Transport / retry
-----------------
A ``tenacity.Retrying`` controller retries transient failures (timeout,
connect error, HTTP 429/5xx) up to 4 times. A 429 honours the ``RetryInfo``
``retryDelay`` Gemini puts in the error body when present, otherwise
exponential backoff. Other 4xx fails fast. Every failure surfaces as
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

_MODELS_PATH = "/models"
_DEFAULT_TIMEOUT_SECONDS = 90.0
_MAX_ATTEMPTS = 4
_MAX_RETRY_WAIT_SECONDS = 30.0

# JSON Schema type -> Gemini (OpenAPI-subset) type
_TYPE_MAP = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _to_gemini_schema(node: Any) -> Any:
    """Adapt one of the strict ``output_schemas`` JSON Schemas to Gemini's
    ``responseSchema`` dialect.

    - ``type: ["x", "null"]`` -> ``type: "X"`` plus ``nullable: true``
    - ``additionalProperties`` is dropped (Gemini rejects it)
    - object ``properties`` get a ``propertyOrdering`` so field order is stable
    - recurses into ``properties`` and ``items``
    """
    if isinstance(node, list):
        return [_to_gemini_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key == "additionalProperties":
            continue
        if key == "type":
            types = value if isinstance(value, list) else [value]
            non_null = [t for t in types if t != "null"]
            if "null" in types:
                out["nullable"] = True
            base = non_null[0] if non_null else "string"
            out["type"] = _TYPE_MAP.get(base, base.upper())
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
            out.setdefault("propertyOrdering", list(value.keys()))
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        else:
            out[key] = value
    return out


class _RetryInfoAwareWait(wait_base):
    """Honour the ``RetryInfo.retryDelay`` Gemini returns on a 429 when it is
    present; fall back to exponential backoff otherwise."""

    def __init__(self) -> None:
        self._fallback = wait_exponential(multiplier=2, min=2, max=_MAX_RETRY_WAIT_SECONDS)

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            delay = self._retry_delay_seconds(exc.response)
            if delay is not None:
                return min(delay + 0.5, _MAX_RETRY_WAIT_SECONDS)
        return self._fallback(retry_state)

    @staticmethod
    def _retry_delay_seconds(response: httpx.Response) -> Optional[float]:
        try:
            details = response.json().get("error", {}).get("details", [])
        except (ValueError, AttributeError):
            return None
        for detail in details:
            if not isinstance(detail, dict):
                continue
            if detail.get("@type", "").endswith("RetryInfo"):
                raw = str(detail.get("retryDelay", "")).rstrip("s")
                try:
                    return float(raw)
                except ValueError:
                    return None
        return None


def _is_transient(exc: BaseException) -> bool:
    """True for failures worth retrying: timeouts, connect errors, 429, 5xx."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class GeminiClient:
    """HTTP adapter for Google's Gemini (Generative Language) API.

    Parameters
    ----------
    api_key:
        Gemini API key. Defaults to ``config.GEMINI_API_KEY``. Missing -> ``LLMConfigError``.
    base_url:
        Defaults to ``config.GEMINI_BASE_URL``.
    model:
        Gemini model id. Defaults to ``config.GEMINI_MODEL``. Never hardcoded.
    timeout:
        Per-request timeout in seconds.
    client:
        Pre-built ``httpx.Client`` (test injection). When provided,
        ``base_url`` / ``timeout`` / auth header are assumed already set on it.
    retry_wait:
        Override the backoff wait strategy (tests pass ``wait_none()``).
    max_attempts:
        Total attempts per call.
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
        self.base_url = (base_url or config.GEMINI_BASE_URL).rstrip("/")
        self.model = model or config.GEMINI_MODEL
        key = api_key if api_key is not None else config.GEMINI_API_KEY

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            if not key:
                raise LLMConfigError(
                    "GEMINI_API_KEY is not set. Add it to .env "
                    "(get a key at https://aistudio.google.com/apikey), "
                    "or set LLM_PROVIDER=groq."
                )
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                headers={"x-goog-api-key": key},
            )

        self._retrying = Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_attempts),
            wait=retry_wait or _RetryInfoAwareWait(),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        logger.debug(
            "GeminiClient ready: base_url=%s model=%s timeout=%.0fs",
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
        """Send a chat request in JSON / structured-output mode and return the text.

        ``schema`` (when given) is adapted to Gemini's ``responseSchema`` so
        generation is constrained to that shape; otherwise plain JSON mode is
        used. ``options`` carries generation knobs: ``{"temperature": 0.2}``
        maps to ``generationConfig.temperature``.
        """
        system_text, contents = self._to_gemini_messages(messages)

        generation_config: Dict[str, Any] = {"responseMimeType": "application/json"}
        if schema is not None:
            generation_config["responseSchema"] = _to_gemini_schema(schema)
        if options and "temperature" in options:
            generation_config["temperature"] = options["temperature"]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "LLM call -> provider=gemini model=%s messages=%d prompt_chars=%d",
            self.model, len(messages), prompt_chars,
        )
        for i, m in enumerate(messages):
            logger.debug("LLM prompt[%d] (%s): %s", i, m.get("role"), m.get("content", ""))

        path = f"/models/{self.model}:generateContent"
        start = time.perf_counter()
        try:
            body = self._retrying(self._post, path, payload)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise LLMError(
                f"Gemini returned HTTP {exc.response.status_code} for {self.model!r}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Gemini at {self.base_url} after retries: {exc}"
            ) from exc
        duration = time.perf_counter() - start

        content = self._extract_content(body)
        logger.info(
            "LLM response <- provider=gemini model=%s response_chars=%d duration=%.2fs",
            self.model, len(content), duration,
        )
        logger.debug("LLM response text: %s", content)
        return content

    def health_check(self) -> bool:
        """Return True if Gemini answers ``/models``. Never raises."""
        try:
            resp = self._client.get(_MODELS_PATH)
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Gemini health check failed (%s): %s", self.base_url, exc)
            return False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "GeminiClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gemini_messages(
        messages: List[Dict[str, str]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Split OpenAI-style messages into (system instruction text, contents).

        ``system`` messages are concatenated into the system instruction;
        ``assistant`` maps to Gemini's ``model`` role, everything else to
        ``user``.
        """
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            if role == "system":
                if text:
                    system_parts.append(text)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
        return "\n\n".join(system_parts), contents

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_content(body: Dict[str, Any]) -> str:
        block_reason = body.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise LLMError(f"Gemini blocked the prompt: {block_reason}")
        try:
            parts = body["candidates"][0]["content"]["parts"]
            content = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Unexpected Gemini response shape (no candidates[0].content.parts): "
                f"{str(body)[:300]}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            finish = body.get("candidates", [{}])[0].get("finishReason")
            raise LLMError(
                f"Gemini returned an empty message (finishReason={finish}): {str(body)[:300]}"
            )
        return content
