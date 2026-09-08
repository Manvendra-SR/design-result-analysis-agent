"""
backend/agents/_common.py
===========================
Shared helper for the LLM agents (Planner, Recommender): send a chat
request, parse a JSON object from the reply, hand it to a caller-supplied
*builder* that validates it and constructs the domain object, and - if any
of that fails in a way a retry might fix - feed the specific error back to
the model and try again, up to a small bounded number of attempts.

Why the retry covers *validation*, not just *parsing*
----------------------------------------------------
Groq's strict JSON-schema mode makes shape errors (``"action": null``, a
missing field, a wrong type) impossible - but it cannot enforce the
business rules: a hyperparameter out of range, ``run_more_experiments`` with
an empty list, an unanswerable question. Those surface from the builder, so
the builder runs *inside* the retry loop: ``JSONParseError`` and
``StructuredOutputError`` / ``pydantic.ValidationError`` raised by it each
trigger another attempt with the specific error fed back to the model.

Per Requirement 15.8 this never *weakens* validation - after the attempt
budget is spent the last error propagates and the calling agent raises
``PlanningError`` / ``RecommendationError``.

A builder may raise something *other* than the retryable set (e.g. the
Planner raises ``PlanningError`` for a genuinely unanswerable question);
that propagates immediately without burning retries.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pydantic import ValidationError

from backend.agents.llm_client import LLMClient
from backend.agents.parsing import JSONParseError, parse_json_response

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Total attempts (initial + retries) for one structured request.
_DEFAULT_MAX_ATTEMPTS = 3

_RETRYABLE = (JSONParseError, ValidationError)


class StructuredOutputError(ValueError):
    """The LLM reply parsed as JSON but was structurally/semantically wrong
    in a way a retry might fix (missing field, wrong type, empty list, ...).

    Builders raise this (rather than the agent's terminal error) so the
    retry loop gets a chance before the agent gives up.
    """


def _correction_message(error: Exception) -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"That reply was not usable: {error}. "
            "Reply again with ONLY a single JSON object of exactly the shape "
            "described above - no prose, no explanation, no markdown fences, "
            "and make sure every required field is present with the right type."
        ),
    }


def request_structured(
    llm_client: LLMClient,
    messages: List[Dict[str, str]],
    *,
    build: Callable[[Dict[str, Any]], T],
    schema: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> T:
    """Chat -> parse JSON -> ``build`` it, retrying on parse/validation errors.

    Parameters
    ----------
    llm_client:
        Any ``LLMClient`` (``GroqClient``, or a stub in tests).
    messages:
        OpenAI-style chat messages (the request as first sent).
    build:
        ``dict -> T``. Validates the parsed JSON and constructs the domain
        object. Raises ``StructuredOutputError`` / ``pydantic.ValidationError``
        for retryable problems; may raise anything else to fail immediately.
    schema:
        JSON Schema forwarded to ``chat_json``. Groq constrains generation to
        it (strict structured output).
    options:
        Generation options passed through to the client (e.g. temperature).
    max_attempts:
        Total attempts (initial + retries). Minimum 1.

    Returns
    -------
    T
        Whatever ``build`` returns.

    Raises
    ------
    JSONParseError / StructuredOutputError / pydantic.ValidationError
        The last such error, once the attempt budget is spent.
    LLMError
        Propagated unchanged from the client on transport failure.
    """
    attempts = max(1, max_attempts)
    convo = list(messages)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        raw = llm_client.chat_json(convo, schema=schema, options=options)
        try:
            parsed = parse_json_response(raw)
            return build(parsed)
        except (StructuredOutputError, *_RETRYABLE) as exc:
            last_error = exc
            if attempt == attempts:
                break
            logger.warning(
                "LLM structured output invalid (attempt %d/%d): %s",
                attempt, attempts, exc,
            )
            convo = [*messages, {"role": "assistant", "content": raw}, _correction_message(exc)]

    assert last_error is not None  # loop always runs at least once
    raise last_error
