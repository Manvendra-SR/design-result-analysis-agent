"""
backend/agents/llm_client.py
==============================
The LLM client contract shared by the Planner and Recommender agents.

The only implementation is ``GroqClient`` (``groq_client.py``) - Groq Cloud,
``openai/gpt-oss-120b`` with strict JSON-schema structured output. This
module holds just the pieces that aren't provider-specific:

- ``LLMError``       - the single exception every LLM failure surfaces as;
- ``LLMConfigError`` - a subclass for "the client cannot be constructed"
                       (e.g. ``GROQ_API_KEY`` missing);
- ``LLMClient``      - the tiny protocol the agents depend on (``chat_json``).

Requirements
------------
15.4  Model + endpoint read from config, never hardcoded (see groq_client.py)
15.8  Malformed / failed LLM output is surfaced as an error, not swallowed
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised for any failure talking to the LLM backend.

    Covers connection failures and timeouts that survived all retries, HTTP
    error responses, and malformed response bodies. Callers (the agents, and
    the FastAPI layer) catch this at their boundary and turn it into a
    user-facing message.
    """


class LLMConfigError(LLMError):
    """Raised when the LLM client cannot be constructed - e.g. no
    ``GROQ_API_KEY`` is set."""


@runtime_checkable
class LLMClient(Protocol):
    """The small surface the Planner / Recommender agents depend on.

    ``chat_json`` sends a chat request in structured-output mode and returns
    the raw assistant text; the agents parse + validate it.
    """

    model: str

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    def health_check(self) -> bool: ...

    def close(self) -> None: ...
