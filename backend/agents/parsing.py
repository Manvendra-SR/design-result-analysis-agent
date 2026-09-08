"""
backend/agents/parsing.py
===========================
``parse_json_response``: robustly extract a JSON object from an LLM's text
response.

Groq's strict ``json_schema`` mode returns a bare JSON object, but this
helper is defensive anyway: if a reply ever arrives wrapped in a
```json ... ``` fence or with a sentence of preamble, it strips that noise
and returns a plain ``dict``. Anything it cannot turn into a top-level JSON
object raises ``JSONParseError`` with a snippet of the offending text, so
the calling agent can fail loudly rather than proceed on garbage
(Requirement 15.8).

Strategy (first match wins)
---------------------------
1. A fenced code block - ```json ... ``` or a bare ``` ... ``` fence.
2. The substring from the first ``{`` to its balanced closing ``}``
   (brace-aware, and quote/escape-aware so braces inside strings don't
   confuse the scan).
3. The whole (stripped) string.

The candidate is then ``json.loads``-ed and checked to be a ``dict``.

Requirements
------------
15.7  Parse structured JSON outputs with robust error handling
15.8  On malformed LLM output, raise an error and do not proceed
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(?P<body>.*?)\s*```",
    re.DOTALL,
)


class JSONParseError(ValueError):
    """Raised when an LLM response cannot be parsed as a top-level JSON object."""


def _extract_balanced_object(text: str) -> str | None:
    """Return the first ``{...}`` span with balanced braces, or None.

    Skips braces that appear inside double-quoted strings, and honours
    backslash escapes, so ``{"note": "a } b"}`` is handled correctly.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_response(response: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from an LLM text response.

    Parameters
    ----------
    response:
        Raw assistant text from ``LLMClient.chat_json``.

    Returns
    -------
    dict
        The parsed top-level JSON object.

    Raises
    ------
    JSONParseError
        If no candidate JSON can be found, it does not parse, or it parses
        to something other than an object (e.g. a bare list or number).
    """
    if response is None or not response.strip():
        raise JSONParseError("LLM response was empty.")

    text = response.strip()

    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group("body").strip())
    balanced = _extract_balanced_object(text)
    if balanced:
        candidates.append(balanced)
    candidates.append(text)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            raise JSONParseError(
                f"Expected a JSON object, got {type(parsed).__name__}: "
                f"{candidate[:200]}"
            )
        return parsed

    snippet = text[:300] + ("..." if len(text) > 300 else "")
    raise JSONParseError(
        f"Could not parse JSON from LLM response ({last_error}). Response was: {snippet}"
    )
