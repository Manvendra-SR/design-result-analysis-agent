"""
backend/agents/_common.py
===========================
Shared helper for the LLM agents (Planner, Recommender): send a chat
request, parse a JSON object from the reply, and - if the first reply is
not valid JSON - make exactly one bounded "repair" attempt before giving
up.

Per Requirement 15.8 the system must fail rather than proceed on malformed
LLM output; the single repair attempt is a pragmatic concession (small
local models occasionally add a stray sentence or code fence on the first
try and correct themselves when told), not an open-ended retry loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.agents.llm_client import OllamaClient
from backend.agents.parsing import JSONParseError, parse_json_response

logger = logging.getLogger(__name__)

_REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON. Reply again with ONLY the JSON "
    "object - no prose, no explanation, no markdown code fences."
)


def request_json_object(
    llm_client: OllamaClient,
    messages: List[Dict[str, str]],
    *,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Chat, parse a JSON object, and retry once on unparseable output.

    Parameters
    ----------
    llm_client:
        The Ollama adapter.
    messages:
        OpenAI-style chat messages.
    options:
        Ollama generation options (e.g. ``{"temperature": 0.2}``).

    Returns
    -------
    dict
        The parsed top-level JSON object.

    Raises
    ------
    JSONParseError
        If neither the first reply nor the repair reply parses to an object.
    LLMError
        Propagated unchanged from the client on transport failure.
    """
    raw = llm_client.chat_completion(messages, format="json", options=options)
    try:
        return parse_json_response(raw)
    except JSONParseError as first_error:
        logger.warning(
            "LLM reply was not valid JSON; making one repair attempt: %s", first_error
        )
        repair_messages = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REPAIR_INSTRUCTION},
        ]
        raw = llm_client.chat_completion(repair_messages, format="json", options=options)
        return parse_json_response(raw)  # a second failure propagates to the caller
