"""
backend/agents/client_factory.py
==================================
``create_llm_client``: build the ``LLMClient`` the Planner and Recommender
agents use, picking the provider from ``config.LLM_PROVIDER``.

    groq   (default) -> GroqClient   (needs GROQ_API_KEY)
    gemini           -> GeminiClient (needs GEMINI_API_KEY)

The agents call this instead of constructing ``GroqClient()`` directly, so
switching provider is a ``.env`` change (``LLM_PROVIDER=gemini``) with no code
edit. Tests still inject their own stub ``LLMClient`` and never reach here.
"""

from __future__ import annotations

from typing import Optional

import backend.config as config
from backend.agents.gemini_client import GeminiClient
from backend.agents.groq_client import GroqClient
from backend.agents.llm_client import LLMClient, LLMConfigError


def create_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Return an ``LLMClient`` for ``provider`` (default: ``config.LLM_PROVIDER``).

    Raises ``LLMConfigError`` for an unknown provider or a missing API key
    (the client constructors raise the latter).
    """
    name = (provider or config.LLM_PROVIDER or "groq").strip().lower()
    if name == "groq":
        return GroqClient()
    if name == "gemini":
        return GeminiClient()
    raise LLMConfigError(
        f"Unknown LLM_PROVIDER {name!r}. Set LLM_PROVIDER to 'groq' or 'gemini'."
    )
