"""
tests/unit/test_client_factory.py
===================================
Unit tests for backend/agents/client_factory.create_llm_client — provider
dispatch from ``LLM_PROVIDER``.
"""

from __future__ import annotations

import pytest

import backend.config as config
from backend.agents.client_factory import create_llm_client
from backend.agents.gemini_client import GeminiClient
from backend.agents.groq_client import GroqClient
from backend.agents.llm_client import LLMConfigError


def test_defaults_to_groq(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    client = create_llm_client()
    assert isinstance(client, GroqClient)
    client.close()


def test_gemini_when_provider_is_gemini(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    client = create_llm_client()
    assert isinstance(client, GeminiClient)
    client.close()


def test_explicit_provider_argument_overrides_config(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    client = create_llm_client("gemini")
    assert isinstance(client, GeminiClient)
    client.close()


def test_unknown_provider_raises_config_error(monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    with pytest.raises(LLMConfigError):
        create_llm_client()
