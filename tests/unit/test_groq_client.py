"""
tests/unit/test_groq_client.py
================================
Unit tests for backend/agents/groq_client.py — the only LLM client.

No network is contacted: GroqClient is given a mock ``httpx.Client``.
"""

from __future__ import annotations

import json

import httpx
import pytest

import backend.config as config
from backend.agents.groq_client import GroqClient
from backend.agents.llm_client import LLMConfigError, LLMError


def test_missing_api_key_raises_config_error(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(LLMConfigError):
        GroqClient()


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(
        base_url="https://api.groq.com/openai/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test"},
    )


def test_chat_json_defaults_to_json_object_mode() -> None:
    seen: dict = {}

    def handler(_req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(_req.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}
        )

    client = GroqClient(model="openai/gpt-oss-120b", client=_mock_client(handler))
    out = client.chat_json(
        [{"role": "user", "content": "q"}], options={"temperature": 0.2}
    )

    assert out == '{"ok": true}'
    assert seen["body"]["model"] == "openai/gpt-oss-120b"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["temperature"] == 0.2


def test_chat_json_uses_strict_json_schema_when_given_a_schema() -> None:
    seen: dict = {}
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    def handler(_req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(_req.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}}]}
        )

    client = GroqClient(client=_mock_client(handler))
    client.chat_json([{"role": "user", "content": "q"}], schema=schema)

    rf = seen["body"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == schema


def test_retries_on_429_then_succeeds() -> None:
    from tenacity import wait_none

    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}}]}
        )

    client = GroqClient(
        client=_mock_client(handler), retry_wait=wait_none(), max_attempts=3
    )
    assert client.chat_json([{"role": "user", "content": "q"}]) == "{}"
    assert calls["n"] == 2


def test_429_honours_retry_after_header() -> None:
    from backend.agents.groq_client import _RateLimitAwareWait

    resp = httpx.Response(
        429, headers={"retry-after": "7"}, request=httpx.Request("POST", "http://x")
    )
    exc = httpx.HTTPStatusError("429", request=resp.request, response=resp)

    class _State:
        class outcome:
            @staticmethod
            def exception():
                return exc

    wait = _RateLimitAwareWait()(_State())  # type: ignore[arg-type]
    assert 7.0 <= wait <= 8.0  # retry_after + a small margin


def test_4xx_fails_fast_as_llm_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    client = GroqClient(client=_mock_client(handler))
    with pytest.raises(LLMError):
        client.chat_json([{"role": "user", "content": "q"}])


def test_unexpected_shape_raises_llm_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"weird": "shape"})

    client = GroqClient(client=_mock_client(handler))
    with pytest.raises(LLMError):
        client.chat_json([{"role": "user", "content": "q"}])
