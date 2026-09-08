"""
tests/unit/test_gemini_client.py
==================================
Unit tests for backend/agents/gemini_client.py — the Gemini LLM client.

No network is contacted: GeminiClient is given a mock ``httpx.Client``.
Mirrors tests/unit/test_groq_client.py.
"""

from __future__ import annotations

import json

import httpx
import pytest

import backend.config as config
from backend.agents.gemini_client import GeminiClient, _to_gemini_schema
from backend.agents.llm_client import LLMConfigError, LLMError
from backend.agents.output_schemas import EXPERIMENT_PLAN_SCHEMA


def test_missing_api_key_raises_config_error(monkeypatch) -> None:
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(LLMConfigError):
        GeminiClient()


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        transport=httpx.MockTransport(handler),
        headers={"x-goog-api-key": "test"},
    )


def _ok(text: str) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]}
    )


def test_chat_json_defaults_to_plain_json_mode() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return _ok('{"ok": true}')

    client = GeminiClient(model="gemini-3.8-flash", client=_mock_client(handler))
    out = client.chat_json(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "q"},
        ],
        options={"temperature": 0.2},
    )

    assert out == '{"ok": true}'
    assert seen["path"].endswith("/models/gemini-3.8-flash:generateContent")
    gen = seen["body"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert "responseSchema" not in gen
    assert gen["temperature"] == 0.2
    # system message is lifted out of contents into systemInstruction
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert seen["body"]["contents"] == [
        {"role": "user", "parts": [{"text": "q"}]}
    ]


def test_chat_json_sends_converted_response_schema_when_given_a_schema() -> None:
    seen: dict = {}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "error": {"type": ["string", "null"]},
            "action": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["error", "action"],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return _ok("{}")

    client = GeminiClient(client=_mock_client(handler))
    client.chat_json([{"role": "user", "content": "q"}], schema=schema)

    rs = seen["body"]["generationConfig"]["responseSchema"]
    assert "additionalProperties" not in rs
    assert rs["type"] == "OBJECT"
    assert rs["properties"]["error"] == {"type": "STRING", "nullable": True}
    assert rs["properties"]["action"] == {"type": "STRING", "enum": ["a", "b"]}
    assert rs["propertyOrdering"] == ["error", "action"]


def test_to_gemini_schema_handles_the_real_plan_schema() -> None:
    converted = _to_gemini_schema(EXPERIMENT_PLAN_SCHEMA)

    def _no_additional_properties(node) -> None:
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for v in node.values():
                _no_additional_properties(v)
        elif isinstance(node, list):
            for v in node:
                _no_additional_properties(v)

    _no_additional_properties(converted)
    assert converted["properties"]["error"]["nullable"] is True
    assert converted["properties"]["error"]["type"] == "STRING"
    hp = converted["properties"]["experiments"]["items"]["properties"]["hyperparameters"]
    assert hp["properties"]["dropout"] == {"type": "NUMBER", "nullable": True}


def test_retries_on_429_then_succeeds() -> None:
    from tenacity import wait_none

    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return _ok("{}")

    client = GeminiClient(
        client=_mock_client(handler), retry_wait=wait_none(), max_attempts=3
    )
    assert client.chat_json([{"role": "user", "content": "q"}]) == "{}"
    assert calls["n"] == 2


def test_429_honours_retry_info_delay() -> None:
    from backend.agents.gemini_client import _RetryInfoAwareWait

    resp = httpx.Response(
        429,
        json={
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "7s",
                    }
                ]
            }
        },
        request=httpx.Request("POST", "http://x"),
    )
    exc = httpx.HTTPStatusError("429", request=resp.request, response=resp)

    class _State:
        class outcome:
            @staticmethod
            def exception():
                return exc

    wait = _RetryInfoAwareWait()(_State())  # type: ignore[arg-type]
    assert 7.0 <= wait <= 8.0


def test_4xx_fails_fast_as_llm_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    client = GeminiClient(client=_mock_client(handler))
    with pytest.raises(LLMError):
        client.chat_json([{"role": "user", "content": "q"}])


def test_unexpected_shape_raises_llm_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"weird": "shape"})

    client = GeminiClient(client=_mock_client(handler))
    with pytest.raises(LLMError):
        client.chat_json([{"role": "user", "content": "q"}])


def test_blocked_prompt_raises_llm_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    client = GeminiClient(client=_mock_client(handler))
    with pytest.raises(LLMError):
        client.chat_json([{"role": "user", "content": "q"}])
