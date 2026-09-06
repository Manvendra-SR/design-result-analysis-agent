"""
tests/unit/test_llm_client.py
===============================
Unit tests for backend/agents/llm_client.py.

All HTTP traffic is served by an ``httpx.MockTransport`` - no Ollama server
is contacted. Retry backoff is disabled per-test via ``retry_wait=wait_none()``.

Test cases (task 4.1)
---------------------
1.  test_chat_completion_returns_message_content
2.  test_request_payload_shape
3.  test_format_json_and_options_forwarded
4.  test_retries_transient_5xx_then_succeeds
5.  test_retries_exhausted_raises_llmerror
6.  test_client_error_4xx_not_retried
7.  test_connection_error_retried_then_llmerror
8.  test_malformed_response_body_raises_llmerror
9.  test_model_name_comes_from_config_not_hardcoded
10. test_health_check_true_and_false
"""

from __future__ import annotations

import json
from typing import Callable, List

import httpx
import pytest
from tenacity import wait_none

import backend.config as config
from backend.agents.llm_client import LLMError, OllamaClient


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test-ollama:11434",
    )
    return OllamaClient(client=http, model="test-model", retry_wait=wait_none())


def _ok(content: str = "hello") -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


# ---------------------------------------------------------------------------
# 1-3. Happy path + request shape
# ---------------------------------------------------------------------------

def test_chat_completion_returns_message_content() -> None:
    client = _client(lambda req: _ok("the answer"))
    assert client.chat_completion([{"role": "user", "content": "q"}]) == "the answer"


def test_request_payload_shape() -> None:
    seen: List[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        assert req.url.path == "/api/chat"
        return _ok()

    _client(handler).chat_completion([{"role": "user", "content": "q"}])

    body = seen[0]
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "q"}]
    assert "format" not in body  # not requested here


def test_format_json_and_options_forwarded() -> None:
    seen: List[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return _ok()

    _client(handler).chat_completion(
        [{"role": "user", "content": "q"}],
        format="json",
        options={"temperature": 0.1},
    )

    assert seen[0]["format"] == "json"
    assert seen[0]["options"] == {"temperature": 0.1}


# ---------------------------------------------------------------------------
# 4-7. Retry / error handling
# ---------------------------------------------------------------------------

def test_retries_transient_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="overloaded")
        return _ok("recovered")

    result = _client(handler).chat_completion([{"role": "user", "content": "q"}])
    assert result == "recovered"
    assert calls["n"] == 3


def test_retries_exhausted_raises_llmerror() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError):
        _client(handler).chat_completion([{"role": "user", "content": "q"}])
    assert calls["n"] == 3


def test_client_error_4xx_not_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    with pytest.raises(LLMError):
        _client(handler).chat_completion([{"role": "user", "content": "q"}])
    assert calls["n"] == 1  # permanent error - one attempt only


def test_connection_error_retried_then_llmerror() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError):
        _client(handler).chat_completion([{"role": "user", "content": "q"}])
    assert calls["n"] == 3


def test_malformed_response_body_raises_llmerror() -> None:
    client = _client(lambda req: httpx.Response(200, json={"unexpected": "shape"}))
    with pytest.raises(LLMError):
        client.chat_completion([{"role": "user", "content": "q"}])


# ---------------------------------------------------------------------------
# 9. Model name from config
# ---------------------------------------------------------------------------

def test_model_name_comes_from_config_not_hardcoded() -> None:
    http = httpx.Client(
        transport=httpx.MockTransport(lambda req: _ok()),
        base_url="http://test-ollama:11434",
    )
    client = OllamaClient(client=http, retry_wait=wait_none())
    assert client.model == config.OLLAMA_MODEL


# ---------------------------------------------------------------------------
# 10. health_check
# ---------------------------------------------------------------------------

def test_health_check_true_and_false() -> None:
    up = _client(lambda req: httpx.Response(200, json={"models": []}))
    assert up.health_check() is True

    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert _client(down).health_check() is False
