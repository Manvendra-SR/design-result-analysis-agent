"""Tests for backend/agents/llm.py with a mocked HTTP transport."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from backend.agents.llm import MAX_COMPLETION_TOKENS, GroqClient, LLMError, request_json
from tests._fakes import StubLLM


def _client(*responses: httpx.Response, seen=None, model="test-model") -> GroqClient:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return queue.pop(0)

    http = httpx.Client(base_url="https://groq.test", transport=httpx.MockTransport(handler))
    return GroqClient(client=http, model=model, wait=lambda _: 0)


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_sends_a_strict_schema_and_returns_the_content() -> None:
    seen = []
    reply = _client(_ok('{"a": 1}'), seen=seen).chat_json([{"role": "user", "content": "hi"}], schema={"type": "object"})
    assert reply == '{"a": 1}'
    fmt = seen[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert seen[0]["model"] == "test-model"


def test_output_is_capped_and_reasoning_effort_is_sent_only_to_reasoning_models() -> None:
    seen = []
    _client(_ok("{}"), seen=seen).chat_json([{"role": "user", "content": "x"}], schema={})
    assert seen[0]["max_completion_tokens"] == MAX_COMPLETION_TOKENS
    assert "reasoning_effort" not in seen[0]

    _client(_ok("{}"), seen=seen, model="openai/gpt-oss-120b").chat_json(
        [{"role": "user", "content": "x"}], schema={}, reasoning_effort="low")
    assert seen[1]["reasoning_effort"] == "low"


def test_token_usage_is_logged(caplog) -> None:
    usage = {"prompt_tokens": 900, "completion_tokens": 400, "total_tokens": 1300,
             "completion_tokens_details": {"reasoning_tokens": 320}}
    response = httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": usage})
    with caplog.at_level(logging.INFO, logger="backend.agents.llm"):
        _client(response).chat_json([{"role": "user", "content": "x"}], schema={})
    assert "prompt=900 completion=400 reasoning=320 total=1300" in caplog.text


def test_rate_limits_and_server_errors_are_retried() -> None:
    client = _client(httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(503), _ok("{}"))
    assert client.chat_json([{"role": "user", "content": "x"}], schema={}) == "{}"


def test_client_errors_fail_fast_as_llm_error() -> None:
    with pytest.raises(LLMError, match="HTTP 401"):
        _client(httpx.Response(401, text="bad key")).chat_json([{"role": "user", "content": "x"}], schema={})


def test_missing_api_key_is_an_llm_error() -> None:
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient(api_key="")


def test_request_json_retries_once_with_the_error() -> None:
    llm = StubLLM({"n": -1}, {"n": 3})

    def build(reply):
        if reply["n"] < 0:
            raise ValueError("n must be positive")
        return reply["n"]

    assert request_json(llm, [{"role": "user", "content": "q"}], {}, build) == 3
    assert "n must be positive" in llm.requests[1][-1]["content"]
