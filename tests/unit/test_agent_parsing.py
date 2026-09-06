"""
tests/unit/test_agent_parsing.py
==================================
Unit tests for backend/agents/parsing.py.

Test cases (task 4.2)
---------------------
1.  test_parses_bare_json_object
2.  test_parses_fenced_json_block
3.  test_parses_fenced_block_without_language_tag
4.  test_parses_json_with_surrounding_prose
5.  test_parses_object_with_braces_inside_strings
6.  test_ignores_trailing_prose_after_object
7.  test_malformed_json_raises_jsonparseerror
8.  test_empty_response_raises
9.  test_top_level_array_raises
10. test_error_message_includes_snippet
"""

from __future__ import annotations

import pytest

from backend.agents.parsing import JSONParseError, parse_json_response


def test_parses_bare_json_object() -> None:
    assert parse_json_response('{"action": "conclude", "n": 3}') == {
        "action": "conclude",
        "n": 3,
    }


def test_parses_fenced_json_block() -> None:
    raw = 'Here is the plan:\n```json\n{"experiments": [], "explanation": "x"}\n```\n'
    assert parse_json_response(raw) == {"experiments": [], "explanation": "x"}


def test_parses_fenced_block_without_language_tag() -> None:
    raw = "```\n{\"a\": 1}\n```"
    assert parse_json_response(raw) == {"a": 1}


def test_parses_json_with_surrounding_prose() -> None:
    raw = 'Sure! The recommendation is {"action": "run_more_experiments"} — hope that helps.'
    assert parse_json_response(raw) == {"action": "run_more_experiments"}


def test_parses_object_with_braces_inside_strings() -> None:
    raw = '{"explanation": "use a set literal like {1, 2} carefully", "action": "conclude"}'
    assert parse_json_response(raw) == {
        "explanation": "use a set literal like {1, 2} carefully",
        "action": "conclude",
    }


def test_ignores_trailing_prose_after_object() -> None:
    raw = '{"a": {"b": 2}}\n\nLet me know if you want changes.'
    assert parse_json_response(raw) == {"a": {"b": 2}}


def test_malformed_json_raises_jsonparseerror() -> None:
    with pytest.raises(JSONParseError):
        parse_json_response('{"action": "conclude", }}}  not json')


def test_empty_response_raises() -> None:
    with pytest.raises(JSONParseError):
        parse_json_response("   \n  ")


def test_top_level_array_raises() -> None:
    with pytest.raises(JSONParseError):
        parse_json_response("[1, 2, 3]")


def test_error_message_includes_snippet() -> None:
    with pytest.raises(JSONParseError) as excinfo:
        parse_json_response("total nonsense, no json here at all")
    assert "no json here" in str(excinfo.value)
