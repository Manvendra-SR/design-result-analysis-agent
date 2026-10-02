"""
backend/api/dependencies.py
=============================
FastAPI dependencies. Tests replace them via ``app.dependency_overrides``.

get_repository        the process-wide Repository (connects on first use)
get_investigator      ``session_id -> None``: runs one investigation to the end
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from backend.agents.llm import GroqClient, LLMError
from backend.agents.planner import Planner
from backend.agents.recommender import Recommender
from backend.database.repository import Repository
from backend.state_machine.graph import run_investigation


@lru_cache(maxsize=1)
def get_repository() -> Repository:
    return Repository()


def _investigate(session_id: str) -> None:
    repo = get_repository()
    try:
        llm = GroqClient()
    except LLMError as exc:  # e.g. no API key: show it on the session instead of crashing
        repo.set_status(session_id, "failed", error=str(exc))
        return
    run_investigation(session_id, repo, Planner(llm), Recommender(llm))


def get_investigator() -> Callable[[str], None]:
    return _investigate
