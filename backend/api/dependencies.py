"""
backend/api/dependencies.py
=============================
FastAPI dependency providers.

``get_context`` is a lazily-built singleton ``StateMachineContext`` (real
``StateManager`` + agents + tools). It is created on the first request that
needs it, not at import time, so ``import backend.api.app`` works without a
database or an Ollama server. Tests swap it out with
``app.dependency_overrides[get_context] = lambda: <stub context>``.
"""

from __future__ import annotations

from functools import lru_cache

from backend.state_machine.context import StateMachineContext


@lru_cache(maxsize=1)
def get_context() -> StateMachineContext:
    """Return the process-wide ``StateMachineContext`` (built once, on first use)."""
    return StateMachineContext.create_default()
