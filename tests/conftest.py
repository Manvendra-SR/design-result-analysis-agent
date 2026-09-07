"""
tests/conftest.py
===================
Shared fixtures for the Phase 5 / Phase 6 tests.

Existing Phase 1-4 test modules keep their own inline fixtures; these are
additive and only apply where a test asks for them by name.

``sqlite_engine`` uses ``StaticPool`` + ``check_same_thread=False`` so a
single in-memory database is shared across threads - required because
FastAPI's ``TestClient`` runs sync endpoints in a worker thread, and the
default in-memory SQLite pool would hand that thread a fresh empty database.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.database.models import Base
from backend.tools.state_manager import StateManager


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def state_manager(sqlite_engine) -> StateManager:
    return StateManager(engine=sqlite_engine)
