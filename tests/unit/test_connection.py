"""
tests/unit/test_connection.py
==============================
Unit tests for backend/database/connection.py.

All tests use in-memory SQLite - no live PostgreSQL required.
SQLite is used solely as a fast, dependency-free stand-in to verify the
engine-creation and retry logic; PostgreSQL-specific features (JSONB, UUID)
are covered by integration tests in Phase 8.

Test cases
----------
1. test_get_engine_with_sqlite        - valid URL -> engine created, connection works
2. test_health_check_returns_true     - test_connection() returns True on valid URL
3. test_health_check_returns_false    - test_connection() returns False on bad URL
4. test_retry_on_operational_error    - transient OperationalError retried up to 3 times
5. test_retry_exhausted_raises        - persistent OperationalError raises after 3 attempts
6. test_reset_engine                  - reset_engine() disposes singleton correctly
7. test_models_create_tables_in_sqlite - ORM models can create schema on SQLite
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

# Reset the module-level singleton before each test so tests are isolated.
import backend.database.connection as conn_module
from backend.database.connection import (
    get_engine,
    reset_engine,
    test_connection as check_db_connection,
)
from backend.database.models import Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite:///:memory:"
BAD_URL = "postgresql://bad:bad@nonexistent-host:9999/nodb"


@pytest.fixture(autouse=True)
def isolate_engine():
    """Reset the engine singleton before and after every test."""
    reset_engine()
    yield
    reset_engine()


# ---------------------------------------------------------------------------
# Test 1: get_engine() succeeds with a valid SQLite URL
# ---------------------------------------------------------------------------
def test_get_engine_with_sqlite():
    """get_engine(database_url=sqlite) returns a working SQLAlchemy Engine."""
    engine = get_engine(SQLITE_URL)
    assert isinstance(engine, Engine), "Expected an Engine instance"

    # Verify the engine is actually usable
    with engine.connect() as con:
        result = con.execute(text("SELECT 42")).scalar()
    assert result == 42


# ---------------------------------------------------------------------------
# Test 2: test_connection() returns True on a good URL
# ---------------------------------------------------------------------------
def test_health_check_returns_true():
    """check_db_connection() returns True when database is reachable."""
    assert check_db_connection(SQLITE_URL) is True


# ---------------------------------------------------------------------------
# Test 3: test_connection() returns False on a bad URL
# ---------------------------------------------------------------------------
def test_health_check_returns_false():
    """check_db_connection() returns False (never raises) when DB is unreachable."""
    result = check_db_connection(BAD_URL)
    assert result is False


# ---------------------------------------------------------------------------
# Test 4: retry logic - OperationalError retried, eventually succeeds
# ---------------------------------------------------------------------------
def test_retry_on_operational_error():
    """
    _connect_with_retry retries on OperationalError.

    Simulate: first 2 calls to engine.connect() raise OperationalError,
    3rd call succeeds. Verify that _connect_with_retry succeeds after 3 attempts.
    """
    real_engine = create_engine(SQLITE_URL)
    real_connect = real_engine.connect
    call_count = {"n": 0}

    mock_connect = MagicMock()

    def fake_connect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise OperationalError("connection refused", None, Exception("refused"))
        # 3rd attempt succeeds - delegate to unpatched real connect
        return real_connect(*args, **kwargs)

    mock_connect.side_effect = fake_connect

    with patch.object(real_engine, "connect", mock_connect):
        # Call the actual retry-decorated function
        conn_module._connect_with_retry(real_engine)

    assert call_count["n"] == 3, f"Expected 3 connect attempts, got {call_count['n']}"


# ---------------------------------------------------------------------------
# Test 5: retry logic - raises OperationalError after 3 failed attempts
# ---------------------------------------------------------------------------
def test_retry_exhausted_raises():
    """Verify that _connect_with_retry raises OperationalError after 3 failed attempts."""
    mock_engine = MagicMock(spec=Engine)
    mock_engine.connect.side_effect = OperationalError("down", None, Exception("down"))

    with pytest.raises(OperationalError):
        conn_module._connect_with_retry(mock_engine)

    assert mock_engine.connect.call_count == 3


# ---------------------------------------------------------------------------
# Test 6: reset_engine() disposes and clears singleton
# ---------------------------------------------------------------------------
def test_reset_engine(monkeypatch):
    """reset_engine() disposes the engine and resets the singleton to None."""
    monkeypatch.setattr("backend.config.DATABASE_URL", SQLITE_URL)

    engine1 = get_engine()
    assert conn_module._engine is not None

    reset_engine()
    assert conn_module._engine is None, "Singleton should be None after reset"

    # A new call should create a fresh engine
    engine2 = get_engine()
    assert conn_module._engine is not None
    # They should be different objects (different connections)
    assert engine1 is not engine2


# ---------------------------------------------------------------------------
# Test 7: ORM models can create tables in SQLite
# ---------------------------------------------------------------------------
def test_models_create_tables_in_sqlite():
    """
    Base.metadata.create_all() produces all 3 tables in SQLite.

    This validates that the ORM model definitions are syntactically correct and
    compatible with SQLite's type system (portable fallbacks for UUID and JSONB).
    """
    engine = create_engine(SQLITE_URL)
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "sessions" in tables, "sessions table missing"
    assert "experiments" in tables, "experiments table missing"
    assert "anomalies" in tables, "anomalies table missing"

    # Spot-check columns on sessions table
    session_cols = {c["name"] for c in inspector.get_columns("sessions")}
    assert "session_id" in session_cols
    assert "research_question" in session_cols
    assert "current_node" in session_cols
    assert "current_recommendation" in session_cols
    assert "cycle_count" in session_cols

    # Spot-check FK on experiments -> sessions
    fks = inspector.get_foreign_keys("experiments")
    fk_tables = {fk["referred_table"] for fk in fks}
    assert "sessions" in fk_tables, "experiments should FK to sessions"

    # Spot-check FK on anomalies -> experiments
    fks = inspector.get_foreign_keys("anomalies")
    fk_tables = {fk["referred_table"] for fk in fks}
    assert "experiments" in fk_tables, "anomalies should FK to experiments"

    Base.metadata.drop_all(engine)