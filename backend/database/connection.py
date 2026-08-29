"""
backend/database/connection.py
================================
SQLAlchemy engine factory with connection pooling and retry logic.

All database access in the application goes through ``get_engine()``.
The engine is a singleton - it is created once and reused.

Retry logic
-----------
``get_engine()`` wraps the initial connectivity check with tenacity so that
transient startup failures (e.g., PostgreSQL not yet ready) are retried up to
3 times with exponential backoff (1 s -> 2 s -> 4 s).

Connection pool
---------------
pool_size=5 : baseline connections kept open (for PostgreSQL)
max_overflow=10 : extra connections allowed when pool is exhausted
pool_pre_ping=True : validate connections before handing them out
                     (avoids "server closed connection" errors after idle)

References
----------
- SQLAlchemy engine: https://docs.sqlalchemy.org/en/20/core/engines.html
- tenacity: https://tenacity.readthedocs.io/
"""

import logging
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Module-level singleton - initialised on first call to get_engine().
_engine: Optional[Engine] = None


def _build_engine(database_url: str) -> Engine:
    """Create a new SQLAlchemy engine with appropriate pool settings."""
    kwargs = {
        "pool_pre_ping": True,
        "echo": False,
    }
    # SQLite memory/file engines use SingletonThreadPool or NullPool
    # which do not accept pool_size or max_overflow.
    if not database_url.startswith("sqlite"):
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10

    return create_engine(database_url, **kwargs)


@retry(
    retry=retry_if_exception_type(OperationalError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _connect_with_retry(engine: Engine) -> None:
    """
    Verify database connectivity with exponential-backoff retry.

    Raises OperationalError after 3 failed attempts.
    """
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection verified.")


def get_engine(database_url: Optional[str] = None) -> Engine:
    """
    Return the singleton SQLAlchemy Engine.

    On first call the engine is created and a connectivity check is performed
    (with retry). Subsequent calls return the cached engine.

    Parameters
    ----------
    database_url:
        Override DATABASE_URL for testing (e.g., ``sqlite:///:memory:``).
        When None, uses ``backend.config.DATABASE_URL``.

    Returns
    -------
    Engine
        Configured SQLAlchemy engine.

    Raises
    ------
    EnvironmentError
        If DATABASE_URL is not configured.
    OperationalError
        If the database cannot be reached after 3 retry attempts.
    """
    global _engine

    # Allow caller to pass a URL (used in tests with SQLite)
    if database_url is not None:
        engine = _build_engine(database_url)
        _connect_with_retry(engine)
        return engine

    if _engine is None:
        # Import here to avoid circular imports; config is always loaded first.
        from backend.config import DATABASE_URL

        if not DATABASE_URL:
            raise EnvironmentError(
                "DATABASE_URL is not set. Configure it in .env."
            )

        logger.info("Creating database engine...")
        _engine = _build_engine(DATABASE_URL)

        try:
            _connect_with_retry(_engine)
        except (OperationalError, RetryError) as exc:
            logger.error("Failed to connect to database after retries: %s", exc)
            _engine = None  # Allow retry on next call
            raise

    return _engine


def reset_engine() -> None:
    """
    Dispose of the current engine and reset the singleton.

    Useful in tests to force re-initialisation with a different URL.
    """
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


def test_connection(database_url: Optional[str] = None) -> bool:
    """
    Return True if the database is reachable, False otherwise.

    Does not raise - intended for health-check use.

    Parameters
    ----------
    database_url:
        Optional URL override (mainly for tests).
    """
    try:
        engine = get_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        return False