"""
backend/database/connection.py
================================
The process-wide SQLAlchemy engine, created on first use from ``DATABASE_URL``.
"""

import logging
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import backend.config as config

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        if not config.DATABASE_URL:
            raise EnvironmentError("DATABASE_URL is not set. Configure it in .env.")
        _engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
    return _engine


def database_reachable(engine: Optional[Engine] = None) -> bool:
    """True if a trivial query succeeds. Never raises (used by /health)."""
    try:
        with (engine or get_engine()).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database connectivity check failed: %s", exc)
        return False
