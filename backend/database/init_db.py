"""
backend/database/init_db.py
============================
Database initialisation script.

Reads backend/database/schema.sql and executes it against the PostgreSQL
database identified by DATABASE_URL in .env.

Usage
-----
    python -m backend.database.init_db

The script is idempotent (all CREATE statements use IF NOT EXISTS),
so it is safe to run multiple times.

Note
----
This script does NOT create the database itself — you must create the database
first in pgAdmin (or with ``createdb``), then run this script to apply the
schema.
"""

import logging
import sys
from pathlib import Path

# Import config first to ensure .env is loaded and logging is configured.
import backend.config  # noqa: F401  (side-effect: loads env + logging)
from backend.database.connection import get_engine

logger = logging.getLogger(__name__)

_SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


def init_db() -> None:
    """Execute schema.sql against the configured PostgreSQL database."""
    if not _SCHEMA_FILE.exists():
        logger.error("schema.sql not found at %s", _SCHEMA_FILE)
        sys.exit(1)

    sql = _SCHEMA_FILE.read_text(encoding="utf-8")

    logger.info("Connecting to database...")
    engine = get_engine()

    logger.info("Running schema.sql...")
    with engine.begin() as conn:
        # Execute the entire SQL file as a single statement block.
        # psycopg2 / SQLAlchemy handle multi-statement scripts via text().
        from sqlalchemy import text

        conn.execute(text(sql))

    logger.info(
        "Database initialised successfully. "
        "Tables: sessions, experiments, anomalies"
    )


if __name__ == "__main__":
    init_db()
