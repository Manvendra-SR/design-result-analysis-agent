"""
backend/database/init_db.py
============================
Create the tables from the ORM models.

    python -m backend.database.init_db           # create missing tables
    python -m backend.database.init_db --reset   # drop investigations and recreate

``--reset`` drops ``sessions`` and ``experiments`` (and the ``anomalies``
table from the previous design, if present) and recreates them. Ingested
datasets are kept - the ``datasets`` table did not change. Create the
database itself first (pgAdmin or ``createdb``).
"""

import logging
import sys

from sqlalchemy import text

import backend.config  # noqa: F401  (loads .env, configures logging)
from backend.database.connection import get_engine
from backend.database.models import Base

logger = logging.getLogger(__name__)

_DROP_ON_RESET = ("anomalies", "experiments", "sessions")


def init_db(reset: bool = False) -> None:
    engine = get_engine()
    if reset:
        with engine.begin() as conn:
            cascade = " CASCADE" if engine.dialect.name == "postgresql" else ""
            for table in _DROP_ON_RESET:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}{cascade}"))
        logger.info("Dropped tables: %s", ", ".join(_DROP_ON_RESET))
    Base.metadata.create_all(engine)
    logger.info("Database ready. Tables: %s", ", ".join(Base.metadata.tables))


if __name__ == "__main__":
    init_db(reset="--reset" in sys.argv)
