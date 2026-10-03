"""
tests/conftest.py
===================
Shared fixtures.

``engine`` is an in-memory SQLite database shared across threads
(``StaticPool``), because FastAPI's TestClient runs endpoints in a worker
thread and a default in-memory pool would give that thread an empty database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.database.models import Base
from backend.database.repository import Repository
from backend.tools.dataset.ingestion import delete_dataset_files, ingest_csv


@pytest.fixture
def engine():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def repo(engine) -> Repository:
    return Repository(engine=engine)


@pytest.fixture
def real_profile(tmp_path):
    """A small, learnable 2-class dataset, ingested for real (CSV on disk)."""
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "colour": rng.choice(["red", "blue"], size=n),
            "label": (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int),
        }
    )
    path = tmp_path / "learnable.csv"
    df.to_csv(path, index=False)
    profile = ingest_csv(str(path), target_column="label")
    yield profile
    delete_dataset_files(profile.dataset_id)
