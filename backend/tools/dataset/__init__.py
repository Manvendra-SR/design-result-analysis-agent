"""
backend/tools/dataset
=======================
Dataset ingestion, preprocessing, splitting, and the Dataset facade.

ingestion.py     - ingest_csv: validate + profile an uploaded CSV
preprocessing.py - one-hot encoding (fixed) + StandardScaler (fit-on-train)
splitting.py     - deterministic 70/15/15 split, fixed per dataset
dataset.py       - Dataset facade: load_split(normalize) -> DatasetSplit

None of these touch the database - ingest_csv returns a DatasetProfile that
the caller persists via Repository.create_dataset
(backend/database/repository.py).
"""

from backend.tools.dataset.dataset import Dataset, DatasetSplit
from backend.tools.dataset.ingestion import ingest_csv

__all__ = ["Dataset", "DatasetSplit", "ingest_csv"]
