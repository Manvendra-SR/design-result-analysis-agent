"""
backend/api/routes/datasets.py
================================
    POST   /api/datasets        ingest a CSV (sent as text) -> DatasetProfile
    GET    /api/datasets        list ingested datasets
    GET    /api/datasets/{id}   one dataset's profile
    DELETE /api/datasets/{id}   delete it (409 while investigations use it, unless ?cascade=true)
"""

from __future__ import annotations

import os
import tempfile
from typing import List

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import get_repository
from backend.api.schemas import DatasetIngestRequest, DeleteResult
from backend.database.repository import Repository
from backend.models.dataset import DatasetProfile
from backend.tools.dataset.ingestion import delete_dataset_files, ingest_csv

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("", response_model=DatasetProfile, status_code=201)
def ingest_dataset(body: DatasetIngestRequest, repo: Repository = Depends(get_repository)) -> DatasetProfile:
    # ingest_csv takes a path: write the text to a temp file, let it validate and
    # copy the data to data/uploads/<id>/, then remove the temp file.
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
    try:
        handle.write(body.csv_content)
        handle.close()
        profile = ingest_csv(  # DatasetValidationError -> 400
            handle.name,
            body.target_column,
            task_type_override=body.task_type_override,
            dataset_name=body.filename,
        )
    finally:
        os.unlink(handle.name)
    repo.create_dataset(profile)
    return profile


@router.get("", response_model=List[DatasetProfile])
def list_datasets(repo: Repository = Depends(get_repository)) -> List[DatasetProfile]:
    return repo.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetProfile)
def get_dataset(dataset_id: str, repo: Repository = Depends(get_repository)) -> DatasetProfile:
    return repo.get_dataset(dataset_id)  # KeyError -> 404


@router.delete("/{dataset_id}", response_model=DeleteResult)
def delete_dataset(
    dataset_id: str,
    cascade: bool = Query(default=False, description="Also delete the investigations that use it"),
    repo: Repository = Depends(get_repository),
) -> DeleteResult:
    sessions, experiments = repo.delete_dataset(dataset_id, cascade=cascade)  # 404 / 409
    delete_dataset_files(dataset_id)
    return DeleteResult(
        deleted="dataset", id=dataset_id, sessions_deleted=sessions, experiments_deleted=experiments
    )
