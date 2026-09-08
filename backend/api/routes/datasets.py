"""
backend/api/routes/datasets.py
================================
Dataset ingestion + lookup.

    POST   /api/datasets           ingest a CSV (JSON body) -> DatasetProfile
    GET    /api/datasets           list ingested datasets
    GET    /api/datasets/{id}      one dataset's profile
    DELETE /api/datasets/{id}      delete a dataset (409 if in use; ?cascade=true)

These three sit *outside* design.md's "7 endpoints" count, but the API is
unusable without them: a session requires a ``dataset_id`` and Requirement 1
mandates that CSV ingestion be reachable. The CSV arrives as text in the
request body (see ``DatasetIngestRequest``), so no ``python-multipart``
dependency is added.

Requirements
------------
1.1  Validate an uploaded CSV before accepting it as a dataset
1.2  Specific, actionable error on invalid input (-> 400 via DatasetValidationError)
1.7  Persist the profile in PostgreSQL; store the file on disk
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import get_context
from backend.api.schemas import DatasetIngestRequest, DeleteResult
from backend.models.dataset import DatasetProfile
from backend.state_machine.context import StateMachineContext
from backend.tools.dataset.ingestion import delete_dataset_files, ingest_csv

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("", response_model=DatasetProfile, status_code=201)
def ingest_dataset(
    body: DatasetIngestRequest,
    ctx: StateMachineContext = Depends(get_context),
) -> DatasetProfile:
    # ingest_csv wants a path; write the posted text to a temp file, let
    # ingest_csv validate + copy it to data/uploads/<id>/, then drop the temp.
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
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

    ctx.state_manager.create_dataset(profile)
    logger.info("Ingested dataset %s (%s)", profile.dataset_id, body.filename)
    return profile


@router.get("", response_model=List[DatasetProfile])
def list_datasets(
    ctx: StateMachineContext = Depends(get_context),
) -> List[DatasetProfile]:
    return ctx.state_manager.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetProfile)
def get_dataset(
    dataset_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> DatasetProfile:
    return ctx.state_manager.get_dataset(dataset_id)  # KeyError -> 404


@router.delete("/{dataset_id}", response_model=DeleteResult)
def delete_dataset(
    dataset_id: str,
    cascade: bool = Query(
        default=False,
        description=(
            "Also delete every investigation scoped to this dataset. Without "
            "it, a dataset still referenced by any session returns 409."
        ),
    ),
    ctx: StateMachineContext = Depends(get_context),
) -> DeleteResult:
    """Delete an ingested dataset (DB row + on-disk file).

    409 if investigations still reference it, unless ``?cascade=true``.
    """
    sessions_deleted, experiments_deleted = ctx.state_manager.delete_dataset(
        dataset_id, cascade=cascade
    )  # KeyError -> 404, DatasetInUseError -> 409
    delete_dataset_files(dataset_id)
    logger.info(
        "Deleted dataset %s (cascade=%s, sessions=%d)",
        dataset_id, cascade, sessions_deleted,
    )
    return DeleteResult(
        deleted="dataset",
        id=dataset_id,
        sessions_deleted=sessions_deleted,
        experiments_deleted=experiments_deleted,
    )
