"""
backend/api/schemas.py
========================
Request/response shapes that exist only at the API edge. Domain models
(``DatasetProfile``, ``Session``, ``ExperimentResult``, ``Analysis``) are
returned directly where they already fit.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from backend.models.experiment import ExperimentResult
from backend.models.investigation import Analysis, Session


class CreateSessionRequest(BaseModel):
    research_question: str = Field(min_length=1, max_length=500)
    dataset_id: str = Field(min_length=1)


class SessionDetail(Session):
    """``GET /api/sessions/{id}``: the session, its runs, and live statistics.

    ``analysis`` is recomputed from the experiments on every request and only
    ever covers the VALIDATION split; test-split results appear solely in
    ``report``, once the investigation has finished.
    """

    experiments: List[ExperimentResult]
    analysis: Optional[Analysis] = None
    max_rounds: int
    n_seeds: int


class DatasetIngestRequest(BaseModel):
    """The CSV arrives as text, so no multipart dependency is needed."""

    filename: str = Field(min_length=1)
    csv_content: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    task_type_override: Optional[str] = Field(
        default=None, description="Force 'classification' or 'regression' instead of the inferred type"
    )


class DeleteResult(BaseModel):
    deleted: str = Field(description="'session' or 'dataset'")
    id: str
    sessions_deleted: int = 0
    experiments_deleted: int = 0


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns."""

    error: str = Field(description="Machine-readable code, e.g. 'not_found'")
    message: str
    details: Optional[dict] = None
