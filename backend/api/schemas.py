"""
backend/api/schemas.py
========================
Request/response models for the FastAPI layer.

These are the API's *edge* types. Internal domain models
(``ExperimentResult``, ``Recommendation``, ``DatasetProfile``,
``SessionSummary``) are returned directly where they already fit - no
duplicate DTOs for the sake of it.

Requirements
------------
12.1  POST /api/sessions request/response
12.2  run-cycle response
12.9  consistent ErrorResponse shape
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.models.recommendation import Recommendation
from backend.models.timestamps import UTCDateTime


class CreateSessionRequest(BaseModel):
    """Body for ``POST /api/sessions``.

    ``dataset_id`` is required - the dataset-first architecture scopes every
    session to one already-ingested dataset (design.md's ``{research_question}``
    -only sketch predates Requirement 1).
    """

    research_question: str = Field(min_length=1, max_length=500)
    dataset_id: str = Field(min_length=1)
    parent_session_id: Optional[str] = Field(
        default=None,
        description=(
            "The concluded investigation this one follows up on. Provenance "
            "only - the new investigation starts with no experiments of its "
            "own, so evidence gathered for the previous question can never "
            "contaminate this one."
        ),
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: UTCDateTime


class SessionDetailResponse(BaseModel):
    """Body for ``GET /api/sessions/{id}`` - the session row plus its experiment count."""

    session_id: str
    dataset_id: str
    parent_session_id: Optional[str] = Field(
        default=None,
        description="The investigation this one follows up on, if any",
    )
    research_question: str
    status: str
    current_node: str
    run_phase: str = Field(
        description="'idle' | 'running' | 'failed' - whether a run-cycle is actually "
        "in progress (distinct from `status`). Drives the UI's running/failed state."
    )
    run_error: Optional[str] = Field(
        default=None, description="Error text from the last failed run; null otherwise"
    )
    termination_reason: Optional[str] = Field(
        default=None,
        description=(
            "Why the investigation stopped. 'agent_concluded' - the Recommender "
            "judged the evidence sufficient. 'cycle_limit' - the safety cap "
            "stopped a loop that still wanted more experiments, which is NOT a "
            "settled answer and must be shown differently. None while active."
        ),
    )
    cycle_count: int
    experiment_count: int
    max_cycles: int = Field(
        description="The MAX_ADAPTIVE_CYCLES safety cap this session runs under"
    )
    plan_explanation: Optional[str] = Field(
        default=None, description="Planner's rationale for the initial experiment design"
    )
    created_at: UTCDateTime
    updated_at: UTCDateTime


class RunCycleResponse(BaseModel):
    """Body for ``POST /api/sessions/{id}/run-cycle`` (mirrors ``CycleResult``).

    One call runs the whole adaptive investigation - the graph loops
    internally until the Recommender concludes (or the safety cap). So
    ``cycles_completed`` is normally > 1, and ``recommendation`` is the
    FINAL one. Fetch ``GET /api/sessions/{id}/cycles`` for the per-cycle
    breakdown.
    """

    session_id: str
    current_node: str
    status: str
    run_phase: str = Field(
        default="idle",
        description="'idle' when the run finished cleanly, 'failed' if it errored",
    )
    termination_reason: Optional[str] = Field(
        default=None,
        description="'agent_concluded' | 'cycle_limit' | null (still active)",
    )
    cycles_completed: int = Field(description="Number of adaptive cycles that ran")
    experiments_completed: int
    recommendation: Optional[Recommendation] = None


class DatasetIngestRequest(BaseModel):
    """Body for ``POST /api/datasets``.

    The CSV is sent as text in ``csv_content`` (rather than a multipart file
    upload) so the API adds no ``python-multipart`` dependency. ``ingest_csv``
    then validates, profiles, and stores a canonical copy on disk.
    """

    filename: str = Field(min_length=1, description="Original file name, for display")
    csv_content: str = Field(min_length=1, description="Raw CSV text")
    target_column: str = Field(min_length=1, description="Column to predict")
    task_type_override: Optional[str] = Field(
        default=None,
        description="Force 'classification' or 'regression' instead of the inferred heuristic",
    )


class DeleteResult(BaseModel):
    """Body returned by the two DELETE endpoints.

    ``sessions_deleted`` / ``experiments_deleted`` report what a cascading
    delete removed alongside the primary resource (both 0 for a plain session
    delete's session count, since the session itself is the primary resource).
    """

    deleted: str = Field(description="'session' or 'dataset'")
    id: str
    sessions_deleted: int = 0
    experiments_deleted: int = 0


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns."""

    error: str = Field(description="Machine-readable error code, e.g. 'not_found'")
    message: str = Field(description="Human-readable explanation")
    details: Optional[dict] = Field(default=None, description="Optional structured context")
