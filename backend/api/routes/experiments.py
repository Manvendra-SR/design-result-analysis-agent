"""
backend/api/routes/experiments.py
===================================
Read-only experiment queries.

    GET /api/sessions/{id}/experiments?status=   experiments for a session
    GET /api/experiments/{id}                    one experiment in detail

Requirements
------------
12.4  GET /api/sessions/{id}/experiments (optional ?status= filter)
12.5  GET /api/experiments/{id}
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.dependencies import get_context
from backend.models.experiment import ExperimentResult
from backend.state_machine.context import StateMachineContext

router = APIRouter(prefix="/api", tags=["experiments"])

_ALLOWED_STATUS = {"success", "failed", "anomalous"}


@router.get("/sessions/{session_id}/experiments", response_model=List[ExperimentResult])
def list_session_experiments(
    session_id: str,
    status: Optional[str] = Query(
        default=None, description="Filter: success | failed | anomalous"
    ),
    ctx: StateMachineContext = Depends(get_context),
) -> List[ExperimentResult]:
    sm = ctx.state_manager
    sm.get_session(session_id)  # KeyError -> 404
    if status is not None and status not in _ALLOWED_STATUS:
        # An unknown status is a client error, not an empty result.
        raise HTTPException(
            status_code=400,
            detail=f"Unknown status {status!r}; expected one of {sorted(_ALLOWED_STATUS)}.",
        )
    return sm.query_experiments(session_id, status=status)


@router.get("/experiments/{experiment_id}", response_model=ExperimentResult)
def get_experiment(
    experiment_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> ExperimentResult:
    return ctx.state_manager.get_experiment(experiment_id)  # KeyError -> 404
