"""
backend/api/routes/sessions.py
================================
Session lifecycle, the workflow endpoint, and the investigation history.

    POST   /api/sessions                       create a session (scoped to a dataset)
    GET    /api/sessions                       list sessions with summary stats
    GET    /api/sessions/{id}                   session detail + experiment count
    DELETE /api/sessions/{id}                   delete a session + its experiments/anomalies
    POST   /api/sessions/{id}/run-cycle        run the whole adaptive investigation
    GET    /api/sessions/{id}/cycles            per-cycle history (for the UI)
    GET    /api/sessions/{id}/recommendation   the final recommendation (404 if none yet)

``POST /run-cycle`` runs the entire investigation - the LangGraph graph
loops ``recommending → executing`` autonomously until the Recommender
concludes (or the ``MAX_ADAPTIVE_CYCLES`` cap). ``GET /cycles`` is how a
human/UI sees what happened in each cycle without it being a black box.

Errors are raised as plain exceptions (``KeyError`` from ``StateManager``,
``CycleError`` / ``PlanningError`` / ``LLMError`` from the state machine) and
turned into ``ErrorResponse`` bodies by ``backend/api/errors.py``.

Requirements
------------
12.1  POST /api/sessions, GET /api/sessions
12.2  POST /api/sessions/{id}/run-cycle
12.7  GET /api/sessions/{id}/recommendation
12.8  GET /api/sessions/{id}
11.9  UI visibly shows the progression through the adaptive loop -> GET /cycles
"""

from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import get_context
from backend.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteResult,
    RunCycleResponse,
    SessionDetailResponse,
)
from backend.models.cycle import SessionCycle
from backend.models.recommendation import Recommendation, SessionSummary
from backend.state_machine.context import StateMachineContext
from backend.state_machine.executor import execute_cycle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse, status_code=201)
def create_session(
    body: CreateSessionRequest,
    ctx: StateMachineContext = Depends(get_context),
) -> CreateSessionResponse:
    sm = ctx.state_manager
    sm.get_dataset(body.dataset_id)  # KeyError -> 404 if the dataset does not exist
    session_id = sm.create_session(body.research_question, body.dataset_id)
    session = sm.get_session(session_id)
    return CreateSessionResponse(session_id=session_id, created_at=session.created_at)


@router.get("", response_model=List[SessionSummary])
def list_sessions(
    ctx: StateMachineContext = Depends(get_context),
) -> List[SessionSummary]:
    return ctx.state_manager.list_sessions()


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> SessionDetailResponse:
    sm = ctx.state_manager
    session = sm.get_session(session_id)  # KeyError -> 404
    experiment_count = len(sm.query_experiments(session_id))
    return SessionDetailResponse(
        session_id=session.session_id,
        dataset_id=session.dataset_id,
        research_question=session.research_question,
        status=session.status,
        current_node=session.current_node,
        run_phase=session.run_phase,
        run_error=session.run_error,
        cycle_count=session.cycle_count,
        experiment_count=experiment_count,
        plan_explanation=session.plan_explanation,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete("/{session_id}", response_model=DeleteResult)
def delete_session(
    session_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> DeleteResult:
    """Delete an investigation and its experiments + anomalies.

    The referenced dataset is left untouched. There is no lock against
    deleting a session while a ``run-cycle`` for it is in flight - avoid
    doing that.
    """
    experiments_deleted = ctx.state_manager.delete_session(session_id)  # KeyError -> 404
    return DeleteResult(
        deleted="session",
        id=session_id,
        experiments_deleted=experiments_deleted,
    )


@router.post("/{session_id}/run-cycle", response_model=RunCycleResponse)
def run_cycle(
    session_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> RunCycleResponse:
    """Run the session's adaptive investigation to conclusion.

    The graph loops plan → execute → validate → analyze → recommend →
    execute → ... autonomously until the Recommender concludes or the
    ``MAX_ADAPTIVE_CYCLES`` safety cap is hit. Real training runs across
    several cycles here, so this can take a while - that is expected
    (Non-Functional Performance 4). If the session crashed mid-run, this
    resumes it and continues to conclusion.
    """
    result = execute_cycle(session_id, ctx)
    return RunCycleResponse(**result.model_dump())


@router.get("/{session_id}/cycles", response_model=List[SessionCycle])
def get_cycles(
    session_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> List[SessionCycle]:
    """The cycle-by-cycle investigation history.

    One entry per adaptive cycle, each combining what actually ran
    (experiments, anomalies) with what the agent decided (statistical
    comparisons, recommendation, whether it continued). Built by joining
    ``sessions.cycle_history`` with the ``experiments`` (grouped by
    ``experiments.cycle``) and ``anomalies`` rows.
    """
    sm = ctx.state_manager
    session = sm.get_session(session_id)  # KeyError -> 404

    history = sm.get_cycle_history(session_id)
    experiments = sm.query_experiments(session_id)
    anomalies = sm.query_anomalies(session_id=session_id)

    exp_cycle: Dict[str, int] = {
        e.experiment_id: (e.cycle or 0) for e in experiments
    }
    exps_by_cycle: Dict[int, list] = {}
    for e in experiments:
        exps_by_cycle.setdefault(e.cycle or 0, []).append(e)
    anoms_by_cycle: Dict[int, list] = {}
    for a in anomalies:
        anoms_by_cycle.setdefault(exp_cycle.get(a.experiment_id, 0), []).append(a)

    history_by_cycle = {h.cycle_number: h for h in history}
    cycle_numbers = sorted(
        set(history_by_cycle) | {c for c in exps_by_cycle if c}
    )

    out: List[SessionCycle] = []
    for n in cycle_numbers:
        entry = history_by_cycle.get(n)
        out.append(
            SessionCycle(
                cycle_number=n,
                plan_explanation=session.plan_explanation if n == 1 else None,
                experiments=exps_by_cycle.get(n, []),
                anomalies=anoms_by_cycle.get(n, []),
                statistical_comparisons=entry.statistical_comparisons if entry else [],
                recommendation=entry.recommendation if entry else None,
                continued=bool(
                    entry and entry.recommendation.action == "run_more_experiments"
                ),
            )
        )
    return out


@router.get("/{session_id}/recommendation", response_model=Recommendation)
def get_recommendation(
    session_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> Recommendation:
    """The FINAL recommendation - what the agent concluded. For the reasoning
    of every intermediate cycle, use ``GET /cycles``."""
    sm = ctx.state_manager
    sm.get_session(session_id)  # KeyError -> 404 if the session does not exist
    recommendation = sm.get_recommendation(session_id)
    if recommendation is None:
        raise HTTPException(
            status_code=404,
            detail="No recommendation for this session yet; run a cycle first.",
        )
    return recommendation
