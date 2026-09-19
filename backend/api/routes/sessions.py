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
from backend.config import MAX_ADAPTIVE_CYCLES
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
    if body.parent_session_id:
        sm.get_session(body.parent_session_id)  # KeyError -> 404
    session_id = sm.create_session(
        body.research_question,
        body.dataset_id,
        parent_session_id=body.parent_session_id,
    )
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
        parent_session_id=session.parent_session_id,
        research_question=session.research_question,
        status=session.status,
        current_node=session.current_node,
        run_phase=session.run_phase,
        run_error=session.run_error,
        termination_reason=session.termination_reason,
        cycle_count=session.cycle_count,
        experiment_count=experiment_count,
        max_cycles=MAX_ADAPTIVE_CYCLES,
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

    One entry per adaptive cycle. Each mixes two scopes, and they are kept
    strictly apart (see ``backend/models/cycle.py``):

    *Per-cycle* - ``experiments``, ``anomalies_detected``,
    ``anomalies_resolved``: what THIS cycle did, from the ``experiments`` and
    ``anomalies`` rows.

    *Cumulative* - ``statistical_comparisons``, ``condition_summaries``,
    ``recommendation``, ``open_anomaly_count``: the state of the whole
    investigation as of this cycle, from ``sessions.cycle_history``.

    Anomaly flags are attributed by the cycle that raised or withdrew them
    (``detected_cycle`` / ``resolved_cycle``), not by the cycle their
    experiment happened to run in - a flag raised in cycle 2 against a cycle-1
    experiment belongs to cycle 2's work.
    """
    sm = ctx.state_manager
    session = sm.get_session(session_id)  # KeyError -> 404

    history = sm.get_cycle_history(session_id)
    experiments = sm.query_experiments(session_id)
    anomalies = sm.query_anomalies(session_id=session_id)

    exps_by_cycle: Dict[int, list] = {}
    for e in experiments:
        exps_by_cycle.setdefault(e.cycle or 0, []).append(e)

    detected_by_cycle: Dict[int, list] = {}
    resolved_by_cycle: Dict[int, list] = {}
    for a in anomalies:
        detected_by_cycle.setdefault(a.detected_cycle or 0, []).append(a)
        if a.resolved_cycle is not None:
            resolved_by_cycle.setdefault(a.resolved_cycle, []).append(a)

    history_by_cycle = {h.cycle_number: h for h in history}
    cycle_numbers = sorted(
        set(history_by_cycle) | {c for c in exps_by_cycle if c}
    )

    out: List[SessionCycle] = []
    cumulative_experiments = 0
    for n in cycle_numbers:
        entry = history_by_cycle.get(n)
        this_cycle = exps_by_cycle.get(n, [])
        cumulative_experiments += len(this_cycle)
        # Flags raised on or before cycle n and not yet withdrawn by then.
        open_at_end = sum(
            1
            for a in anomalies
            if (a.detected_cycle or 0) <= n
            and (a.resolved_cycle is None or a.resolved_cycle > n)
        )
        out.append(
            SessionCycle(
                cycle_number=n,
                plan_explanation=session.plan_explanation if n == 1 else None,
                experiments=this_cycle,
                anomalies_detected=detected_by_cycle.get(n, []),
                anomalies_resolved=resolved_by_cycle.get(n, []),
                cumulative_experiment_count=cumulative_experiments,
                open_anomaly_count=open_at_end,
                statistical_comparisons=entry.statistical_comparisons if entry else [],
                skipped_comparisons=entry.skipped_comparisons if entry else [],
                condition_summaries=entry.condition_summaries if entry else [],
                recommendation=entry.recommendation if entry else None,
                continued=bool(
                    entry
                    and entry.termination_reason is None
                    and entry.recommendation.action == "run_more_experiments"
                ),
                termination_reason=entry.termination_reason if entry else None,
            )
        )
    return out


@router.get("/{session_id}/recommendation", response_model=Recommendation)
def get_recommendation(
    session_id: str,
    ctx: StateMachineContext = Depends(get_context),
) -> Recommendation:
    """The FINAL recommendation, exactly as the agent produced it.

    Note that ``action`` may still be ``run_more_experiments`` on a concluded
    session: that means the ``MAX_ADAPTIVE_CYCLES`` cap stopped the loop while
    the agent still wanted more evidence. Read
    ``GET /sessions/{id}.termination_reason`` alongside this and present the
    two cases differently - a capped run is not a settled answer. For the
    reasoning of every intermediate cycle, use ``GET /cycles``.
    """
    sm = ctx.state_manager
    sm.get_session(session_id)  # KeyError -> 404 if the session does not exist
    recommendation = sm.get_recommendation(session_id)
    if recommendation is None:
        raise HTTPException(
            status_code=404,
            detail="No recommendation for this session yet; run a cycle first.",
        )
    return recommendation
