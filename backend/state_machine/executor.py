"""
backend/state_machine/executor.py
===================================
``execute_cycle`` - the one entry point the API layer calls to run a
session's adaptive investigation.

    load session.current_node from PostgreSQL
        -> invoke the LangGraph graph starting from that node
        -> the graph loops planning → execute → validate → analyze →
           recommend → (execute → ... ) autonomously
        -> re-read the session and return a summary

**One call runs the whole investigation.** The graph loops
``recommending → executing`` internally (see ``graph.py``) until the
Recommender concludes or ``config.MAX_ADAPTIVE_CYCLES`` forces a stop. The
only reason a call would resume mid-loop is a crash: the persisted
``current_node`` lets the next call pick up where it died and then continue
to conclusion.

Requirements
------------
13.1-13.9  Orchestrates the LangGraph state machine
12.2       Backs POST /api/sessions/{id}/run-cycle
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from backend.config import MAX_ADAPTIVE_CYCLES
from backend.models.recommendation import Recommendation
from backend.state_machine.context import StateMachineContext
from backend.state_machine.graph import build_adaptive_loop_graph
from backend.state_machine.state import CONCLUDED

logger = logging.getLogger(__name__)

# Per cycle the graph takes ~4 super-steps (executing, validating, analyzing,
# recommending) plus planning (1) for cycle 1. LangGraph's default limit of 25
# would cap the loop well before MAX_ADAPTIVE_CYCLES, so raise it generously -
# the node-level cap in ``recommending`` is the real stop.
_RECURSION_LIMIT = MAX_ADAPTIVE_CYCLES * 6 + 20


class CycleError(RuntimeError):
    """Raised when a run cannot proceed (session already concluded, or the
    safety cap was somehow exceeded)."""


class CycleResult(BaseModel):
    """Summary of one ``execute_cycle`` call (maps 1:1 onto ``RunCycleResponse``)."""

    session_id: str
    current_node: str = Field(
        description="Where the session ended up ('concluded' on success; the "
        "crashed node if a node raised)"
    )
    status: str = Field(description="'active' | 'concluded'")
    run_phase: str = Field(
        default="idle",
        description="'idle' on a clean finish, 'failed' if a node raised",
    )
    cycles_completed: int = Field(
        description="Number of adaptive cycles run (== sessions.cycle_count)"
    )
    experiments_completed: int = Field(
        description="Total experiments stored for the session (all cycles, all statuses)"
    )
    recommendation: Optional[Recommendation] = Field(
        default=None, description="The FINAL recommendation (the one that concluded the loop)"
    )


def execute_cycle(session_id: str, context: StateMachineContext) -> CycleResult:
    """Run ``session_id``'s adaptive investigation to conclusion.

    Parameters
    ----------
    session_id:
        An existing session (created via ``StateManager.create_session``).
    context:
        The dependency bundle (persistence + agents + tools).

    Returns
    -------
    CycleResult

    Raises
    ------
    KeyError
        If the session does not exist (surfaces as HTTP 404).
    CycleError
        If the session is already concluded (409), or the LangGraph
        recursion limit was hit despite the cycle cap (500 - a bug).
    PlanningError / RecommendationError / LLMError
        Propagated unchanged from a node; the session's ``current_node`` is
        left un-advanced so a later retry re-runs that node and continues.
    """
    sm = context.state_manager
    session = sm.get_session(session_id)  # KeyError -> 404

    if session.status == CONCLUDED or session.current_node == CONCLUDED:
        raise CycleError(f"Session {session_id!r} is already concluded.")

    if session.run_phase == "running":
        # A prior run left the flag set without clearing it (hard crash, or a
        # genuine concurrent submit). Resuming from `current_node` is the right
        # move either way; just note it.
        logger.warning(
            "execute_cycle: session=%s was already run_phase='running'; resuming",
            session_id,
        )

    logger.info(
        "execute_cycle: session=%s starting from node=%s (cycle_count=%d, cap=%d)",
        session_id, session.current_node, session.cycle_count, MAX_ADAPTIVE_CYCLES,
    )

    graph = build_adaptive_loop_graph(context)
    sm.set_run_phase(session_id, "running")
    try:
        graph.invoke(
            {"session_id": session_id, "current_node": session.current_node},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except GraphRecursionError as exc:
        sm.set_run_phase(
            session_id, "failed", run_error="Adaptive loop exceeded the recursion limit."
        )
        raise CycleError(
            f"Adaptive loop exceeded the LangGraph recursion limit for session "
            f"{session_id!r} - the {MAX_ADAPTIVE_CYCLES}-cycle cap should have "
            f"stopped it first. This is a bug."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - record, then re-raise unchanged
        # A node raised (LLM failure, planner could not produce a plan, ...).
        # `current_node` is left un-advanced by the node, so a later retry
        # resumes there; persist WHY so the UI can show a failed state that
        # survives a page refresh.
        sm.set_run_phase(session_id, "failed", run_error=str(exc)[:2000])
        raise

    sm.set_run_phase(session_id, "idle")

    session = sm.get_session(session_id)
    experiments = sm.query_experiments(session_id)
    result = CycleResult(
        session_id=session_id,
        current_node=session.current_node,
        status=session.status,
        run_phase=session.run_phase,
        cycles_completed=session.cycle_count,
        experiments_completed=len(experiments),
        recommendation=sm.get_recommendation(session_id),
    )
    logger.info(
        "execute_cycle: session=%s done -> node=%s status=%s cycles=%d experiments=%d",
        session_id, result.current_node, result.status,
        result.cycles_completed, result.experiments_completed,
    )
    return result
