"""
backend/state_machine/state.py
================================
The LangGraph state schema for the adaptive loop, plus the canonical node
names.

CRITICAL design constraint (Requirement 13.2 / design.md)
--------------------------------------------------------
LangGraph state carries **only** the session id and the current workflow
position. It does NOT hold experiments, anomalies, statistical results, or
recommendations - all of that lives in PostgreSQL and is loaded/saved by
each node at execution time via ``StateManager``. This eliminates any
synchronisation between the state machine and the database: PostgreSQL is
the single source of truth.

Node names
----------
The names below are also the values stored in ``sessions.current_node`` and
surfaced by the API, so the frontend's loop visualiser and the graph use
one vocabulary. They are the gerund forms already documented on the
``sessions.current_node`` column (``planning`` | ``executing`` | ... |
``concluded``); the transitions and the node *set* match the design's
Planning / Execution / Validation / Analysis / Recommendation stages.

Requirements
------------
13.1  LangGraph state machine with nodes for planning..recommendation
13.2  State object contains only workflow position (session_id + current_node)
"""

from __future__ import annotations

from typing import TypedDict


class AdaptiveLoopState(TypedDict):
    """Everything the LangGraph state machine tracks between nodes."""

    session_id: str
    current_node: str


# ---------------------------------------------------------------------------
# Canonical node names (also the persisted ``sessions.current_node`` values)
# ---------------------------------------------------------------------------
PLANNING = "planning"
EXECUTING = "executing"
VALIDATING = "validating"
ANALYZING = "analyzing"
RECOMMENDING = "recommending"
CONCLUDED = "concluded"

#: Nodes the graph can be *entered* at when resuming a session mid-cycle.
RESUMABLE_NODES = (PLANNING, EXECUTING, VALIDATING, ANALYZING, RECOMMENDING)
