"""
backend/state_machine/graph.py
================================
The LangGraph ``StateGraph`` for the adaptive loop.

Shape
-----
    START ─(entry router on current_node)─► planning ─► executing ─► validating
                                                          ▲               │
                                     ┌────────────────────┘               ▼
                                     │                                analyzing
                                     │                                    │
                                     │        run_more_experiments        ▼
                                     └──────────────────────────────  recommending
                                                                          │ conclude
                                                                          ▼
                                                                         END

Two distinct routing mechanisms, deliberately kept separate:

1. **Entry router on ``START``** - fires *once* per ``graph.invoke()``.
   ``executor.execute_cycle`` invokes the graph with the session's persisted
   ``current_node``, and this router jumps straight to that node. This is
   crash recovery / resume - no LangGraph checkpointer, because PostgreSQL
   already holds the position.

2. **Conditional edge ``recommending → {executing | END}``** - fires every
   time ``recommending`` finishes, *inside* the same invocation. This is the
   adaptive loop: one ``execute_cycle`` call runs plan → execute → validate
   → analyze → recommend, and then loops back to ``executing`` for another
   cycle until the Recommender concludes (or the ``MAX_ADAPTIVE_CYCLES``
   safety cap forces a conclusion - see ``nodes.recommending``).

The ``recommending`` node already returns ``{"current_node": "executing"}``
or ``{"current_node": "concluded"}``; the conditional edge just reads that -
no duplicate decision logic.

Requirements
------------
13.1  StateGraph with planning/execution/validation/analysis/recommendation nodes
13.7  Conditional transition out of recommendation (execution vs conclusion)
13.8  run_more_experiments -> back to execution; conclude -> terminal
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.state_machine.context import StateMachineContext
from backend.state_machine.nodes import AdaptiveLoopNodes
from backend.state_machine.state import (
    ANALYZING,
    CONCLUDED,
    EXECUTING,
    PLANNING,
    RECOMMENDING,
    RESUMABLE_NODES,
    VALIDATING,
    AdaptiveLoopState,
)


def _entry_router(state: AdaptiveLoopState) -> str:
    """Send the invocation to whatever node the session is currently parked at.

    Mechanism (1): crash recovery / resume. Fires once, at ``START``.
    """
    node = state["current_node"]
    if node not in RESUMABLE_NODES:
        raise ValueError(
            f"Cannot resume the adaptive loop from node {node!r}; "
            f"expected one of {RESUMABLE_NODES}."
        )
    return node


def _after_recommending(state: AdaptiveLoopState) -> str:
    """Route out of ``recommending``: another cycle, or stop.

    Mechanism (2): the adaptive loop. ``recommending`` has already persisted
    its decision and set ``state["current_node"]`` to ``"executing"``
    (run another cycle) or ``"concluded"`` (stop).
    """
    return state["current_node"]


def build_adaptive_loop_graph(context: StateMachineContext):
    """Compile the adaptive-loop ``StateGraph`` bound to ``context``.

    Compilation is cheap; ``executor.execute_cycle`` builds a fresh graph
    per call so it always reflects the given context.
    """
    nodes = AdaptiveLoopNodes(context)

    graph = StateGraph(AdaptiveLoopState)
    graph.add_node(PLANNING, nodes.planning)
    graph.add_node(EXECUTING, nodes.executing)
    graph.add_node(VALIDATING, nodes.validating)
    graph.add_node(ANALYZING, nodes.analyzing)
    graph.add_node(RECOMMENDING, nodes.recommending)

    graph.add_edge(PLANNING, EXECUTING)
    graph.add_edge(EXECUTING, VALIDATING)
    graph.add_edge(VALIDATING, ANALYZING)
    graph.add_edge(ANALYZING, RECOMMENDING)

    # The adaptive loop: back to executing for another cycle, or stop.
    graph.add_conditional_edges(
        RECOMMENDING,
        _after_recommending,
        {EXECUTING: EXECUTING, CONCLUDED: END},
    )

    # Crash recovery / fresh-invocation entry.
    graph.add_conditional_edges(
        START,
        _entry_router,
        {node: node for node in RESUMABLE_NODES},
    )

    return graph.compile()
