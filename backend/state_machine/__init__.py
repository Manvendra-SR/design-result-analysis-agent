# backend/state_machine/__init__.py
"""
backend/state_machine
======================
The LangGraph state machine that orchestrates the adaptive loop (Phase 5).

Philosophy: LangGraph tracks only *where* a session is (``session_id`` +
``current_node``); every node loads what it needs from PostgreSQL and saves
its results back before returning. PostgreSQL is the single source of truth,
so there is no LangGraph checkpointer and no state duplication.

Modules
-------
state.py     - AdaptiveLoopState (session_id + current_node) and node names
context.py   - StateMachineContext: the StateManager + agents + tools bundle
nodes.py     - AdaptiveLoopNodes: planning / executing / validating /
               analyzing / recommending
graph.py     - build_adaptive_loop_graph: the compiled StateGraph, resumable
               from any node via a router on START
executor.py  - execute_cycle(session_id, context) -> CycleResult (the API entry point)
"""

from backend.state_machine.context import StateMachineContext
from backend.state_machine.executor import CycleError, CycleResult, execute_cycle
from backend.state_machine.graph import build_adaptive_loop_graph
from backend.state_machine.state import AdaptiveLoopState

__all__ = [
    "StateMachineContext",
    "execute_cycle",
    "CycleResult",
    "CycleError",
    "build_adaptive_loop_graph",
    "AdaptiveLoopState",
]
