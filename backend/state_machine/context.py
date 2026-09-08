"""
backend/state_machine/context.py
==================================
``StateMachineContext``: the dependency bundle every LangGraph node needs -
the persistence layer (``StateManager``), the two LLM agents, and the three
deterministic tools.

Bundling them in one injected object (rather than each node importing
module-level singletons, as design.md's sketch did) means:

- tests build a context with an in-memory ``StateManager`` and stubbed
  agents / runner, and pass it straight in - the same constructor-injection
  pattern used for ``StateManager(engine=...)`` and
  ``ExperimentPlannerAgent(llm_client=...)``;
- the FastAPI layer builds exactly one real context and reuses it.

Both agents share a single LLM client (one HTTP connection pool).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.agents.client_factory import create_llm_client
from backend.agents.planner import ExperimentPlannerAgent
from backend.agents.recommender import RecommenderAgent
from backend.tools.anomaly_detector import AnomalyDetector
from backend.tools.experiment_runner import ExperimentRunner
from backend.tools.state_manager import StateManager
from backend.tools.statistical_analyzer import StatisticalAnalyzer


@dataclass
class StateMachineContext:
    """Everything the adaptive-loop nodes depend on."""

    state_manager: StateManager
    planner: ExperimentPlannerAgent
    recommender: RecommenderAgent
    runner: ExperimentRunner
    detector: AnomalyDetector
    analyzer: StatisticalAnalyzer

    @classmethod
    def create_default(cls) -> "StateMachineContext":
        """Build a context wired to the real database and the configured LLM.

        Reads ``DATABASE_URL`` / ``LLM_PROVIDER`` / ``GROQ_*`` / ``GEMINI_*``
        through ``backend.config`` - no network or database I/O happens here
        (``StateManager`` connects lazily on first use; the LLM client only
        builds an HTTP client). A missing API key for the selected provider
        raises ``LLMConfigError`` here.
        """
        llm = create_llm_client()
        return cls(
            state_manager=StateManager(),
            planner=ExperimentPlannerAgent(llm_client=llm),
            recommender=RecommenderAgent(llm_client=llm),
            runner=ExperimentRunner(),
            detector=AnomalyDetector(),
            analyzer=StatisticalAnalyzer(),
        )
