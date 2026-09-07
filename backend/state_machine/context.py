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

Both agents share a single ``OllamaClient`` (one HTTP connection pool).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.agents.llm_client import OllamaClient
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
        """Build a context wired to the real database and the configured Ollama server.

        Reads ``DATABASE_URL`` / ``OLLAMA_*`` through ``backend.config`` - no
        network or database I/O happens here (``StateManager`` connects
        lazily on first use, ``OllamaClient`` only builds an HTTP client).
        """
        llm = OllamaClient()
        return cls(
            state_manager=StateManager(),
            planner=ExperimentPlannerAgent(llm_client=llm),
            recommender=RecommenderAgent(llm_client=llm),
            runner=ExperimentRunner(),
            detector=AnomalyDetector(),
            analyzer=StatisticalAnalyzer(),
        )
