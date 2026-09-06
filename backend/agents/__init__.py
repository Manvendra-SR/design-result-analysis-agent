# backend/agents/__init__.py
"""
backend/agents
===============
LLM agents for the Adaptive ML Experiment Agent (Phase 4).

LLM provider: a locally-running Ollama server, reached over its HTTP API.
The model tag comes from ``backend.config.OLLAMA_MODEL`` (``.env``) - no
model name is hardcoded anywhere in this package.

Modules
-------
llm_client.py   - OllamaClient: the single HTTP adapter (transport + retry +
                  logging). LLMError on any failure.
parsing.py      - parse_json_response: pull a JSON object out of an LLM's
                  text reply. JSONParseError when it cannot.
_common.py      - request_json_object: chat + parse + one repair attempt
                  (shared by both agents).
planner.py      - ExperimentPlannerAgent: research question + DatasetProfile
                  -> validated ExperimentPlan. PlanningError /
                  PlanValidationError.
recommender.py  - RecommenderAgent: experiments + pre-computed stats +
                  anomalies -> Recommendation. RecommendationError.

Deterministic separation (Requirement 12): nothing in this package imports
scipy, numpy, torch, or sqlalchemy. Agents reason over structured input and
emit structured output; statistics and training happen elsewhere.
"""

from backend.agents.llm_client import LLMError, OllamaClient
from backend.agents.parsing import JSONParseError, parse_json_response
from backend.agents.planner import (
    ExperimentPlannerAgent,
    PlanningError,
    PlanValidationError,
)
from backend.agents.recommender import RecommendationError, RecommenderAgent

__all__ = [
    "OllamaClient",
    "LLMError",
    "parse_json_response",
    "JSONParseError",
    "ExperimentPlannerAgent",
    "PlanningError",
    "PlanValidationError",
    "RecommenderAgent",
    "RecommendationError",
]
