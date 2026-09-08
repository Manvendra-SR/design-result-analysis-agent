# backend/agents/__init__.py
"""
backend/agents
===============
LLM agents for the Adaptive ML Experiment Agent (Phase 4).

LLM backend: selected by ``LLM_PROVIDER`` - ``groq`` (default; Groq Cloud
``openai/gpt-oss-120b``, needs ``GROQ_API_KEY``) or ``gemini`` (Google Gemini
API, needs ``GEMINI_API_KEY`` / ``GEMINI_MODEL``). Only the Planner and
Recommender call an LLM; the Anomaly_Detector is template-based. No model
name is hardcoded in this package.

Modules
-------
llm_client.py   - LLMError / LLMConfigError, and the LLMClient protocol.
groq_client.py  - GroqClient: the Groq adapter (strict json_schema structured
                  output, Retry-After-aware retry).
gemini_client.py - GeminiClient: the Gemini adapter (responseSchema structured
                  output, RetryInfo-aware retry).
client_factory.py - create_llm_client: build the client for LLM_PROVIDER.
output_schemas.py - the strict JSON Schemas for the Planner / Recommender
                  outputs, passed to GroqClient.chat_json(schema=...).
parsing.py      - parse_json_response: pull a JSON object out of an LLM's
                  text reply. JSONParseError when it cannot.
_common.py      - request_structured: chat + parse + validate + build, inside a
                  bounded retry loop that also covers semantic/validation
                  errors (StructuredOutputError). Shared by both agents.
planner.py      - ExperimentPlannerAgent: research question + DatasetProfile
                  -> validated ExperimentPlan (seeds deterministically
                  repaired). PlanningError / PlanValidationError.
recommender.py  - RecommenderAgent: experiments + pre-computed stats +
                  anomalies -> Recommendation (recommended seeds
                  deterministically repaired). RecommendationError.

Deterministic separation (Requirement 12): nothing in this package imports
scipy, numpy, torch, or sqlalchemy. Agents reason over structured input and
emit structured output; statistics and training happen elsewhere.
"""

from backend.agents.client_factory import create_llm_client
from backend.agents.gemini_client import GeminiClient
from backend.agents.groq_client import GroqClient
from backend.agents.llm_client import LLMClient, LLMConfigError, LLMError
from backend.agents.parsing import JSONParseError, parse_json_response
from backend.agents.planner import (
    ExperimentPlannerAgent,
    PlanningError,
    PlanValidationError,
)
from backend.agents.recommender import RecommendationError, RecommenderAgent

__all__ = [
    "LLMClient",
    "GroqClient",
    "GeminiClient",
    "create_llm_client",
    "LLMError",
    "LLMConfigError",
    "parse_json_response",
    "JSONParseError",
    "ExperimentPlannerAgent",
    "PlanningError",
    "PlanValidationError",
    "RecommenderAgent",
    "RecommendationError",
]
