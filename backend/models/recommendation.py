"""
backend/models/recommendation.py
==================================
Pydantic v2 data models for recommendations and session summaries.

Recommendation
    Output of Phase 4 Recommender_Agent.
    Stored as a JSON blob in ``sessions.current_recommendation``.
    Only the *latest* recommendation per session is retained.

SessionSummary
    Lightweight session overview returned by StateManager.list_sessions().
    Includes a computed experiment_count (JOIN aggregate, not a DB column).

Requirements
------------
6.3  Recommender_Agent generates natural language explanation
6.4  Recommender_Agent recommends specific experiment configurations
6.9  Recommender_Agent recommends concluding when sufficient evidence exists
7.6  State_Manager supports listing sessions with summary statistics
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field

from backend.models.experiment import ExperimentConfiguration


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
class Recommendation(BaseModel):
    """Recommendation produced by Phase 4 Recommender_Agent.

    Stored in ``sessions.current_recommendation`` as a JSON blob (serialised
    via ``model.model_dump_json()``).  Overwritten on each new cycle —
    historical recommendations are not persisted (by design, see
    DESIGN_REVIEW_CHANGES.md).

    action values
    -------------
    ``"run_more_experiments"``
        ``recommended_experiments`` list is non-empty; state machine
        transitions back to Execution node.
    ``"conclude"``
        Sufficient evidence accumulated; state machine transitions to
        Conclusion node.

    Requirements covered: 6.3, 6.4, 6.9
    """

    action: Literal["run_more_experiments", "conclude"] = Field(
        description="Next workflow action"
    )
    recommended_experiments: List[ExperimentConfiguration] = Field(
        default_factory=list,
        description=(
            "Configurations to execute next; empty list when action='conclude'"
        ),
    )
    explanation: str = Field(
        description="LLM-generated natural language reasoning (labelled 'Interpretation' in UI)"
    )
    evidence_summary: str = Field(
        description="LLM summary of accumulated statistical evidence"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Recommendation generation timestamp (UTC)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "action": "run_more_experiments",
                "recommended_experiments": [
                    {
                        "dataset_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "model_type": "mlp",
                        "hyperparameters": {
                            "dropout": 0.1,
                            "learning_rate": 0.001,
                            "batch_size": 32,
                            "hidden_size": 64,
                            "epochs": 20,
                        },
                        "preprocessing": {"normalize": False},
                        "random_seed": 100,
                    }
                ],
                "explanation": (
                    "Dropout=0.2 shows +2.3% accuracy over no-dropout, but "
                    "variance is high (p=0.12). Recommend exploring dropout=0.1 "
                    "and dropout=0.3 with additional seeds for tighter estimates."
                ),
                "evidence_summary": (
                    "6/6 experiments completed. Dropout 0.2 mean accuracy 0.943 "
                    "vs dropout 0.0 mean 0.921. t=1.89, p=0.12, d=0.54 (medium). "
                    "Underpowered warning: n<5 per condition."
                ),
                "timestamp": "2024-01-15T10:40:00Z",
            }
        }
    }


# ---------------------------------------------------------------------------
# SessionSummary
# ---------------------------------------------------------------------------
class SessionSummary(BaseModel):
    """Lightweight session overview for list endpoints.

    ``experiment_count`` is a computed aggregate (COUNT JOIN) and is not
    a column in the sessions table.

    Requirements covered: 7.6
    """

    session_id: str = Field(description="UUID primary key")
    research_question: str = Field(description="Natural language research question")
    status: str = Field(description="'active' | 'concluded'")
    cycle_count: int = Field(description="Number of completed adaptive cycles")
    experiment_count: int = Field(
        description="Total experiments stored for this session"
    )
    created_at: datetime = Field(description="Session creation timestamp (UTC)")
