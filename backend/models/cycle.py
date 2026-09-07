"""
backend/models/cycle.py
=========================
Pydantic models for the cycle-by-cycle investigation history.

Why this exists
---------------
The adaptive loop (Phase 5) now runs many cycles inside a single
``execute_cycle`` / ``POST /run-cycle`` call, with no human in between. The
"latest-only" session columns (``current_recommendation``,
``latest_analysis``) therefore can no longer reconstruct what happened in
each cycle. Two additive persistence changes fix that:

- ``experiments.cycle`` - which adaptive cycle produced each experiment
  (anomalies are grouped by joining through the experiment);
- ``sessions.cycle_history`` - a JSON array of ``CycleHistoryEntry``, one
  per completed cycle, appended by the ``recommending`` node.

``CycleHistoryEntry``
    Exactly what is stored in ``sessions.cycle_history``: the recommendation
    the agent produced at the end of a cycle, plus the statistical
    comparisons it was based on.

``SessionCycle``
    The API view (``GET /api/sessions/{id}/cycles``): a ``CycleHistoryEntry``
    joined with that cycle's experiments and anomalies, so the UI can render
    "Cycle N" with everything under it.

DESIGN_REVIEW_CHANGES.md note
-----------------------------
That review removed the ``recommendations`` and ``statistical_comparisons``
tables, on the rationale "only the current recommendation matters for
workflow". That held under the old human-gated model (the human saw and
approved each recommendation live). It no longer holds under autonomous
multi-cycle execution - the human never sees the intermediate state, so it
must be persisted. This is done with one JSON column on ``sessions`` (same
pattern as ``current_recommendation``), not a new table.

Requirements
------------
8.3  Retrieve all experiments and recommendations for a session, ordered
11.9 UI visibly shows the progression through the adaptive loop at each stage
12.6 (spirit) statistical results visible per cycle
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from backend.models.anomaly import AnomalyReport
from backend.models.experiment import ExperimentResult
from backend.models.recommendation import Recommendation
from backend.models.statistics import StatisticalComparison


class CycleHistoryEntry(BaseModel):
    """One completed adaptive cycle's decision record.

    Stored (as a JSON object) in the ``sessions.cycle_history`` array by the
    ``recommending`` node. Append-only - never mutated once written.
    """

    cycle_number: int = Field(description="1-based adaptive cycle number")
    recommendation: Recommendation = Field(
        description="What the Recommender decided at the end of this cycle "
        "(action, next configs, explanation, evidence_summary). When the "
        "safety cap forced a stop, evidence_summary says so."
    )
    statistical_comparisons: List[StatisticalComparison] = Field(
        default_factory=list,
        description="The scipy comparisons this cycle's recommendation was based on",
    )
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class SessionCycle(BaseModel):
    """One adaptive cycle, fully expanded - the API view for the UI.

    ``GET /api/sessions/{id}/cycles`` returns a list of these, ordered by
    ``cycle_number``. Everything a human needs to answer "what happened in
    cycle N" is here:

    - ``plan_explanation`` (cycle 1 only) - why the agent chose the initial
      experiments;
    - ``experiments`` / ``anomalies`` - what actually ran and what was flagged;
    - ``statistical_comparisons`` - what the analysis showed;
    - ``recommendation`` - what the agent concluded and why it continued/stopped.
    """

    cycle_number: int
    plan_explanation: Optional[str] = Field(
        default=None,
        description="Planner's rationale for the initial design (cycle 1 only)",
    )
    experiments: List[ExperimentResult] = Field(default_factory=list)
    anomalies: List[AnomalyReport] = Field(default_factory=list)
    statistical_comparisons: List[StatisticalComparison] = Field(default_factory=list)
    recommendation: Optional[Recommendation] = Field(
        default=None,
        description="None only if the graph crashed before this cycle's recommending node",
    )
    continued: bool = Field(
        default=False,
        description="True if the agent ran another cycle after this one "
        "(recommendation.action == 'run_more_experiments')",
    )
