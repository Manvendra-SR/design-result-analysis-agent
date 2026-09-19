"""
backend/models/cycle.py
=========================
Pydantic models for the cycle-by-cycle investigation history.

Why this exists
---------------
The adaptive loop runs many cycles inside a single ``execute_cycle`` /
``POST /run-cycle`` call, with no human in between. The "latest-only" session
columns (``current_recommendation``, ``latest_analysis``) therefore cannot
reconstruct what happened in each cycle. Two additive persistence changes fix
that:

- ``experiments.cycle`` - which adaptive cycle produced each experiment;
- ``sessions.cycle_history`` - a JSON array of ``CycleHistoryEntry``, one per
  completed cycle, appended by the ``recommending`` node.

Per-cycle vs cumulative - the distinction that matters
-------------------------------------------------------
Two different scopes coexist in one cycle record, and conflating them is what
made earlier output look self-contradictory ("Cycle 3: 3 experiments ·
0 anomalies" printed directly above "Two MLP runs were flagged as anomalous"):

*Per-cycle (a delta - what this cycle did)*
    ``experiments``, ``anomalies_detected``, ``anomalies_resolved``.

*Cumulative (the whole investigation through this cycle)*
    ``statistical_comparisons``, ``condition_summaries``, ``recommendation``,
    ``open_anomaly_count``.

The Recommender reasons over cumulative evidence **by design** - that is what
makes the loop adaptive - so its prose legitimately refers to earlier cycles.
Every field is now labelled with its scope here and in the UI, and the
Recommender is given the cycle each experiment and anomaly belongs to so it can
attribute them explicitly instead of speaking timelessly.

Requirements
------------
8.3  Retrieve all experiments and recommendations for a session, ordered
11.9 UI visibly shows the progression through the adaptive loop at each stage
12.6 (spirit) statistical results visible per cycle
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from backend.models.anomaly import AnomalyReport
from backend.models.experiment import ExperimentResult
from backend.models.recommendation import Recommendation
from backend.models.statistics import (
    ComparisonSkip,
    ConditionSummary,
    StatisticalComparison,
)
from backend.models.timestamps import UTCDateTime

#: Why an investigation stopped. ``agent_concluded`` - the Recommender judged
#: the evidence sufficient. ``cycle_limit`` - the MAX_ADAPTIVE_CYCLES safety cap
#: stopped a loop that still wanted to continue; that is NOT a settled answer
#: and must never be presented as one.
TerminationReason = Literal["agent_concluded", "cycle_limit"]


class CycleHistoryEntry(BaseModel):
    """One completed adaptive cycle's decision record.

    Stored (as a JSON object) in the ``sessions.cycle_history`` array by the
    ``recommending`` node. Append-only - never mutated once written, and the
    Recommender's output is stored **verbatim**. When the safety cap stops the
    loop, ``termination_reason`` records that separately rather than the
    recommendation being rewritten to look like a conclusion.
    """

    cycle_number: int = Field(description="1-based adaptive cycle number")
    recommendation: Recommendation = Field(
        description="What the Recommender decided at the end of this cycle, "
        "exactly as it produced it (action, next configs, explanation, "
        "evidence_summary). Based on cumulative evidence."
    )
    statistical_comparisons: List[StatisticalComparison] = Field(
        default_factory=list,
        description="CUMULATIVE: scipy comparisons over every experiment run so far",
    )
    skipped_comparisons: List[ComparisonSkip] = Field(
        default_factory=list,
        description="CUMULATIVE: condition pairs that could not be compared, and why",
    )
    condition_summaries: List[ConditionSummary] = Field(
        default_factory=list,
        description="CUMULATIVE: per-condition descriptive statistics",
    )
    termination_reason: Optional[TerminationReason] = Field(
        default=None,
        description="Set only on the final cycle: why the investigation stopped",
    )
    recorded_at: UTCDateTime = Field(default_factory=datetime.utcnow)


class SessionCycle(BaseModel):
    """One adaptive cycle, fully expanded - the API view for the UI.

    ``GET /api/sessions/{id}/cycles`` returns a list of these, ordered by
    ``cycle_number``. Field names carry their scope so the UI cannot mix a
    per-cycle count with a cumulative statement (see the module docstring).
    """

    cycle_number: int

    plan_explanation: Optional[str] = Field(
        default=None,
        description="Planner's rationale for the initial design (cycle 1 only)",
    )

    # --- per-cycle (what this cycle actually did) ---------------------------
    experiments: List[ExperimentResult] = Field(
        default_factory=list,
        description="PER-CYCLE: experiments (training runs) executed in this cycle",
    )
    anomalies_detected: List[AnomalyReport] = Field(
        default_factory=list,
        description="PER-CYCLE: anomaly flags first raised by this cycle's validation node",
    )
    anomalies_resolved: List[AnomalyReport] = Field(
        default_factory=list,
        description=(
            "PER-CYCLE: flags this cycle's validation node withdrew because the "
            "extra replicates showed the value was ordinary after all"
        ),
    )

    # --- cumulative (the whole investigation through this cycle) ------------
    cumulative_experiment_count: int = Field(
        default=0,
        description="CUMULATIVE: experiments run in this and all earlier cycles",
    )
    open_anomaly_count: int = Field(
        default=0,
        description="CUMULATIVE: anomaly flags still open at the end of this cycle",
    )
    statistical_comparisons: List[StatisticalComparison] = Field(
        default_factory=list,
        description="CUMULATIVE: what the analysis showed over all evidence so far",
    )
    skipped_comparisons: List[ComparisonSkip] = Field(
        default_factory=list,
        description="CUMULATIVE: pairs that could not be compared, with the real reason",
    )
    condition_summaries: List[ConditionSummary] = Field(
        default_factory=list,
        description="CUMULATIVE: per-condition descriptive statistics",
    )
    recommendation: Optional[Recommendation] = Field(
        default=None,
        description="CUMULATIVE decision. None only if the graph crashed before "
        "this cycle's recommending node",
    )
    continued: bool = Field(
        default=False,
        description="True if another cycle ran after this one",
    )
    termination_reason: Optional[TerminationReason] = Field(
        default=None,
        description="Set only on the final cycle: why the investigation stopped",
    )
