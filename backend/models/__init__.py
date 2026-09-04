# backend/models/__init__.py
"""
backend/models
===============
Pydantic v2 data-transfer objects for the Adaptive ML Experiment Agent.

Public API
----------
ExperimentConfiguration   - input spec for one ML experiment run
ExperimentResult          - completed experiment with metrics (Phase 3 output)
AnomalyReport             - one detected anomaly (Phase 3 output)
StatisticalComparison     - t-test result between two conditions (Phase 3, not persisted)
SummaryStatistics         - descriptive statistics for a metric set (Phase 3, not persisted)
Recommendation            - next-action recommendation (Phase 4 output, stored as JSON blob)
SessionSummary            - lightweight session overview with experiment count (Phase 2)
"""

from backend.models.anomaly import AnomalyReport
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.recommendation import Recommendation, SessionSummary
from backend.models.statistics import StatisticalComparison, SummaryStatistics

__all__ = [
    "ExperimentConfiguration",
    "ExperimentResult",
    "AnomalyReport",
    "StatisticalComparison",
    "SummaryStatistics",
    "Recommendation",
    "SessionSummary",
]
