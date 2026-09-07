# backend/models/__init__.py
"""
backend/models
===============
Pydantic v2 data-transfer objects for the Adaptive ML Experiment Agent.

Public API
----------
DatasetProfile            - profile of a user-uploaded, ingested dataset
PreprocessingConfig       - the one preprocessing choice exposed per experiment
DatasetValidationError    - raised by ingest_csv on invalid input
ExperimentConfiguration   - input spec for one ML experiment run
ExperimentResult          - completed experiment with metrics (Phase 3 output)
ExperimentPlan            - Planner_Agent output: configs + rationale (Phase 4)
AnomalyReport             - one detected anomaly (Phase 3 output)
StatisticalComparison     - t-test result between two conditions (Phase 3, not persisted)
SummaryStatistics         - descriptive statistics for a metric set (Phase 3, not persisted)
Recommendation            - next-action recommendation (Phase 4 output, stored as JSON blob)
SessionSummary            - lightweight session overview with experiment count (Phase 2)
CycleHistoryEntry         - one completed adaptive cycle's decision record (Phase 5, stored)
SessionCycle              - one adaptive cycle fully expanded for the API/UI (Phase 5/6)
"""

from backend.models.anomaly import AnomalyReport
from backend.models.cycle import CycleHistoryEntry, SessionCycle
from backend.models.dataset import DatasetProfile, DatasetValidationError, PreprocessingConfig
from backend.models.experiment import (
    ExperimentConfiguration,
    ExperimentPlan,
    ExperimentResult,
)
from backend.models.recommendation import Recommendation, SessionSummary
from backend.models.statistics import StatisticalComparison, SummaryStatistics

__all__ = [
    "DatasetProfile",
    "PreprocessingConfig",
    "DatasetValidationError",
    "ExperimentConfiguration",
    "ExperimentResult",
    "ExperimentPlan",
    "AnomalyReport",
    "StatisticalComparison",
    "SummaryStatistics",
    "Recommendation",
    "SessionSummary",
    "CycleHistoryEntry",
    "SessionCycle",
]
