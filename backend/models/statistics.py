"""
backend/models/statistics.py
==============================
Pydantic v2 data models for statistical comparison results.

These models are NEVER persisted to the database.
Statistical comparisons are computed on-demand by Phase 3 Statistical_Analyzer
and passed directly to Phase 4 Recommender_Agent as structured JSON.

StatisticalComparison
    Result of an independent samples t-test (scipy.stats.ttest_ind)
    comparing two experimental conditions.

SummaryStatistics
    Descriptive statistics for a metric across a set of experiments.

References
----------
- scipy.stats.ttest_ind: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
- Cohen's d: standard effect size for two independent groups
- 95% CI via scipy.stats.t.interval

Requirements
------------
5.6  Statistical_Analyzer returns p-value, effect size, CI, sample sizes
5.7  Statistical_Analyzer uses only scipy/pandas, never LLM-generated stats
"""

from __future__ import annotations

import uuid
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class StatisticalComparison(BaseModel):
    """Result of an independent samples t-test between two conditions.

    Computed by Statistical_Analyzer (Phase 3) using scipy.
    Never stored in PostgreSQL — ephemeral per-cycle result.

    Interpretation helpers (UI only, not in this model):
    - p < 0.05  : "Statistically significant difference"
    - |effect_size| > 0.8 : "Large effect"
    - |effect_size| > 0.5 : "Medium effect"
    - |effect_size| > 0.2 : "Small effect"

    Warning values
    --------------
    ``"underpowered"`` : sample size < 5 per condition (set by Statistical_Analyzer)

    Requirements covered: 5.6, 5.7
    """

    comparison_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Ephemeral UUID for this comparison result",
    )
    condition_a_name: str = Field(
        description="Human-readable label for condition A (e.g. 'dropout_0.0')"
    )
    condition_b_name: str = Field(
        description="Human-readable label for condition B (e.g. 'dropout_0.5')"
    )
    metric: str = Field(
        description="Metric being compared (e.g. 'accuracy', 'val_loss')"
    )
    t_statistic: float = Field(description="t-test statistic (scipy.stats.ttest_ind)")
    p_value: float = Field(description="Two-tailed p-value")
    effect_size: float = Field(description="Cohen's d effect size")
    confidence_interval: Tuple[float, float] = Field(
        description="95% confidence interval on mean difference (low, high)"
    )
    sample_sizes: Tuple[int, int] = Field(
        description="Number of successful experiments per condition (n_a, n_b)"
    )
    warning: Optional[str] = Field(
        default=None,
        description="'underpowered' if n < 5 per condition; None otherwise",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "comparison_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "condition_a_name": "dropout_0.0",
                "condition_b_name": "dropout_0.5",
                "metric": "accuracy",
                "t_statistic": 2.45,
                "p_value": 0.03,
                "effect_size": 0.62,
                "confidence_interval": [0.01, 0.05],
                "sample_sizes": [6, 6],
                "warning": None,
            }
        }
    }


class SummaryStatistics(BaseModel):
    """Descriptive statistics for a metric across a set of experiments.

    Computed by Statistical_Analyzer.compute_summary_statistics() (Phase 3).
    Anomalous experiments are excluded from all calculations.

    Requirements covered: 5.1, 5.5
    """

    mean: float = Field(description="Arithmetic mean of the metric values")
    std: float = Field(description="Standard deviation (population std, numpy)")
    min: float = Field(description="Minimum observed value")
    max: float = Field(description="Maximum observed value")
    median: float = Field(description="Median value (numpy.median)")
    count: int = Field(description="Number of successful experiments included")
