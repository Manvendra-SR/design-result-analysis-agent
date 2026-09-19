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
from typing import List, Literal, Optional, Tuple

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
    test_type: Literal["two_sample_t", "one_sample_t"] = Field(
        default="two_sample_t",
        description=(
            "'two_sample_t' - scipy.stats.ttest_ind between two varying "
            "conditions (the normal case). 'one_sample_t' - "
            "scipy.stats.ttest_1samp, used when one condition is "
            "*deterministic* (every replicate produced an identical value, as "
            "linear_baseline does: its result does not depend on random_seed). "
            "That condition is then treated as a known constant to test the "
            "other against, rather than the comparison being abandoned."
        ),
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


class ComparisonSkip(BaseModel):
    """A condition pair the analysis node could **not** compare, and why.

    Previously the analysis node caught ``InsufficientDataError`` /
    ``InsufficientVarianceError`` and silently dropped the pair, so an empty
    comparison list was indistinguishable from "nothing to compare" - the UI
    had to guess a reason, and the Recommender never learned that a knowledge
    gap existed. Every skip is now recorded and travels with the analysis.
    """

    condition_a_name: str
    condition_b_name: str
    metric: str
    reason_code: Literal["insufficient_data", "insufficient_variance"] = Field(
        description=(
            "'insufficient_data' - a condition has too few successful "
            "replicates to test. 'insufficient_variance' - *both* conditions "
            "are deterministic, so no t-test of any kind is defined."
        )
    )
    reason: str = Field(description="Human-readable detail, including the actual counts")


class ConditionSummary(BaseModel):
    """Descriptive statistics for one condition, computed by the analysis node.

    Supplied to the Recommender so it never has to count rows itself (it was
    previously miscounting replicates in its prose), and shown in the UI so a
    human can see the per-condition picture even when no pairwise comparison
    was possible.
    """

    condition_name: str
    metric: str
    n_successful: int = Field(description="Replicates contributing to these statistics")
    n_anomalous: int = Field(description="Replicates currently excluded as anomalous")
    n_failed: int = Field(description="Replicates that errored during training")
    mean: float
    std: float
    min: float
    max: float
    deterministic: bool = Field(
        description=(
            "True when every successful replicate produced an identical metric "
            "value (std == 0) - i.e. random_seed has no effect on this model, "
            "as with linear_baseline. Extra seeds add no information here."
        )
    )


class AnalysisSnapshot(BaseModel):
    """Everything the analysis node produced in one cycle.

    Computed over **all** of the session's experiments so far (statistics get
    stronger as replicates accumulate), so this is cumulative-through-cycle-N,
    not a per-cycle delta. Persisted to ``sessions.latest_analysis`` for the
    recommendation node and copied into that cycle's ``CycleHistoryEntry``.
    """

    comparisons: List[StatisticalComparison] = Field(default_factory=list)
    skipped: List[ComparisonSkip] = Field(default_factory=list)
    condition_summaries: List[ConditionSummary] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.comparisons and not self.skipped


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
