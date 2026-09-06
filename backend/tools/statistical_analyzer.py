"""
backend/tools/statistical_analyzer.py
========================================
StatisticalAnalyzer: scipy-based statistical comparison between two
experimental conditions, plus descriptive summary statistics.

Error signalling
----------------
``backend/models/statistics.py``'s ``StatisticalComparison`` has no
``error`` field (only ``warning``, for the "underpowered" case) — a Phase 2
decision already in place. design.md's inline sketch shows
``return StatisticalComparison(error="insufficient_data")``, which does not
match that model. Rather than widen the Pydantic contract for two rare
failure paths, this module raises typed exceptions instead:

- ``InsufficientDataError``     : fewer than 2 successful experiments in
                                   either condition (Requirement 5.1's
                                   t-test needs at least 2 samples per group)
- ``InsufficientVarianceError`` : a condition's values have essentially no
                                   spread (std < 1e-6), which would make a
                                   t-test statistic meaningless

Callers (Phase 4's Recommender_Agent, Phase 6's API layer) catch these at
their boundary and turn them into a user-facing message — matching
design.md's own "Statistical Errors" category, which describes exactly
this pair of conditions as things the caller must handle.

Requirements
------------
5.1  Independent samples t-test via scipy
5.2  Effect size (Cohen's d)
5.3  95% confidence interval
5.4  "underpowered" warning for small samples
5.5  Exclude anomalous/failed results from calculations
5.6  Return p-value, effect size, CI, sample sizes as structured data
5.7  Only scipy/numpy, never LLM-generated statistics
5.8  Validate sufficient variance before testing
5.9  Error on insufficient variance
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import stats

from backend.models.experiment import ExperimentResult
from backend.models.statistics import StatisticalComparison, SummaryStatistics

_MIN_SAMPLES_PER_CONDITION = 2
_UNDERPOWERED_THRESHOLD = 5
_MIN_STD = 1e-6
_CONFIDENCE_LEVEL = 0.95


class InsufficientDataError(ValueError):
    """Raised when a condition has fewer than 2 successful experiments."""


class InsufficientVarianceError(ValueError):
    """Raised when a condition's values have essentially zero variance."""


def _successful_values(
    experiments: List[ExperimentResult], metric: str
) -> np.ndarray:
    """Extract ``metric`` from successful experiments only.

    Excludes both "failed" (no metrics) and "anomalous" experiments, per
    Requirement 5.5 / task 3.9 ("Exclude anomalous experiments
    (status != 'success')").
    """
    return np.array(
        [
            exp.metrics[metric]
            for exp in experiments
            if exp.status == "success" and exp.metrics is not None
        ],
        dtype=float,
    )


def _infer_label(experiments: List[ExperimentResult]) -> str:
    """Build a human-readable condition label from the first experiment's config.

    Used as the default for ``condition_a_name``/``condition_b_name`` when
    the caller doesn't supply one — ``compare_conditions`` per the design
    interface takes no name arguments, but ``StatisticalComparison`` needs
    them for display.
    """
    if not experiments:
        return "condition"
    cfg = experiments[0].config
    hp = ", ".join(f"{k}={v}" for k, v in sorted(cfg.hyperparameters.items()))
    return f"{cfg.model_type}({hp})"


class StatisticalAnalyzer:
    """scipy-based statistical comparison tool.

    All calculations use scipy/numpy; nothing here is LLM-generated
    (Requirement 5.7, 11.1).
    """

    def compare_conditions(
        self,
        condition_a: List[ExperimentResult],
        condition_b: List[ExperimentResult],
        metric: str = "accuracy",
        condition_a_name: Optional[str] = None,
        condition_b_name: Optional[str] = None,
    ) -> StatisticalComparison:
        """Independent samples t-test comparing two experimental conditions.

        Parameters
        ----------
        condition_a, condition_b:
            Experiments for each condition. Only ``status == "success"``
            experiments contribute values (Requirement 5.5).
        metric:
            Key to read from each experiment's ``metrics`` dict.
        condition_a_name, condition_b_name:
            Human-readable labels. When omitted, inferred from the first
            experiment's configuration (see ``_infer_label``).

        Returns
        -------
        StatisticalComparison

        Raises
        ------
        InsufficientDataError
            If either condition has fewer than 2 successful experiments.
        InsufficientVarianceError
            If either condition's values have std < 1e-6.

        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9
        """
        a_vals = _successful_values(condition_a, metric)
        b_vals = _successful_values(condition_b, metric)

        if len(a_vals) < _MIN_SAMPLES_PER_CONDITION or len(b_vals) < _MIN_SAMPLES_PER_CONDITION:
            raise InsufficientDataError(
                f"Statistical comparison requires at least "
                f"{_MIN_SAMPLES_PER_CONDITION} successful experiments per "
                f"condition; got {len(a_vals)} and {len(b_vals)}."
            )

        std_a, std_b = float(np.std(a_vals)), float(np.std(b_vals))
        if std_a < _MIN_STD or std_b < _MIN_STD:
            raise InsufficientVarianceError(
                f"Insufficient variance for a t-test: std_a={std_a:.2e}, "
                f"std_b={std_b:.2e} (minimum {_MIN_STD:.0e})."
            )

        t_statistic, p_value = stats.ttest_ind(a_vals, b_vals)

        mean_a, mean_b = float(np.mean(a_vals)), float(np.mean(b_vals))
        pooled_std = float(np.sqrt((np.var(a_vals) + np.var(b_vals)) / 2))
        effect_size = (mean_a - mean_b) / pooled_std

        n_a, n_b = len(a_vals), len(b_vals)
        dof = n_a + n_b - 2
        se = pooled_std * np.sqrt(1 / n_a + 1 / n_b)
        ci_low, ci_high = stats.t.interval(
            _CONFIDENCE_LEVEL, dof, loc=mean_a - mean_b, scale=se
        )

        warning = (
            "underpowered"
            if n_a < _UNDERPOWERED_THRESHOLD or n_b < _UNDERPOWERED_THRESHOLD
            else None
        )

        return StatisticalComparison(
            condition_a_name=condition_a_name or _infer_label(condition_a),
            condition_b_name=condition_b_name or _infer_label(condition_b),
            metric=metric,
            t_statistic=float(t_statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_low), float(ci_high)),
            sample_sizes=(n_a, n_b),
            warning=warning,
        )

    def compute_summary_statistics(
        self, experiments: List[ExperimentResult], metric: str = "accuracy"
    ) -> SummaryStatistics:
        """Descriptive statistics for ``metric`` across successful experiments.

        Excludes anomalous/failed experiments, same as ``compare_conditions``.

        Raises
        ------
        InsufficientDataError
            If there are no successful experiments to summarize.

        Requirements: 5.1, 5.5
        """
        values = _successful_values(experiments, metric)
        if len(values) == 0:
            raise InsufficientDataError(
                "compute_summary_statistics requires at least 1 successful "
                "experiment; got 0."
            )

        return SummaryStatistics(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            median=float(np.median(values)),
            count=len(values),
        )
