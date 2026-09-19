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

- ``InsufficientDataError``     : too few successful replicates to run any
                                   test (Requirement 5.1's t-test needs at
                                   least 2 samples in the varying condition)
- ``InsufficientVarianceError`` : *both* conditions are deterministic, so no
                                   t-test of any kind is defined

Callers (Phase 4's Recommender_Agent, Phase 6's API layer) catch these at
their boundary and turn them into a user-facing message — matching
design.md's own "Statistical Errors" category, which describes exactly
this pair of conditions as things the caller must handle. The analysis node
records every skip with its reason rather than dropping it silently.

Deterministic conditions (why a t-test still happens)
-----------------------------------------------------
``linear_baseline`` fits logistic/linear regression on a fixed train/val
split, so ``random_seed`` has no effect: every replicate returns a byte
-identical metric and the group's std is exactly 0. Treating that as
"insufficient variance" abandoned the comparison entirely — in a real run,
16 MLP replicates versus 6 identical baseline replicates produced *zero*
statistical output for six straight cycles, which in turn starved the
Recommender of the significant result it needed in order to conclude.

A deterministic condition is not missing information; it is a **known
constant**. The correct test is a one-sample t-test of the varying condition
against that constant (``scipy.stats.ttest_1samp``), which is what this module
now does, tagging the result ``test_type="one_sample_t"`` so nothing downstream
mistakes it for a two-sample comparison. Only when *both* sides are
deterministic is there genuinely nothing to test.

A degenerate group must have at least 2 identical observations to be trusted as
a constant — a single observation is just an unreplicated measurement, and is
reported as insufficient data instead.

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
from backend.models.statistics import (
    ConditionSummary,
    StatisticalComparison,
    SummaryStatistics,
)

_MIN_SAMPLES_PER_CONDITION = 2
_UNDERPOWERED_THRESHOLD = 5
_MIN_STD = 1e-6
_CONFIDENCE_LEVEL = 0.95


class InsufficientDataError(ValueError):
    """Raised when there are too few successful replicates to test."""


class InsufficientVarianceError(ValueError):
    """Raised when *both* conditions are deterministic, leaving nothing to test.

    A single deterministic condition is handled as a known constant via a
    one-sample t-test — see the module docstring.
    """


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
            If the varying condition has fewer than 2 successful experiments,
            or a deterministic condition has fewer than 2 (so its constancy is
            unverified).
        InsufficientVarianceError
            If BOTH conditions are deterministic.

        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9
        """
        a_vals = _successful_values(condition_a, metric)
        b_vals = _successful_values(condition_b, metric)
        name_a = condition_a_name or _infer_label(condition_a)
        name_b = condition_b_name or _infer_label(condition_b)

        if (
            len(a_vals) < _MIN_SAMPLES_PER_CONDITION
            or len(b_vals) < _MIN_SAMPLES_PER_CONDITION
        ):
            raise InsufficientDataError(
                f"Statistical comparison requires at least "
                f"{_MIN_SAMPLES_PER_CONDITION} successful experiments per "
                f"condition; {name_a!r} has {len(a_vals)} and {name_b!r} has "
                f"{len(b_vals)}."
            )

        std_a, std_b = float(np.std(a_vals)), float(np.std(b_vals))
        deterministic_a = std_a < _MIN_STD
        deterministic_b = std_b < _MIN_STD

        if deterministic_a and deterministic_b:
            raise InsufficientVarianceError(
                f"Both conditions are deterministic ({name_a!r} is constant at "
                f"{float(np.mean(a_vals)):.6g}, {name_b!r} at "
                f"{float(np.mean(b_vals)):.6g}), so no t-test is defined. "
                f"Vary a hyperparameter, or compare against a condition whose "
                f"result depends on the random seed."
            )

        if deterministic_a or deterministic_b:
            return self._one_sample_comparison(
                a_vals=a_vals,
                b_vals=b_vals,
                constant_is_a=deterministic_a,
                metric=metric,
                name_a=name_a,
                name_b=name_b,
            )

        return self._two_sample_comparison(
            a_vals=a_vals,
            b_vals=b_vals,
            metric=metric,
            name_a=name_a,
            name_b=name_b,
        )

    # ------------------------------------------------------------------
    # Test implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _two_sample_comparison(
        *,
        a_vals: np.ndarray,
        b_vals: np.ndarray,
        metric: str,
        name_a: str,
        name_b: str,
    ) -> StatisticalComparison:
        """Independent-samples t-test - the normal case, both sides vary."""
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
            condition_a_name=name_a,
            condition_b_name=name_b,
            metric=metric,
            test_type="two_sample_t",
            t_statistic=float(t_statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_low), float(ci_high)),
            sample_sizes=(n_a, n_b),
            warning=warning,
        )

    @staticmethod
    def _one_sample_comparison(
        *,
        a_vals: np.ndarray,
        b_vals: np.ndarray,
        constant_is_a: bool,
        metric: str,
        name_a: str,
        name_b: str,
    ) -> StatisticalComparison:
        """One condition is deterministic: test the other against it as a
        known constant (``scipy.stats.ttest_1samp``).

        Results stay oriented "a versus b" regardless of which side is the
        constant - ``t_statistic``, ``effect_size`` and the confidence interval
        all describe ``mean(a) - mean(b)``, exactly as in the two-sample case,
        so nothing downstream needs to special-case the sign.
        """
        varying = b_vals if constant_is_a else a_vals
        constant = float(np.mean(a_vals if constant_is_a else b_vals))
        # +1 when the varying side is A, so (varying - constant) already reads
        # as (a - b); -1 when it is B, and the difference must be flipped.
        sign = -1.0 if constant_is_a else 1.0

        t_raw, p_value = stats.ttest_1samp(varying, constant)

        n = len(varying)
        mean_diff = sign * (float(np.mean(varying)) - constant)
        # Sample std (ddof=1) - the conventional denominator for a one-sample
        # t-test's standard error and for Cohen's d against a fixed value.
        sd = float(np.std(varying, ddof=1))
        effect_size = mean_diff / sd
        se = sd / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(
            _CONFIDENCE_LEVEL, n - 1, loc=mean_diff, scale=se
        )

        # Only the varying side contributes replicates to the test's power; the
        # constant side adds no information however many times it is rerun.
        warning = "underpowered" if n < _UNDERPOWERED_THRESHOLD else None

        return StatisticalComparison(
            condition_a_name=name_a,
            condition_b_name=name_b,
            metric=metric,
            test_type="one_sample_t",
            t_statistic=float(sign * t_raw),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_low), float(ci_high)),
            sample_sizes=(len(a_vals), len(b_vals)),
            warning=warning,
        )

    def summarize_condition(
        self,
        experiments: List[ExperimentResult],
        metric: str,
        condition_name: str,
    ) -> Optional[ConditionSummary]:
        """Descriptive statistics plus replicate accounting for one condition.

        Returns ``None`` when the condition has no successful replicates yet
        (nothing to describe). Supplied to the Recommender so it never has to
        count rows out of a JSON blob - it was previously getting those counts
        wrong in its prose - and shown in the UI so a human can see each
        condition even when no pairwise comparison was possible.
        """
        values = _successful_values(experiments, metric)
        if len(values) == 0:
            return None
        std = float(np.std(values))
        return ConditionSummary(
            condition_name=condition_name,
            metric=metric,
            n_successful=len(values),
            n_anomalous=sum(1 for e in experiments if e.status == "anomalous"),
            n_failed=sum(1 for e in experiments if e.status == "failed"),
            mean=float(np.mean(values)),
            std=std,
            min=float(np.min(values)),
            max=float(np.max(values)),
            # >1 identical replicate is what demonstrates seed-independence; a
            # single observation has std 0 trivially and proves nothing.
            deterministic=std < _MIN_STD and len(values) >= _MIN_SAMPLES_PER_CONDITION,
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
