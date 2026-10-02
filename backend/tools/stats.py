"""
backend/tools/stats.py
========================
The statistics. Pure numpy, no LLM, nothing stored - every call recomputes
from the experiments.

The question it answers: *does level X of the factor perform better than the
reference level on this dataset?*

Method: paired bootstrap over evaluation rows
---------------------------------------------
Every run scores each row of the same validation (or test) split: 1/0 for a
correct/incorrect classification, squared error for regression. For one
condition, those per-row scores are averaged over its seeds, which smooths out
initialisation noise. Then, for a level and the reference:

    d_i = score_level(row i) - score_reference(row i)      (paired by row)
    diff = mean(d)                                          (= accuracy/MSE difference)
    resample the rows with replacement N_BOOTSTRAP times -> 95% percentile CI

So the sample is the evaluation data, not the handful of seeds. The interval
reflects how much the difference could move on different rows drawn from the
same population, which is the dominant source of uncertainty here. A
seed-independent model (linear_baseline) needs no special case: it just has
one run.

Verdict: "better" when the whole CI lies on the improving side of zero,
"worse" when it lies on the other side, otherwise "inconclusive".

Limitation (deliberate): this conditions on the one training split, so it does
not capture variability from retraining on different data. That would need
k-fold cross-validation, which is out of scope.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentResult
from backend.models.investigation import Analysis, Comparison, ConditionSummary, Level, Plan

N_BOOTSTRAP = 2000
CONFIDENCE = 0.95
_RESAMPLE_BUDGET = 2_000_000  # max (resamples x rows) held in memory at once


def metric_for(task_type: str) -> Tuple[str, bool]:
    """(metric name, higher is better) for a task type."""
    return ("accuracy", True) if task_type == "classification" else ("mse", False)


def majority_rate(profile: DatasetProfile) -> Optional[float]:
    """Accuracy of always predicting the most common class (classification only).

    The sanity floor: a classifier that does not beat this has learned nothing
    useful, however far above 1/n_classes it is.
    """
    if profile.task_type != "classification" or not profile.class_distribution:
        return None
    counts = profile.class_distribution.values()
    return max(counts) / sum(counts)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int = 0) -> Tuple[float, float, float]:
    """Mean paired difference ``a - b`` and its bootstrap CI, resampling rows."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOTSTRAP)
    chunk = max(1, _RESAMPLE_BUDGET // n)
    for start in range(0, N_BOOTSTRAP, chunk):
        size = min(chunk, N_BOOTSTRAP - start)
        means[start:start + size] = d[rng.integers(0, n, size=(size, n))].mean(axis=1)
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(d.mean()), float(low), float(high)


def verdict(low: float, high: float, higher_is_better: bool) -> str:
    if low > 0:
        return "better" if higher_is_better else "worse"
    if high < 0:
        return "worse" if higher_is_better else "better"
    return "inconclusive"


def _scores(result: ExperimentResult, split: str) -> List[float]:
    return result.val_scores if split == "val" else result.test_scores  # type: ignore[return-value]


def group_by_level(plan: Plan, results: Sequence[ExperimentResult]) -> Dict[str, List[ExperimentResult]]:
    """Results keyed by level label, for every level in the plan."""
    groups: Dict[str, List[ExperimentResult]] = {plan.label(v): [] for v in plan.levels}
    for r in results:
        label = plan.label(plan.level_of(r.config))
        if label in groups:
            groups[label].append(r)
    return groups


def condition_rows(results: Sequence[ExperimentResult], split: str) -> Optional[np.ndarray]:
    """Per-row scores averaged over a condition's successful runs (None if none)."""
    ok = [_scores(r, split) for r in results if r.status == "ok"]
    if not ok:
        return None
    return np.mean(np.array(ok, dtype=float), axis=0)


def compare(
    plan: Plan,
    level: Level,
    groups: Dict[str, List[ExperimentResult]],
    split: str,
    higher_is_better: bool,
) -> Optional[Comparison]:
    """``level`` vs the plan's reference on ``split``; None if either has no successful run."""
    rows = condition_rows(groups[plan.label(level)], split)
    ref_rows = condition_rows(groups[plan.label(plan.reference)], split)
    if rows is None or ref_rows is None:
        return None
    diff, low, high = bootstrap_diff(rows, ref_rows)
    return Comparison(
        level=level,
        label=plan.label(level),
        diff=diff,
        ci_low=low,
        ci_high=high,
        verdict=verdict(low, high, higher_is_better),
    )


def analyze(
    plan: Plan,
    results: Sequence[ExperimentResult],
    profile: DatasetProfile,
    split: str = "val",
) -> Analysis:
    """Per-level summaries, and every level compared against the reference."""
    metric, higher_is_better = metric_for(profile.task_type)
    groups = group_by_level(plan, results)

    conditions = []
    for level in plan.levels:
        runs = groups[plan.label(level)]
        per_run = [float(np.mean(_scores(r, split))) for r in runs if r.status == "ok"]
        conditions.append(
            ConditionSummary(
                level=level,
                label=plan.label(level),
                is_reference=level == plan.reference,
                n_ok=len(per_run),
                n_failed=sum(1 for r in runs if r.status == "failed"),
                mean=float(np.mean(per_run)) if per_run else None,
                seed_std=float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None,
            )
        )

    comparisons = [
        c
        for level in plan.levels
        if level != plan.reference
        for c in [compare(plan, level, groups, split, higher_is_better)]
        if c is not None
    ]

    any_ok = next((r for r in results if r.status == "ok"), None)
    return Analysis(
        split=split,  # type: ignore[arg-type]
        metric=metric,  # type: ignore[arg-type]
        higher_is_better=higher_is_better,
        n_rows=len(_scores(any_ok, split)) if any_ok else 0,
        majority_rate=majority_rate(profile),
        conditions=conditions,
        comparisons=comparisons,
    )
    

def best_challenger(plan: Plan, analysis: Analysis) -> Optional[Level]:
    """The non-reference level with the best validation mean (None if none ran)."""
    candidates = [c for c in analysis.conditions if not c.is_reference and c.mean is not None]
    if not candidates:
        return None
    pick = max if analysis.higher_is_better else min
    return pick(candidates, key=lambda c: c.mean).level  # type: ignore[arg-type, return-value]
