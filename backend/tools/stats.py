"""
backend/tools/stats.py
========================
The statistics. Pure numpy, no LLM, nothing stored - every call recomputes
from the experiments.

Method: paired bootstrap over evaluation rows
---------------------------------------------
Every run scores each row of the same validation (or test) split: 1/0 for a
correct/incorrect classification, squared error for regression. For one
candidate, those per-row scores are averaged over its seeds, which smooths out
initialisation noise. Then, for any two candidates a and b:

    d_i = score_a(row i) - score_b(row i)                  (paired by row)
    diff = mean(d)                                          (= accuracy/MSE difference)
    resample the rows with replacement N_BOOTSTRAP times -> percentile CI

So the sample is the evaluation data, not the handful of seeds. A
seed-independent model (linear, decision tree) needs no special case: it just
has one run.

Verdict: "better" when the whole CI lies on the improving side of zero,
"worse" when it lies on the other side, otherwise "inconclusive".

What is compared with what
--------------------------
On VALIDATION (exploratory, steers the loop - never a claim):
    vs reference   every candidate vs the reference (does it beat simple?)
    vs leader      selection mode: every candidate vs the best one so far;
                   a candidate whose CI still includes zero is a *contender*
    vs parent      selection mode: a refined child vs the candidate it came
                   from (the effect of the one knob it changed)

On TEST (confirmatory, read once by ``final_comparisons``): the winner vs the
reference and, in selection mode, vs the best candidate of another family -
both chosen from validation before test is read. With k comparisons each CI
is at 1 - 0.05/k (Bonferroni), so the two claims share the 5% error budget.

Limitation (deliberate): this conditions on the one training split, so it does
not capture variability from retraining on different data. That would need
k-fold cross-validation, which is out of scope.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentResult
from backend.models.investigation import Analysis, Comparison, ConditionSummary, Plan, ScoreCI

N_BOOTSTRAP = 2000
CONFIDENCE = 0.95
_ALPHA = 1 - CONFIDENCE
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


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int = 0, confidence: float = CONFIDENCE) -> Tuple[float, float, float]:
    """Mean paired difference ``a - b`` and its bootstrap CI, resampling rows."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOTSTRAP)
    chunk = max(1, _RESAMPLE_BUDGET // n)
    for start in range(0, N_BOOTSTRAP, chunk):
        size = min(chunk, N_BOOTSTRAP - start)
        means[start:start + size] = d[rng.integers(0, n, size=(size, n))].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
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


def group_by_candidate(plan: Plan, results: Sequence[ExperimentResult]) -> Dict[str, List[ExperimentResult]]:
    """Results keyed by candidate id (via the seed-free configuration key), for every candidate."""
    groups: Dict[str, List[ExperimentResult]] = {c.id: [] for c in plan.candidates}
    by_key = {c.config("-").key(): c.id for c in plan.candidates}
    for r in results:
        candidate_id = by_key.get(r.config.key())
        if candidate_id is not None:
            groups[candidate_id].append(r)
    return groups


def condition_rows(results: Sequence[ExperimentResult], split: str) -> Optional[np.ndarray]:
    """Per-row scores averaged over a candidate's successful runs (None if none)."""
    ok = [_scores(r, split) for r in results if r.status == "ok"]
    if not ok:
        return None
    return np.mean(np.array(ok, dtype=float), axis=0)


def compare(
    plan: Plan,
    a: str,
    b: str,
    groups: Dict[str, List[ExperimentResult]],
    split: str,
    higher_is_better: bool,
    anchor: str = "reference",
    confidence: float = CONFIDENCE,
) -> Optional[Comparison]:
    """Candidate ``a`` vs candidate ``b`` on ``split``; None if either has no successful run."""
    rows, other = condition_rows(groups[a], split), condition_rows(groups[b], split)
    if rows is None or other is None:
        return None
    diff, low, high = bootstrap_diff(rows, other, confidence=confidence)
    return Comparison(
        a=a, label=plan.get(a).label, b=b, against=plan.get(b).label, anchor=anchor,  # type: ignore[arg-type]
        diff=diff, ci_low=low, ci_high=high, confidence=confidence,
        verdict=verdict(low, high, higher_is_better),  # type: ignore[arg-type]
    )


def score_ci(rows: np.ndarray, confidence: float = CONFIDENCE) -> ScoreCI:
    """The mean of per-row scores with its bootstrap CI (the paired bootstrap against zero)."""
    value, low, high = bootstrap_diff(rows, np.zeros_like(rows), confidence=confidence)
    return ScoreCI(value=value, ci_low=low, ci_high=high)


def analyze(
    plan: Plan,
    results: Sequence[ExperimentResult],
    profile: DatasetProfile,
    split: str = "val",
) -> Analysis:
    """Per-candidate summaries and the comparisons that steer the loop."""
    metric, higher_is_better = metric_for(profile.task_type)
    groups = group_by_candidate(plan, results)
    sign = 1.0 if higher_is_better else -1.0

    conditions = []
    for c in plan.candidates:
        runs = groups[c.id]
        ok = [r for r in runs if r.status == "ok"]
        per_run = [float(np.mean(_scores(r, split))) for r in ok]
        train = [r.metrics[f"train_{metric}"] for r in ok if r.metrics and f"train_{metric}" in r.metrics]
        mean = float(np.mean(per_run)) if per_run else None
        train_metric = float(np.mean(train)) if train else None
        conditions.append(
            ConditionSummary(
                id=c.id, label=c.label, family=c.model_type, parent=c.parent, change=c.change,
                is_reference=c.id == plan.reference,
                n_ok=len(per_run),
                n_failed=sum(1 for r in runs if r.status == "failed"),
                mean=mean,
                seed_std=float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None,
                train_metric=train_metric,
                gap=sign * (train_metric - mean) if mean is not None and train_metric is not None else None,
            )
        )

    measured = [c for c in conditions if c.mean is not None]
    leader = max(measured, key=lambda c: sign * c.mean).id if measured else None  # type: ignore[operator]

    def pairs():
        for c in plan.candidates:
            if c.id != plan.reference:
                yield c.id, plan.reference, "reference"
        if plan.mode == "selection" and leader is not None:
            for c in plan.candidates:
                if c.id != leader:
                    yield c.id, leader, "leader"
            for c in plan.candidates:
                if c.parent is not None and c.parent != plan.reference:
                    yield c.id, c.parent, "parent"

    comparisons = [
        cmp for a, b, anchor in pairs()
        for cmp in [compare(plan, a, b, groups, split, higher_is_better, anchor)] if cmp is not None
    ]

    contenders: List[str] = []
    if plan.mode == "selection" and leader is not None:
        clearly_worse = {c.a for c in comparisons if c.anchor == "leader" and c.verdict == "worse"}
        contenders = [c.id for c in measured if c.id not in clearly_worse]
        for cond in conditions:
            cond.contender = cond.id in contenders

    any_ok = next((r for r in results if r.status == "ok"), None)
    return Analysis(
        split=split,  # type: ignore[arg-type]
        metric=metric,  # type: ignore[arg-type]
        higher_is_better=higher_is_better,
        mode=plan.mode,
        n_rows=len(_scores(any_ok, split)) if any_ok else 0,
        majority_rate=majority_rate(profile),
        leader=leader,
        contenders=contenders,
        conditions=conditions,
        comparisons=comparisons,
    )


def settled(plan: Plan, analysis: Analysis) -> bool:
    """Selection mode: every contender belongs to one family, so the ranking question is answered."""
    families = {plan.get(c).model_type for c in analysis.contenders}
    return plan.mode == "selection" and len(families) == 1


def select_final(plan: Plan, analysis: Analysis) -> Tuple[Optional[str], Optional[str]]:
    """(winner, runner-up) from VALIDATION, before the test split is read.

    winner     the best non-reference candidate
    runner-up  selection mode: the best candidate from a family other than the
               winner's and the reference's (None if there is none)
    """
    sign = 1.0 if analysis.higher_is_better else -1.0
    ranked = sorted((c for c in analysis.conditions if c.mean is not None), key=lambda c: -sign * c.mean)  # type: ignore[operator]
    winner = next((c for c in ranked if not c.is_reference), None)
    if winner is None:
        return None, None
    if plan.mode != "selection":
        return winner.id, None
    excluded = {winner.family, plan.get(plan.reference).model_type}
    runner_up = next((c for c in ranked if c.family not in excluded), None)
    return winner.id, runner_up.id if runner_up else None


def final_comparisons(
    plan: Plan,
    results: Sequence[ExperimentResult],
    winner: str,
    runner_up: Optional[str],
    higher_is_better: bool,
) -> Tuple[Optional[Comparison], Optional[Comparison], float]:
    """The confirmatory TEST comparisons, fixed in advance: (primary, secondary, confidence)."""
    k = 2 if runner_up is not None else 1
    confidence = 1 - _ALPHA / k
    groups = group_by_candidate(plan, results)
    primary = compare(plan, winner, plan.reference, groups, "test", higher_is_better, "reference", confidence)
    secondary = (
        compare(plan, winner, runner_up, groups, "test", higher_is_better, "runner_up", confidence)
        if runner_up is not None else None
    )
    return primary, secondary, confidence
