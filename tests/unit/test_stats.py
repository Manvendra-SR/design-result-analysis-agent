"""Tests for backend/tools/stats.py - the paired bootstrap, the validation evidence and the final selection."""

from __future__ import annotations

import numpy as np
import pytest

from backend.models.experiment import ExperimentResult
from backend.models.investigation import BaseSetup, Plan
from backend.tools import stats
from tests._fakes import dropout_plan, make_profile, scores_with_accuracy, selection_plan


def _result(plan: Plan, candidate_id, seed=0, accuracy=0.7, status="ok", val=None, train=None) -> ExperimentResult:
    cfg = plan.get(candidate_id).config("ds-1", seed)
    if status == "failed":
        return ExperimentResult(session_id="s", round=1, config=cfg, status="failed", error="boom")
    val = val if val is not None else scores_with_accuracy(accuracy, seed)
    metrics = {"accuracy": float(np.mean(val))}
    if train is not None:
        metrics["train_accuracy"] = train
    return ExperimentResult(
        session_id="s", round=1, config=cfg, status="ok", metrics=metrics, val_scores=val, test_scores=val,
    )


def _shared(accuracy: float, seed: int = 7, n: int = 400) -> list:
    """Scores built from one shared draw, so candidates differ only where their accuracy differs."""
    base = np.random.default_rng(seed).random(n)
    return (base < accuracy).astype(float).tolist()


# --------------------------------------------------------------------- bootstrap
def test_identical_scores_give_a_zero_width_interval() -> None:
    a = np.array([1.0, 0.0, 1.0, 1.0, 0.0] * 40)
    diff, low, high = stats.bootstrap_diff(a, a.copy())
    assert (diff, low, high) == (0.0, 0.0, 0.0)
    assert stats.verdict(low, high, higher_is_better=True) == "inconclusive"


def test_clear_difference_excludes_zero() -> None:
    a = np.ones(300)
    b = np.array([1.0, 0.0] * 150)
    diff, low, high = stats.bootstrap_diff(a, b)
    assert diff == pytest.approx(0.5)
    assert 0 < low < diff < high
    assert stats.verdict(low, high, higher_is_better=True) == "better"
    assert stats.verdict(low, high, higher_is_better=False) == "worse"  # e.g. higher MSE


def test_interval_covers_the_true_difference() -> None:
    rng = np.random.default_rng(1)
    a = (rng.random(2000) < 0.80).astype(float)
    b = (rng.random(2000) < 0.75).astype(float)
    _, low, high = stats.bootstrap_diff(a, b)
    assert low < 0.05 < high


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.random(500), rng.random(500)
    assert stats.bootstrap_diff(a, b) == stats.bootstrap_diff(a, b)


def test_a_higher_confidence_level_gives_a_wider_interval() -> None:
    rng = np.random.default_rng(4)
    a, b = (rng.random(1000) < 0.75).astype(float), (rng.random(1000) < 0.7).astype(float)
    _, low95, high95 = stats.bootstrap_diff(a, b)
    _, low975, high975 = stats.bootstrap_diff(a, b, confidence=0.975)
    assert low975 < low95 and high975 > high95


def test_noise_level_difference_is_inconclusive() -> None:
    rng = np.random.default_rng(3)
    a = (rng.random(200) < 0.7).astype(float)
    b = a.copy()
    b[:3] = 1 - b[:3]  # three rows differ
    _, low, high = stats.bootstrap_diff(a, b)
    assert stats.verdict(low, high, True) == "inconclusive"


def test_score_ci_brackets_the_mean() -> None:
    rows = np.array(_shared(0.7))
    ci = stats.score_ci(rows)
    assert ci.ci_low < ci.value < ci.ci_high
    assert ci.value == pytest.approx(rows.mean())


# ---------------------------------------------------------------------- analyze
def test_analyze_summarises_each_candidate_and_compares_with_the_reference() -> None:
    plan = dropout_plan(levels=(0.0, 0.5))
    results = [_result(plan, "c1", s, accuracy=0.6, train=0.62) for s in range(3)]
    results += [_result(plan, "c2", s, accuracy=0.95) for s in range(3)]
    analysis = stats.analyze(plan, results, make_profile())

    assert analysis.metric == "accuracy" and analysis.higher_is_better and analysis.mode == "effect"
    assert analysis.n_rows == 200
    assert [c.label for c in analysis.conditions] == ["dropout=0", "dropout=0.5"]
    ref = analysis.conditions[0]
    assert ref.is_reference and ref.n_ok == 3 and ref.seed_std is not None
    assert ref.gap == pytest.approx(0.62 - ref.mean)  # train better than validation -> positive
    [comparison] = analysis.comparisons  # effect mode: only vs the reference
    assert (comparison.a, comparison.b, comparison.anchor) == ("c2", "c1", "reference")
    assert comparison.verdict == "better"
    assert comparison.diff == pytest.approx(analysis.conditions[1].mean - ref.mean)
    assert analysis.leader == "c2" and analysis.contenders == []


def test_failed_runs_are_excluded_but_counted() -> None:
    plan = dropout_plan()
    results = [_result(plan, "c1", 0), _result(plan, "c1", 1, status="failed"), _result(plan, "c2", 0)]
    ref = stats.analyze(plan, results, make_profile()).conditions[0]
    assert (ref.n_ok, ref.n_failed) == (1, 1)
    assert ref.seed_std is None  # a single run has no spread


def test_unusual_but_valid_runs_stay_in_the_evidence() -> None:
    plan = dropout_plan()
    results = [_result(plan, "c1", s, accuracy=0.7) for s in range(3)]
    results.append(_result(plan, "c1", 3, val=[0.0] * 200))  # wildly unlike its siblings
    results.append(_result(plan, "c2", 0))
    ref = stats.analyze(plan, results, make_profile()).conditions[0]
    assert ref.n_ok == 4


def test_seed_independent_models_need_no_special_case() -> None:
    plan = selection_plan(families=("linear_baseline", "mlp"))
    results = [_result(plan, "c1", 0, accuracy=0.7)]  # one run
    results += [_result(plan, "c2", s, accuracy=0.72) for s in range(3)]
    analysis = stats.analyze(plan, results, make_profile())
    assert analysis.conditions[0].n_ok == 1 and analysis.conditions[1].n_ok == 3


def test_no_comparison_without_a_successful_reference() -> None:
    plan = dropout_plan()
    results = [_result(plan, "c1", 0, status="failed"), _result(plan, "c2", 0)]
    assert stats.analyze(plan, results, make_profile()).comparisons == []


def test_majority_rate() -> None:
    assert stats.majority_rate(make_profile()) == pytest.approx(0.6)
    assert stats.majority_rate(make_profile(task_type="regression")) is None


def test_regression_uses_mse_and_lower_is_better() -> None:
    plan = Plan.effect("normalize", [False, True], False, BaseSetup(model_type="linear_baseline"), "scale?")
    results = [_result(plan, "c1", 0, val=[4.0] * 100), _result(plan, "c2", 0, val=[1.0] * 100)]
    analysis = stats.analyze(plan, results, make_profile(task_type="regression"))
    assert analysis.metric == "mse" and not analysis.higher_is_better
    assert analysis.comparisons[0].verdict == "better"
    assert analysis.leader == "c2"
    assert stats.select_final(plan, analysis) == ("c2", None)


# ---------------------------------------------------------- selection evidence
def _exact(accuracy: float, seed: int, n: int = 400) -> list:
    """Exactly ``accuracy`` of n rows correct, at positions drawn independently per seed."""
    scores = np.zeros(n)
    scores[np.random.default_rng(seed).permutation(n)[: round(accuracy * n)]] = 1.0
    return scores.tolist()


def _selection_results(plan, accuracies):
    """Independent scores per candidate (identical across its seeds)."""
    return [_result(plan, cid, s, val=_exact(acc, seed=int(cid[1:])))
            for cid, acc in accuracies.items()
            for s in range(3 if plan.get(cid).model_type in ("random_forest", "mlp") else 1)]


def _with_forest_child(plan: Plan) -> Plan:
    return plan.with_candidates(plan.refine("c3", "max_depth", [12], 2, contenders=["c3"]))


def test_leader_contenders_and_parent_comparisons() -> None:
    plan = _with_forest_child(selection_plan())
    # linear 0.70, tree 0.55, forest 0.80, mlp 0.79, forest child 0.81
    results = _selection_results(plan, {"c1": 0.70, "c2": 0.55, "c3": 0.80, "c4": 0.79, "c5": 0.81})
    analysis = stats.analyze(plan, results, make_profile())

    assert analysis.leader == "c5"
    assert analysis.contenders == ["c3", "c4", "c5"]  # the tree and the linear model are clearly worse
    assert [c.id for c in analysis.conditions if c.contender] == ["c3", "c4", "c5"]
    anchors = {(c.a, c.anchor) for c in analysis.comparisons}
    assert {("c5", "parent"), ("c4", "leader"), ("c2", "reference")} <= anchors
    assert ("c3", "parent") not in anchors  # roots have no parent
    assert next(c for c in analysis.conditions if c.id == "c5").change == "max_depth=12"


def test_settled_when_every_contender_is_one_family() -> None:
    plan = selection_plan()
    clear = stats.analyze(plan, _selection_results(plan, {"c1": 0.6, "c2": 0.6, "c3": 0.9, "c4": 0.6}),
                          make_profile())
    assert clear.contenders == ["c3"] and stats.settled(plan, clear)
    close = stats.analyze(plan, _selection_results(plan, {"c1": 0.6, "c2": 0.6, "c3": 0.800, "c4": 0.795}),
                          make_profile())
    assert not stats.settled(plan, close)
    effect = dropout_plan()
    assert not stats.settled(effect, stats.analyze(effect, [_result(effect, "c1"), _result(effect, "c2")],
                                                   make_profile()))


def test_select_final_picks_the_runner_up_from_another_family() -> None:
    plan = _with_forest_child(selection_plan())
    analysis = stats.analyze(
        plan, _selection_results(plan, {"c1": 0.70, "c2": 0.75, "c3": 0.80, "c4": 0.78, "c5": 0.81}),
        make_profile())
    assert stats.select_final(plan, analysis) == ("c5", "c4")  # not c3: the winner's own family

    two = selection_plan(families=("linear_baseline", "mlp"))
    analysis = stats.analyze(two, _selection_results(two, {"c1": 0.7, "c2": 0.8}), make_profile())
    assert stats.select_final(two, analysis) == ("c2", None)  # no third family: one test comparison


def test_select_final_skips_the_reference_even_when_it_leads() -> None:
    plan = selection_plan(families=("linear_baseline", "decision_tree", "mlp"))
    analysis = stats.analyze(plan, _selection_results(plan, {"c1": 0.9, "c2": 0.6, "c3": 0.7}), make_profile())
    assert analysis.leader == "c1"
    assert stats.select_final(plan, analysis) == ("c3", "c2")


def test_final_comparisons_use_bonferroni_for_two_claims() -> None:
    plan = selection_plan()
    results = _selection_results(plan, {"c1": 0.70, "c2": 0.75, "c3": 0.80, "c4": 0.78})
    primary, secondary, confidence = stats.final_comparisons(plan, results, "c3", "c4", True)
    assert confidence == pytest.approx(0.975)
    assert (primary.b, primary.anchor) == ("c1", "reference")
    assert primary.confidence == pytest.approx(0.975)
    assert (secondary.b, secondary.anchor) == ("c4", "runner_up")
    single, none, confidence = stats.final_comparisons(plan, results, "c3", None, True)
    assert none is None and confidence == pytest.approx(0.95)
    assert single.confidence == pytest.approx(0.95)
