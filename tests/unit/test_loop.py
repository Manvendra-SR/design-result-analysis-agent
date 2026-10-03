"""Tests for the investigation loop (backend/state_machine/graph.py)."""

from __future__ import annotations

from backend.agents.llm import LLMError
from backend.agents.planner import PlanningError
from backend.tools import stats
from backend.state_machine.graph import configs_for, run_investigation
from tests._fakes import FakePlanner, FakeRecommender, dropout_plan, fake_run, make_profile, selection_plan

# Validation/test accuracy per family in the selection fake: the forest and the
# MLP are close, the linear model and the tree are clearly behind.
FAMILY_ACCURACY = {"linear_baseline": 0.55, "decision_tree": 0.5, "random_forest": 0.8, "mlp": 0.78}


def _family_accuracy(cfg):
    return FAMILY_ACCURACY[cfg.model_type] + 0.001 * cfg.hyperparameters.get("min_samples_leaf", 1)


def _family_seed(cfg):
    """Independent draws per family; a family's candidates share theirs (so they rank by accuracy)."""
    return cfg.random_seed + 100 * list(FAMILY_ACCURACY).index(cfg.model_type)


def _selection_run():
    return fake_run(accuracy_of=_family_accuracy, seed_of=_family_seed)


def _setup(repo, question="Does dropout help?"):
    repo.create_dataset(make_profile("ds-1"))
    return repo.create_session(question, "ds-1").session_id


def _run(repo, sid, planner=None, recommender=None, max_rounds=3, run=None):
    run_investigation(sid, repo, planner or FakePlanner(), recommender or FakeRecommender(),
                      run=run or fake_run(), max_rounds=max_rounds, n_seeds=3)
    return repo.get_session(sid)


def test_seeds_per_family() -> None:
    plan = selection_plan()
    configs = configs_for(plan.candidates, "ds", n_seeds=3)
    assert [(c.model_type, c.random_seed) for c in configs] == [
        ("linear_baseline", 0), ("decision_tree", 0),  # seed-independent: one run
        ("random_forest", 0), ("random_forest", 1), ("random_forest", 2),
        ("mlp", 0), ("mlp", 1), ("mlp", 2),
    ]


# ------------------------------------------------------------------ effect mode
def test_effect_refine_then_conclude(repo) -> None:
    sid = _setup(repo)
    recommender = FakeRecommender(refine=[[0.2]])
    session = _run(repo, sid, recommender=recommender)

    assert session.status == "done", session.error
    assert [c.label for c in session.plan.candidates] == ["dropout=0", "dropout=0.5", "dropout=0.2"]
    assert [(d.round, d.action, d.decided_by, d.new_candidates) for d in session.decisions] == [
        (1, "refine", "agent", ["c3"]), (2, "conclude", "agent", []),
    ]
    rounds = [e.round for e in repo.list_experiments(sid)]
    assert rounds == [1] * 6 + [2] * 3

    report = session.report
    assert report.mode == "effect" and report.secondary is None and report.confidence == 0.95
    assert report.winner == "dropout=0.5" and report.reference == "dropout=0"  # best on validation
    assert report.primary.verdict == "better"
    assert report.rounds_run == 2 and report.stopped_by == "agent"
    assert report.headline.startswith("On the held-out test set (200 rows), dropout=0.5 performs better")
    assert "(95% CI" in report.headline
    assert report.effort == {"mlp": 3} and report.candidates_tried == 3
    assert report.interpretation == "It helps."


def test_round_budget_forces_a_conclusion_without_asking_the_llm(repo) -> None:
    sid = _setup(repo)
    recommender = FakeRecommender(refine=[[0.1], [0.2], [0.3]])
    session = _run(repo, sid, recommender=recommender, max_rounds=2)

    assert recommender.decide_calls == [1]  # never called on the last round
    assert session.decisions[-1].decided_by == "budget"
    assert session.report.stopped_by == "budget" and session.report.rounds_run == 2


def test_an_invalid_refinement_fails_the_session_instead_of_running_it(repo) -> None:
    sid = _setup(repo)
    session = _run(repo, sid, recommender=FakeRecommender(refine=[[0.5]]))  # already tried
    assert session.status == "failed" and "already tried" in session.error
    assert len(repo.list_experiments(sid)) == 6  # nothing extra was trained


def test_reference_can_be_the_best_level(repo) -> None:
    run = fake_run(accuracy_of=lambda cfg: 0.9 if cfg.hyperparameters["dropout"] == 0.0 else 0.6)
    session = _run(repo, _setup(repo), planner=FakePlanner(dropout_plan(levels=(0.0, 0.3, 0.6))), run=run)
    assert session.report.primary.verdict == "worse"
    assert session.report.headline.startswith(
        "dropout=0 was the best candidate on validation, so its strongest alternative was tested.")
    assert "performs worse than dropout=0" in session.report.headline


# --------------------------------------------------------------- selection mode
def test_selection_compares_families_then_refines_contenders(repo) -> None:
    sid = _setup(repo, "Which model is best?")
    recommender = FakeRecommender(refine=[("c3", "min_samples_leaf", [5.0, 20.0])])
    session = _run(repo, sid, planner=FakePlanner(selection_plan()), recommender=recommender,
                   run=_selection_run())
    assert session.status == "done", session.error

    first = recommender.seen[0]
    assert first.leader == "c3" and set(first.contenders) == {"c3", "c4"}  # linear and tree are out
    assert [c.label for c in session.plan.candidates[4:]] == [
        "random_forest, min_samples_leaf=5", "random_forest, min_samples_leaf=20"]
    runs = repo.list_experiments(sid)
    assert len(runs) == 8 + 2 * 3  # round 1: 1 + 1 + 3 + 3; round 2: two forest children x 3 seeds

    report = session.report
    assert report.mode == "selection" and report.winner_id == "c6"
    assert report.runner_up == "mlp" and report.secondary is not None
    assert report.confidence == 0.975 and report.primary.confidence == 0.975  # two claims, Bonferroni
    assert "97.5% CI" in report.headline and "strongest rival" in report.headline
    assert report.effort == {"linear_baseline": 1, "decision_tree": 1, "random_forest": 3, "mlp": 1}
    assert report.winner_score.ci_low <= report.winner_score.value <= report.winner_score.ci_high


def test_settled_contenders_conclude_without_an_llm_call(repo) -> None:
    sid = _setup(repo, "Which model is best?")
    recommender = FakeRecommender(refine=[("c3", "max_depth", [5.0])])
    clear = {"linear_baseline": 0.5, "decision_tree": 0.5, "random_forest": 0.95, "mlp": 0.5}
    session = _run(repo, sid, planner=FakePlanner(selection_plan()), recommender=recommender,
                   run=fake_run(accuracy_of=lambda cfg: clear[cfg.model_type], seed_of=_family_seed))
    assert recommender.decide_calls == []
    assert [(d.action, d.decided_by) for d in session.decisions] == [("conclude", "settled")]
    assert session.report.stopped_by == "settled"


def test_hard_limits_bound_runs_and_llm_calls(repo) -> None:
    """An agent that always proposes the most it may still cannot exceed the budget."""
    sid = _setup(repo, "Which model is best?")
    planner = FakePlanner(selection_plan())  # identical scores for every candidate: everyone stays a contender
    recommender = FakeRecommender(refine=[("c4", "hidden_size", [16.0, 32.0, 128.0]),
                                          ("c4", "dropout", [0.1, 0.2, 0.3]),
                                          ("c4", "epochs", [5.0, 10.0, 30.0])])
    session = _run(repo, sid, planner=planner, recommender=recommender,
                   run=fake_run(accuracy_of=lambda cfg: 0.8))
    assert session.status == "done", session.error
    llm_calls = planner.calls + len(recommender.decide_calls) + recommender.interpret_calls
    assert llm_calls == 4  # plan + 2 decisions + interpretation = MAX_ROUNDS + 1
    assert len(session.plan.candidates) == 4 + 2 * 3
    assert len(repo.list_experiments(sid)) == 26  # 8 + 2 rounds x 3 candidates x 3 seeds


# ------------------------------------------------------------------- both modes
def test_test_split_is_read_only_after_the_last_decision(repo, monkeypatch) -> None:
    reads = []
    real_scores = stats._scores
    monkeypatch.setattr(stats, "_scores", lambda r, split: reads.append(split) or real_scores(r, split))

    class Spy(FakeRecommender):
        def decide(self, *args):
            reads.append("decide")
            return super().decide(*args)

        def interpret(self, *args):
            reads.append("interpret")
            return super().interpret(*args)

    sid = _setup(repo, "Which model is best?")
    _run(repo, sid, planner=FakePlanner(selection_plan()), recommender=Spy(refine=[("c4", "dropout", [0.2])]),
         run=_selection_run())
    first_test = reads.index("test")
    assert "decide" in reads and max(i for i, r in enumerate(reads) if r == "decide") < first_test
    assert "val" not in reads[first_test:reads.index("interpret")]  # finalize reads test only
    assert "test" not in reads[reads.index("interpret"):]


def test_failed_runs_do_not_stop_the_investigation(repo) -> None:
    def flaky(config, profile, session_id, round):
        result = fake_run()(config, profile, session_id, round)
        if config.random_seed == 1:
            return result.model_copy(update={"status": "failed", "error": "nan", "metrics": None,
                                             "val_scores": None, "test_scores": None})
        return result

    session = _run(repo, _setup(repo), run=flaky)
    assert session.status == "done"
    assert sum(e.status == "failed" for e in repo.list_experiments(session.session_id)) == 2


def test_planning_error_marks_the_session_failed(repo) -> None:
    session = _run(repo, _setup(repo), planner=FakePlanner(error=PlanningError("Needs a CNN.")))
    assert session.status == "failed" and "Needs a CNN." in session.error
    assert session.report is None


def test_nothing_to_compare_fails_clearly(repo) -> None:
    def always_fail(config, profile, session_id, round):
        return fake_run()(config, profile, session_id, round).model_copy(
            update={"status": "failed", "error": "x", "metrics": None, "val_scores": None, "test_scores": None})

    session = _run(repo, _setup(repo), run=always_fail)
    assert session.status == "failed" and "nothing to compare" in session.error


def test_missing_interpretation_still_produces_a_report(repo) -> None:
    session = _run(repo, _setup(repo), recommender=FakeRecommender(interpretation=LLMError("rate limited")))
    assert session.status == "done"
    assert session.report.interpretation is None and session.report.headline
