"""Tests for the Planner and Recommender agents, with a scripted LLM."""

from __future__ import annotations

import json

import pytest

from backend.agents.llm import LLMError
from backend.agents.planner import Planner, PlanningError, describe_dataset
from backend.agents.recommender import Recommender
from backend.models.experiment import ALL_KNOBS
from backend.models.investigation import Comparison, Report, ScoreCI
from backend.tools import stats
from tests._fakes import StubLLM, dropout_plan, make_profile, scores_with_accuracy, selection_plan
from tests.unit.test_stats import _result, _selection_results

_HP_NULL = {k: None for k in ALL_KNOBS}


def _plan_reply(**overrides):
    reply = {
        "error": None,
        "mode": "effect",
        "families": [],
        "factor": "dropout",
        "levels": ["0", "0.2", "0.5"],
        "reference": "0",
        "base": {"model_type": "mlp", "normalize": None, "hyperparameters": {**_HP_NULL, "epochs": 10}},
        "rationale": "Dropout is the factor in question.",
    }
    reply.update(overrides)
    return reply


def _selection_reply(families=("mlp", "random_forest", "linear_baseline", "decision_tree")):
    return _plan_reply(mode="selection", families=list(families), factor=None, levels=[], reference=None,
                       rationale="Compare all four families.")


# ---------------------------------------------------------------------- planner
def test_planner_builds_an_effect_plan() -> None:
    plan = Planner(StubLLM(_plan_reply())).plan("Does dropout help?", make_profile())
    assert plan.mode == "effect" and plan.factor == "dropout"
    assert [c.value for c in plan.candidates] == [0.0, 0.2, 0.5] and plan.reference == "c1"
    assert plan.base.hyperparameters == {"epochs": 10.0}  # nulls dropped
    assert plan.candidates[0].normalize is True  # null -> the mlp family default


def test_planner_builds_a_selection_plan_with_the_simplest_family_as_reference() -> None:
    plan = Planner(StubLLM(_selection_reply())).plan("Which model is best?", make_profile())
    assert plan.mode == "selection"
    assert [c.model_type for c in plan.candidates] == ["linear_baseline", "decision_tree", "random_forest", "mlp"]
    assert plan.get(plan.reference).model_type == "linear_baseline"


def test_planner_feeds_a_rule_violation_back_once() -> None:
    llm = StubLLM(_plan_reply(levels=["0", "1.5"]), _plan_reply())
    plan = Planner(llm).plan("q", make_profile())
    assert [c.value for c in plan.candidates] == [0.0, 0.2, 0.5]
    assert len(llm.requests) == 2
    assert "invalid" in llm.requests[1][-1]["content"]


def test_planner_rejects_a_single_family_selection() -> None:
    llm = StubLLM(_selection_reply(families=["mlp"]), _selection_reply())
    assert len(Planner(llm).plan("q", make_profile()).candidates) == 4
    assert "distinct families" in llm.requests[1][-1]["content"]


def test_planner_gives_up_with_llm_error() -> None:
    with pytest.raises(LLMError, match="after 2 attempts"):
        Planner(StubLLM(_plan_reply(levels=["0", "lots"]))).plan("q", make_profile())


def test_planner_rejects_too_many_levels() -> None:
    llm = StubLLM(_plan_reply(levels=[str(v / 10) for v in range(7)]), _plan_reply())
    assert len(Planner(llm).plan("q", make_profile()).candidates) == 3


def test_unanswerable_question_raises_planning_error_without_retry() -> None:
    llm = StubLLM(_plan_reply(error="This needs a convolutional network."))
    with pytest.raises(PlanningError, match="convolutional"):
        Planner(llm).plan("Do CNNs help?", make_profile())
    assert len(llm.requests) == 1


def test_the_dataset_description_is_counts_only_and_bounded() -> None:
    profile = make_profile().model_copy(update={
        "n_classes": 40, "class_distribution": {f"class_{i}": 100 - i for i in range(40)}})
    text = describe_dataset(profile)
    assert "class_0" in text and "class_9" in text and "class_10" not in text and "and 30 more" in text
    assert "'a'" not in text  # feature column names are never sent


# ------------------------------------------------------------------ recommender
def _effect_analysis():
    plan = dropout_plan()
    results = [_result(plan, "c1", s, accuracy=0.6) for s in range(2)] + [_result(plan, "c2", 0, accuracy=0.62)]
    return plan, stats.analyze(plan, results, make_profile())


def _selection_analysis():
    plan = selection_plan()
    results = _selection_results(plan, {"c1": 0.60, "c2": 0.55, "c3": 0.80, "c4": 0.79})
    return plan, stats.analyze(plan, results, make_profile())


def test_recommender_refines_the_factor_in_effect_mode() -> None:
    plan, analysis = _effect_analysis()
    llm = StubLLM({"action": "refine", "parent": None, "knob": None, "values": ["0.2", "0.3"], "rationale": "between"})
    decision = Recommender(llm).decide("q", plan, analysis, round=1, max_rounds=3)
    assert decision.action == "refine" and decision.values == [0.2, 0.3]
    assert (decision.parent, decision.knob, decision.round, decision.decided_by) == ("c1", "dropout", 1, "agent")
    sent = json.loads(llm.requests[0][1]["content"])
    assert sent["rounds_left"] == 2 and sent["refinable"] == {"dropout": [0.0, 1.0]}


def test_recommender_rejects_already_tried_levels() -> None:
    plan, analysis = _effect_analysis()
    llm = StubLLM(
        {"action": "refine", "parent": None, "knob": None, "values": ["0.5"], "rationale": "again"},
        {"action": "conclude", "parent": None, "knob": None, "values": [], "rationale": "enough"},
    )
    decision = Recommender(llm).decide("q", plan, analysis, 1, 3)
    assert decision.action == "conclude"
    assert "already tried" in llm.requests[1][-1]["content"]


def test_recommender_rejects_out_of_range_levels() -> None:
    plan, analysis = _effect_analysis()
    llm = StubLLM({"action": "refine", "parent": None, "knob": None, "values": ["2.0"], "rationale": "big"})
    with pytest.raises(LLMError):
        Recommender(llm).decide("q", plan, analysis, 1, 3)


def test_recommender_refines_one_knob_of_a_contender() -> None:
    plan, analysis = _selection_analysis()
    assert set(analysis.contenders) == {"c3", "c4"}
    llm = StubLLM({"action": "refine", "parent": "c4", "knob": "hidden_size", "values": ["128"], "rationale": "x"})
    decision = Recommender(llm).decide("q", plan, analysis, 1, 3)
    assert (decision.parent, decision.knob, decision.values) == ("c4", "hidden_size", [128.0])


@pytest.mark.parametrize(
    "parent, knob, message",
    [("c2", "max_depth", "not a contender"), ("c3", "hidden_size", "can refine only")],
)
def test_recommender_rule_violations_are_fed_back(parent, knob, message) -> None:
    plan, analysis = _selection_analysis()
    llm = StubLLM({"action": "refine", "parent": parent, "knob": knob, "values": ["4"], "rationale": "x"},
                  {"action": "conclude", "parent": None, "knob": None, "values": [], "rationale": "ok"})
    assert Recommender(llm).decide("q", plan, analysis, 1, 3).action == "conclude"
    assert message in llm.requests[1][-1]["content"]


def test_recommender_conclude_ignores_values() -> None:
    plan, analysis = _effect_analysis()
    llm = StubLLM({"action": "conclude", "parent": "c1", "knob": "dropout", "values": ["0.3"], "rationale": "done"})
    decision = Recommender(llm).decide("q", plan, analysis, 2, 3)
    assert decision.action == "conclude" and decision.values == []


def test_the_evidence_is_compact_and_bounded() -> None:
    """At the most candidates the budget allows (4 + 2 x 3), the decide prompt stays small."""
    plan = selection_plan()
    for knob, values in (("hidden_size", [16, 32, 128]), ("dropout", [0.1, 0.2, 0.3])):
        plan = plan.with_candidates(plan.refine("c4", knob, values, 2))
    results = [_result(plan, c.id, 0, val=scores_with_accuracy(0.8, seed=i), train=0.9)
               for i, c in enumerate(plan.candidates)]
    analysis = stats.analyze(plan, results, make_profile())

    llm = StubLLM({"action": "conclude", "parent": None, "knob": None, "values": [], "rationale": "ok"})
    Recommender(llm).decide("q" * 300, plan, analysis, 3, 3)
    system, user = (m["content"] for m in llm.requests[0])
    payload = json.loads(user)
    assert len(payload["candidates"]) == 10
    assert "val_scores" not in user and "test" not in user  # aggregates only; the test split never appears
    assert ": " not in user and "\n" not in user              # compact JSON
    # about 3.5 characters per token: well under 2,000 tokens even with a 300-character question
    assert (len(system) + len(user)) / 3.5 < 2000


def test_interpret_returns_the_text_with_low_reasoning_effort() -> None:
    comparison = Comparison(a="c2", label="dropout=0.5", b="c1", against="dropout=0", anchor="reference",
                            diff=0.02, ci_low=-0.01, ci_high=0.05, verdict="inconclusive")
    score = ScoreCI(value=0.8, ci_low=0.75, ci_high=0.85)
    report = Report(mode="effect", winner="dropout=0.5", winner_id="c2", reference="dropout=0", metric="accuracy",
                    higher_is_better=True, winner_score=score, reference_score=score, primary=comparison,
                    confidence=0.95, winner_val_score=0.81, val_to_test_drop=0.01, effort={"mlp": 2},
                    candidates_tried=2, n_test_rows=200, rounds_run=1, stopped_by="agent", headline="h")
    seen = []

    class Recording(StubLLM):
        def chat_json(self, messages, schema, temperature=0.2, **options):
            seen.append(options)
            return super().chat_json(messages, schema, temperature)

    llm = Recording({"interpretation": "  No reliable difference.  "})
    assert Recommender(llm).interpret("q", dropout_plan(), report) == "No reliable difference."
    assert seen == [{"reasoning_effort": "low"}]
