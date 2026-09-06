"""
tests/unit/test_planner.py
============================
Unit tests for backend/agents/planner.py.

The LLM is always a stub returning canned strings - no Ollama server is
contacted (design.md "Mocked LLM Testing").

Test cases (task 4.6)
---------------------
1.  test_valid_question_produces_experiment_plan
2.  test_dataset_id_is_forced_onto_every_config
3.  test_unanswerable_question_raises_planning_error
4.  test_malformed_json_raises_after_one_repair_attempt
5.  test_repair_attempt_can_succeed
6.  test_fewer_than_three_seeds_per_condition_raises
7.  test_single_condition_raises
8.  test_out_of_range_hyperparameter_raises
9.  test_empty_research_question_raises
10. test_no_experiments_in_response_raises
11. test_prompt_contains_question_and_dataset_facts
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest

from backend.agents.planner import (
    ExperimentPlannerAgent,
    PlanningError,
    PlanValidationError,
)
from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentPlan


class _StubLLM:
    """Minimal stand-in for OllamaClient: returns canned replies in order."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses) or ["{}"]
        self.calls: List[list] = []

    def chat_completion(self, messages, *, format=None, options=None) -> str:  # noqa: A002
        self.calls.append(messages)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _profile(dataset_id: str = "ds-real") -> DatasetProfile:
    return DatasetProfile(
        dataset_id=dataset_id,
        original_filename="churn.csv",
        storage_path="data/uploads/ds-real/data.csv",
        target_column="churned",
        feature_columns=["tenure", "charges", "plan"],
        numeric_columns=["tenure", "charges"],
        categorical_columns=["plan"],
        task_type="classification",
        n_rows=4200,
        n_features=6,
        n_classes=2,
        class_labels=["no", "yes"],
        class_distribution={"no": 3100, "yes": 1100},
        missing_value_counts={},
        split_seed=42,
    )


def _plan_json(
    dataset_id: str = "ds-real",
    dropouts=(0.0, 0.2),
    seeds=(42, 43, 44),
    explanation: str = "Vary dropout, holding everything else fixed.",
    extra_hp: Optional[dict] = None,
) -> str:
    experiments = []
    for dropout in dropouts:
        hp = {
            "dropout": dropout,
            "learning_rate": 0.001,
            "batch_size": 32,
            "hidden_size": 64,
            "epochs": 10,
        }
        if extra_hp:
            hp.update(extra_hp)
        for seed in seeds:
            experiments.append(
                {
                    "dataset_id": dataset_id,
                    "model_type": "mlp",
                    "hyperparameters": hp,
                    "preprocessing": {"normalize": False},
                    "random_seed": seed,
                }
            )
    return json.dumps({"experiments": experiments, "explanation": explanation})


# ---------------------------------------------------------------------------
# 1-2. Happy path
# ---------------------------------------------------------------------------

def test_valid_question_produces_experiment_plan() -> None:
    agent = ExperimentPlannerAgent(llm_client=_StubLLM(_plan_json()))

    plan = agent.plan_experiments("Does dropout improve accuracy?", _profile())

    assert isinstance(plan, ExperimentPlan)
    assert plan.total_count == 6
    assert plan.explanation
    assert {c.hyperparameters["dropout"] for c in plan.experiments} == {0.0, 0.2}


def test_dataset_id_is_forced_onto_every_config() -> None:
    # LLM hallucinated a wrong id; the agent must overwrite it.
    agent = ExperimentPlannerAgent(llm_client=_StubLLM(_plan_json(dataset_id="WRONG-ID")))

    plan = agent.plan_experiments("Does dropout help?", _profile("ds-real"))

    assert {c.dataset_id for c in plan.experiments} == {"ds-real"}


# ---------------------------------------------------------------------------
# 3-5. Error paths / JSON handling
# ---------------------------------------------------------------------------

def test_unanswerable_question_raises_planning_error() -> None:
    stub = _StubLLM(json.dumps({"error": "This question is not about the dataset."}))
    agent = ExperimentPlannerAgent(llm_client=stub)

    with pytest.raises(PlanningError) as excinfo:
        agent.plan_experiments("Is GPT-4 better than GPT-3?", _profile())
    assert "not about the dataset" in str(excinfo.value)


def test_malformed_json_raises_after_one_repair_attempt() -> None:
    stub = _StubLLM("not json at all", "still not json")
    agent = ExperimentPlannerAgent(llm_client=stub)

    with pytest.raises(PlanningError):
        agent.plan_experiments("Does dropout help?", _profile())
    assert len(stub.calls) == 2  # initial + one repair


def test_repair_attempt_can_succeed() -> None:
    stub = _StubLLM("oops no json", _plan_json())
    agent = ExperimentPlannerAgent(llm_client=stub)

    plan = agent.plan_experiments("Does dropout help?", _profile())
    assert plan.total_count == 6
    assert len(stub.calls) == 2


# ---------------------------------------------------------------------------
# 6-8. Config validation
# ---------------------------------------------------------------------------

def test_fewer_than_three_seeds_per_condition_raises() -> None:
    agent = ExperimentPlannerAgent(llm_client=_StubLLM(_plan_json(seeds=(42, 43))))

    with pytest.raises(PlanValidationError) as excinfo:
        agent.plan_experiments("Does dropout help?", _profile())
    assert "random seed" in str(excinfo.value)


def test_single_condition_raises() -> None:
    # Only one dropout value -> one condition -> nothing to compare.
    agent = ExperimentPlannerAgent(llm_client=_StubLLM(_plan_json(dropouts=(0.2,))))

    with pytest.raises(PlanValidationError) as excinfo:
        agent.plan_experiments("Does dropout help?", _profile())
    assert "at least 2 conditions" in str(excinfo.value)


def test_out_of_range_hyperparameter_raises() -> None:
    agent = ExperimentPlannerAgent(llm_client=_StubLLM(_plan_json(dropouts=(0.0, 1.5))))

    with pytest.raises(PlanValidationError):
        agent.plan_experiments("Does dropout help?", _profile())


def test_unsupported_model_type_raises() -> None:
    payload = json.dumps(
        {
            "experiments": [
                {
                    "dataset_id": "ds-real",
                    "model_type": "transformer",  # not mlp / linear_baseline
                    "hyperparameters": {},
                    "preprocessing": {"normalize": False},
                    "random_seed": s,
                }
                for s in (42, 43, 44)
            ]
            + [
                {
                    "dataset_id": "ds-real",
                    "model_type": "linear_baseline",
                    "hyperparameters": {},
                    "preprocessing": {"normalize": False},
                    "random_seed": s,
                }
                for s in (42, 43, 44)
            ],
            "explanation": "compare a transformer to a linear baseline",
        }
    )
    agent = ExperimentPlannerAgent(llm_client=_StubLLM(payload))

    with pytest.raises(PlanValidationError):
        agent.plan_experiments("Does model complexity help?", _profile())


# ---------------------------------------------------------------------------
# 9-10. Shape guards
# ---------------------------------------------------------------------------

def test_empty_research_question_raises() -> None:
    agent = ExperimentPlannerAgent(llm_client=_StubLLM(_plan_json()))
    with pytest.raises(PlanningError):
        agent.plan_experiments("   ", _profile())


def test_no_experiments_in_response_raises() -> None:
    agent = ExperimentPlannerAgent(
        llm_client=_StubLLM(json.dumps({"experiments": [], "explanation": "x"}))
    )
    with pytest.raises(PlanValidationError):
        agent.plan_experiments("Does dropout help?", _profile())


# ---------------------------------------------------------------------------
# 11. Prompt content
# ---------------------------------------------------------------------------

def test_prompt_contains_question_and_dataset_facts() -> None:
    stub = _StubLLM(_plan_json())
    agent = ExperimentPlannerAgent(llm_client=stub)

    agent.plan_experiments("Does normalization help stability?", _profile("ds-real"))

    user_msg = stub.calls[0][-1]["content"]
    assert "Does normalization help stability?" in user_msg
    assert "ds-real" in user_msg
    assert "classification" in user_msg
    assert "churned" in user_msg
