"""
tests/unit/test_recommender.py
================================
Unit tests for backend/agents/recommender.py.

The LLM is always a stub (design.md "Mocked LLM Testing"). The recommender
must only *interpret* the statistics it is handed - these tests also check
it never recomputes them.

Test cases (task 4.10)
----------------------
1.  test_conclude_action_returns_empty_recommendation_list
2.  test_run_more_experiments_parses_recommended_configs
3.  test_recommended_config_dataset_id_is_forced
4.  test_invalid_action_raises
5.  test_run_more_experiments_with_no_configs_raises
6.  test_missing_explanation_raises
7.  test_malformed_json_raises_after_exhausting_retries (+ semantic-retry, seed dedup)
8.  test_no_experiments_raises
9.  test_conclude_ignores_any_recommended_experiments
10. test_precomputed_statistics_are_passed_into_the_prompt_verbatim
11. test_statistics_accepted_as_a_plain_sequence
"""

from __future__ import annotations

import json
from typing import List

import pytest

from backend.agents.recommender import RecommendationError, RecommenderAgent
from backend.models.anomaly import AnomalyReport
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.recommendation import Recommendation
from backend.models.statistics import StatisticalComparison


class _StubLLM:
    model = "stub"

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses) or ["{}"]
        self.calls: List[list] = []

    def chat_json(self, messages, *, schema=None, options=None) -> str:
        self.calls.append(messages)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        pass


def _exp(status: str = "success", dataset_id: str = "ds-1", dropout: float = 0.0, seed: int = 0) -> ExperimentResult:
    return ExperimentResult(
        session_id="s1",
        config=ExperimentConfiguration(
            dataset_id=dataset_id,
            model_type="mlp",
            hyperparameters={"dropout": dropout, "learning_rate": 0.001, "batch_size": 32},
            random_seed=seed,
        ),
        task_type="classification" if status != "failed" else None,
        metrics=(
            {
                "train_loss": 0.1,
                "val_loss": 0.12,
                "accuracy": 0.90,
                "n_classes": 2,
                "n_val_samples": 100,
                "training_time_seconds": 1.0,
            }
            if status != "failed"
            else None
        ),
        status=status,  # type: ignore[arg-type]
        error="boom" if status == "failed" else None,
    )


def _comparison() -> StatisticalComparison:
    return StatisticalComparison(
        condition_a_name="dropout_0.0",
        condition_b_name="dropout_0.2",
        metric="accuracy",
        t_statistic=2.51,
        p_value=0.031,
        effect_size=0.72,
        confidence_interval=(0.011, 0.049),
        sample_sizes=(5, 5),
        warning=None,
    )


def _conclude_json() -> str:
    return json.dumps(
        {
            "action": "conclude",
            "recommended_experiments": [],
            "explanation": "Dropout 0.2 beats 0.0 with p=0.031 across 5 seeds each.",
            "evidence_summary": "10 successful experiments; significant, medium-large effect.",
        }
    )


def _more_json(dataset_id: str = "ds-1") -> str:
    return json.dumps(
        {
            "action": "run_more_experiments",
            "recommended_experiments": [
                {
                    "dataset_id": dataset_id,
                    "model_type": "mlp",
                    "hyperparameters": {"dropout": 0.1, "learning_rate": 0.001, "batch_size": 32},
                    "preprocessing": {"normalize": False},
                    "random_seed": 7,
                }
            ],
            "explanation": "Explore dropout=0.1 to locate the optimum.",
            "evidence_summary": "Trend suggests the 0.1-0.2 range; variance still high.",
        }
    )


_EXPERIMENTS = [_exp(dropout=d, seed=s) for d in (0.0, 0.2) for s in range(5)]


# ---------------------------------------------------------------------------
# 1-3. Happy paths
# ---------------------------------------------------------------------------

def test_conclude_action_returns_empty_recommendation_list() -> None:
    agent = RecommenderAgent(llm_client=_StubLLM(_conclude_json()))

    rec = agent.recommend_next("Does dropout help?", _EXPERIMENTS, {"a_vs_b": _comparison()}, [])

    assert isinstance(rec, Recommendation)
    assert rec.action == "conclude"
    assert rec.recommended_experiments == []
    assert "p=0.031" in rec.explanation


def test_run_more_experiments_parses_recommended_configs() -> None:
    agent = RecommenderAgent(llm_client=_StubLLM(_more_json()))

    rec = agent.recommend_next("Does dropout help?", _EXPERIMENTS, {"a_vs_b": _comparison()}, [])

    assert rec.action == "run_more_experiments"
    # The LLM proposed one config (dropout=0.1, seed=7); seed repair tops the
    # single proposed condition up to 3 distinct seeds.
    assert {c.hyperparameters["dropout"] for c in rec.recommended_experiments} == {0.1}
    assert len({c.random_seed for c in rec.recommended_experiments}) >= 3
    assert 7 in {c.random_seed for c in rec.recommended_experiments}


def test_recommended_config_dataset_id_is_forced() -> None:
    agent = RecommenderAgent(llm_client=_StubLLM(_more_json(dataset_id="HALLUCINATED")))

    rec = agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])

    assert rec.recommended_experiments[0].dataset_id == "ds-1"


# ---------------------------------------------------------------------------
# 4-8. Error paths
# ---------------------------------------------------------------------------

def test_invalid_action_raises() -> None:
    bad = json.dumps({"action": "do_something_else", "explanation": "x", "evidence_summary": "y"})
    agent = RecommenderAgent(llm_client=_StubLLM(bad))

    with pytest.raises(RecommendationError):
        agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])


def test_run_more_experiments_with_no_configs_raises() -> None:
    bad = json.dumps(
        {
            "action": "run_more_experiments",
            "recommended_experiments": [],
            "explanation": "x",
            "evidence_summary": "y",
        }
    )
    agent = RecommenderAgent(llm_client=_StubLLM(bad))

    with pytest.raises(RecommendationError):
        agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])


def test_missing_explanation_raises() -> None:
    bad = json.dumps({"action": "conclude", "recommended_experiments": [], "evidence_summary": "y"})
    agent = RecommenderAgent(llm_client=_StubLLM(bad))

    with pytest.raises(RecommendationError):
        agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])


def test_malformed_json_raises_after_exhausting_retries() -> None:
    stub = _StubLLM("no json", "still no json", "nope")
    agent = RecommenderAgent(llm_client=stub)

    with pytest.raises(RecommendationError):
        agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])
    assert len(stub.calls) == 3  # initial + 2 retries (bounded)


def test_semantic_validation_error_triggers_a_retry_that_can_succeed() -> None:
    # First reply is valid JSON but `action` is null - must be retried, not
    # hard-failed. Second reply is good.
    bad = json.dumps({"action": None, "explanation": "x", "evidence_summary": "y"})
    stub = _StubLLM(bad, _conclude_json())
    agent = RecommenderAgent(llm_client=stub)

    rec = agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])

    assert rec.action == "conclude"
    assert len(stub.calls) == 2
    # the correction message names the specific problem
    assert "action" in stub.calls[1][-1]["content"]


def test_duplicate_recommended_seeds_are_deduped_then_repaired() -> None:
    payload = json.dumps(
        {
            "action": "run_more_experiments",
            "recommended_experiments": [
                {
                    "dataset_id": "ds-1",
                    "model_type": "mlp",
                    "hyperparameters": {"dropout": 0.1, "learning_rate": 0.001, "batch_size": 32},
                    "preprocessing": {"normalize": False},
                    "random_seed": 42,
                }
            ]
            * 3,  # same config three times
            "explanation": "explore dropout=0.1",
            "evidence_summary": "trend suggests 0.1",
        }
    )
    agent = RecommenderAgent(llm_client=_StubLLM(payload))

    rec = agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])

    seeds = [c.random_seed for c in rec.recommended_experiments]
    assert len(seeds) == len(set(seeds)) == 3  # deduped to 1, topped up to 3


def test_no_experiments_raises() -> None:
    agent = RecommenderAgent(llm_client=_StubLLM(_conclude_json()))

    with pytest.raises(RecommendationError):
        agent.recommend_next("Does dropout help?", [], {}, [])


# ---------------------------------------------------------------------------
# 9-11. Behaviour details
# ---------------------------------------------------------------------------

def test_conclude_ignores_any_recommended_experiments() -> None:
    payload = json.dumps(
        {
            "action": "conclude",
            "recommended_experiments": [
                {
                    "dataset_id": "ds-1",
                    "model_type": "mlp",
                    "hyperparameters": {"dropout": 0.1},
                    "preprocessing": {"normalize": False},
                    "random_seed": 9,
                }
            ],
            "explanation": "Done.",
            "evidence_summary": "Enough evidence.",
        }
    )
    agent = RecommenderAgent(llm_client=_StubLLM(payload))

    rec = agent.recommend_next("Does dropout help?", _EXPERIMENTS, {}, [])
    assert rec.recommended_experiments == []


def test_precomputed_statistics_are_passed_into_the_prompt_verbatim() -> None:
    stub = _StubLLM(_conclude_json())
    agent = RecommenderAgent(llm_client=stub)

    anomaly = AnomalyReport(
        experiment_id="exp-abc",
        rule="outlier_detection",
        explanation="val_loss 3.7 sigma from mean",
        severity="warning",
    )
    agent.recommend_next("Does dropout help?", _EXPERIMENTS, {"a_vs_b": _comparison()}, [anomaly])

    prompt = stub.calls[0][-1]["content"]
    # The recommender is handed the scipy numbers - it must not recompute them.
    assert "0.031" in prompt          # p_value passed through
    assert "p_value" in prompt
    assert "outlier_detection" in prompt  # anomaly passed through


def test_statistics_accepted_as_a_plain_sequence() -> None:
    agent = RecommenderAgent(llm_client=_StubLLM(_conclude_json()))

    rec = agent.recommend_next("Does dropout help?", _EXPERIMENTS, [_comparison()], [])
    assert rec.action == "conclude"
