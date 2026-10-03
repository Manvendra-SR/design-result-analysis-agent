"""Tests for the family registry and Plan (backend/models/*): one change at a time, enforced by construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.models.experiment import FAMILIES, ExperimentConfiguration
from backend.models.investigation import BaseSetup, Plan
from tests._fakes import dropout_plan, selection_plan


# --------------------------------------------------------------------- registry
@pytest.mark.parametrize(
    "model_type, hp, message",
    [
        ("mlp", {"max_depth": 3}, "unknown mlp hyperparameter"),
        ("linear_baseline", {"hidden_size": 8}, "unknown linear_baseline"),
        ("mlp", {"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
        ("mlp", {"epochs": 60}, "epochs must be in"),          # capped at 50 to bound one run's cost
        ("mlp", {"batch_size": 8}, "batch_size must be in"),   # at least 16, same reason
        ("decision_tree", {"max_depth": 2.5}, "whole number"),
        ("random_forest", {"max_features": 1.5}, "max_features"),
    ],
)
def test_configurations_outside_the_registry_are_rejected(model_type, hp, message) -> None:
    with pytest.raises(ValidationError, match=message):
        ExperimentConfiguration(dataset_id="ds", model_type=model_type, hyperparameters=hp)


def test_registry_seeds_and_preprocessing() -> None:
    assert [f for f, spec in FAMILIES.items() if spec.seeded] == ["random_forest", "mlp"]
    assert FAMILIES["mlp"].normalize and FAMILIES["linear_baseline"].normalize
    assert not FAMILIES["random_forest"].normalize  # trees are scale-invariant
    assert "batch_size" not in FAMILIES["mlp"].refinable


def test_the_configuration_key_ignores_the_seed() -> None:
    a = ExperimentConfiguration(dataset_id="ds", model_type="mlp", hyperparameters={"dropout": 0.2}, random_seed=0)
    b = a.model_copy(update={"random_seed": 2})
    assert a.key() == b.key()
    assert a.key() != a.model_copy(update={"hyperparameters": {"dropout": 0.3}}).key()


# ------------------------------------------------------------------ effect mode
def test_effect_levels_change_only_the_factor() -> None:
    plan = dropout_plan(levels=(0.0, 0.3))
    ref, other = plan.candidates
    assert (ref.id, ref.parent, ref.label) == ("c1", None, "dropout=0")
    assert (other.parent, other.knob, other.value, other.label) == ("c1", "dropout", 0.3, "dropout=0.3")
    assert ref.hyperparameters == {"hidden_size": 16, "epochs": 2, "dropout": 0.0}
    assert other.hyperparameters == {"hidden_size": 16, "epochs": 2, "dropout": 0.3}
    assert ref.normalize is True  # the mlp family default when the base leaves it unset


def test_reference_comes_first_whatever_the_order() -> None:
    plan = dropout_plan(levels=(0.5, 0.0, 0.2), reference=0.0)
    assert [c.label for c in plan.candidates] == ["dropout=0", "dropout=0.5", "dropout=0.2"]
    assert plan.reference == "c1"


def test_model_type_levels_use_each_familys_own_knobs_and_preprocessing() -> None:
    plan = Plan.effect("model_type", ["linear_baseline", "random_forest", "mlp"], "linear_baseline",
                       BaseSetup(hyperparameters={"hidden_size": 32, "max_depth": 6}), "r")
    linear, forest, mlp = plan.candidates
    assert linear.hyperparameters == {} and linear.normalize
    assert forest.hyperparameters == {"max_depth": 6} and not forest.normalize
    assert mlp.hyperparameters == {"hidden_size": 32}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(levels=[0.1]), "at least 2 levels"),
        (dict(levels=[0.1, 0.1], reference=0.1), "distinct"),
        (dict(levels=[0.1, 0.2], reference=0.3), "not one of the levels"),
        (dict(levels=[0.0, 1.5], reference=0.0), "dropout"),
        (dict(levels=[0.0, "high"], reference=0.0), "must be a number"),
        (dict(base=BaseSetup(model_type="linear_baseline")), "not a hyperparameter of linear_baseline"),
    ],
)
def test_invalid_effect_plans_are_rejected(kwargs, message) -> None:
    fields = {**dict(factor="dropout", levels=[0.0, 0.5], reference=0.0, base=BaseSetup(), rationale="r"),
              **kwargs}
    with pytest.raises(ValueError, match=message):
        Plan.effect(**fields)


def test_effect_refinement_keeps_the_factor_and_rejects_repeats() -> None:
    plan = dropout_plan()
    [new] = plan.refine(None, None, [0.2], round=2)
    assert (new.id, new.parent, new.knob, new.round) == ("c3", "c1", "dropout", 2)
    with pytest.raises(ValueError, match="varies only dropout"):
        plan.refine("c1", "hidden_size", [32], 2)
    with pytest.raises(ValueError, match="already tried"):
        plan.refine(None, "dropout", [0.5], 2)
    with pytest.raises(ValueError, match="1-3 values"):
        plan.refine(None, None, [0.1, 0.2, 0.3, 0.4], 2)


# --------------------------------------------------------------- selection mode
def test_selection_starts_every_family_at_defaults_with_the_simplest_as_reference() -> None:
    plan = selection_plan(families=("mlp", "random_forest", "linear_baseline"))
    assert [c.label for c in plan.candidates] == ["linear_baseline", "random_forest", "mlp"]
    assert plan.reference == "c1"
    assert plan.get("c3").hyperparameters == FAMILIES["mlp"].defaults
    with pytest.raises(ValueError, match="distinct families"):
        Plan.selection(["mlp", "mlp"], "r")
    with pytest.raises(ValueError, match="unknown model families"):
        Plan.selection(["mlp", "xgboost"], "r")


def test_selection_refinement_changes_one_knob_of_a_contender() -> None:
    plan = selection_plan()
    forest = plan.get("c3")
    new = plan.refine("c3", "min_samples_leaf", [5, 20], round=2, contenders=["c3", "c4"])
    assert [c.id for c in new] == ["c5", "c6"]
    assert new[0].hyperparameters == {**forest.hyperparameters, "min_samples_leaf": 5.0}
    assert new[0].label == "random_forest, min_samples_leaf=5" and new[0].parent == "c3"

    grown = plan.with_candidates(new)
    assert grown.candidate_of(new[1].config("ds", seed=2)).id == "c6"


@pytest.mark.parametrize(
    "parent, knob, values, message",
    [
        ("c2", "max_depth", [4], "not a contender"),
        ("c3", "hidden_size", [128], "can refine only"),
        ("c4", "batch_size", [64], "can refine only"),   # a speed knob, not refinable
        ("c4", "epochs", [80], "epochs must be in"),
        ("c4", "hidden_size", [64], "already tried"),     # the mlp default
        ("c9", "max_depth", [4], "unknown parent"),
    ],
)
def test_selection_refinement_rules(parent, knob, values, message) -> None:
    with pytest.raises(ValueError, match=message):
        selection_plan().refine(parent, knob, values, 2, contenders=["c3", "c4"])


def test_json_round_trip_keeps_level_types() -> None:
    norm = Plan.effect("normalize", [False, True], False, BaseSetup(), "r")
    restored = Plan.model_validate(norm.model_dump(mode="json"))
    assert [c.value for c in restored.candidates] == [False, True]
    epochs = Plan.effect("epochs", [10, 20], 10, BaseSetup(), "r")
    restored = Plan.model_validate(epochs.model_dump(mode="json"))
    assert [c.value for c in restored.candidates] == [10.0, 20.0] and restored.candidates[1].label == "epochs=20"
    sel = selection_plan()
    assert Plan.model_validate(sel.model_dump(mode="json")) == sel
