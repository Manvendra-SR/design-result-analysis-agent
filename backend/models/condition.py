"""
backend/models/condition.py
=============================
The canonical definition of an experimental **condition**, and the vocabulary
that surrounds it.

Why this module exists
----------------------
"Condition" was previously defined twice, differently:

- ``state_machine/nodes.py`` grouped by ``(model_type, normalize, hyperparameters)``
- ``tools/anomaly_detector.py`` grouped by ``(dataset_id, model_type, hyperparameters)``

so the statistical analyzer and the anomaly detector could disagree about which
experiments belonged together. Both now import from here.

The vocabulary this project uses
--------------------------------
========================  ====================================================
Investigation             One ``sessions`` row: one research question, its own
                          cycles / experiments / anomalies. Never shares
                          evidence with another investigation.
Adaptive cycle            One pass of plan/execute → validate → analyze →
                          recommend. Recorded on ``experiments.cycle``.
Experiment                One ``experiments`` row = **one complete training
                          run** of a single configuration.
Configuration             ``ExperimentConfiguration``: dataset + model_type +
                          hyperparameters + preprocessing + random_seed.
Condition                 A configuration **with the random seed removed** -
                          the thing a statistical comparison compares.
Replicate                 One experiment within a condition, distinguished
                          only by its ``random_seed``.
Training epoch            One pass over the training set *inside* a single
                          experiment (``hyperparameters["epochs"]``). It is a
                          property of one experiment, never a count of them.
========================  ====================================================

So ``6 experiments`` with ``epochs=20`` means six independent training runs of
twenty epochs each - not six epochs, and not one run of 120.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from backend.models.experiment import ExperimentConfiguration

#: (dataset_id, model_type, normalize, sorted hyperparameters)
ConditionKey = Tuple[str, str, bool, Tuple[Tuple[str, float], ...]]


def condition_key(config: ExperimentConfiguration) -> ConditionKey:
    """The identity of a condition: everything about a configuration *except*
    its ``random_seed``.

    ``dataset_id`` is included so results from two different datasets are never
    pooled (their loss scales are unrelated); ``preprocessing.normalize`` is
    included because it materially changes the model being fitted.
    """
    return (
        config.dataset_id,
        config.model_type,
        config.preprocessing.normalize,
        tuple(sorted(config.hyperparameters.items())),
    )


def condition_label(config: ExperimentConfiguration) -> str:
    """A short human-readable name for a condition (no seed - seeds are
    replicates *within* a condition)."""
    parts = [config.model_type]
    if config.preprocessing.normalize:
        parts.append("normalized")
    hp = ", ".join(f"{k}={_trim(v)}" for k, v in sorted(config.hyperparameters.items()))
    if hp:
        parts.append(hp)
    return " | ".join(parts)


def _trim(value: float) -> str:
    """Render 20.0 as '20' so labels read like the configuration the user typed."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Matched seeds across conditions
# ---------------------------------------------------------------------------
#: Replicates are numbered from 1, and EVERY condition uses the same numbers.
#: The k-th replicate of any condition carries seed k, so "dropout=0.0 vs
#: dropout=0.2" is compared on seeds (1, 2, 3) against seeds (1, 2, 3) rather
#: than (1, 2, 3) against (4, 5, 6).
_FIRST_SEED = 1


def assign_matched_seeds(
    configs: List[ExperimentConfiguration],
    min_replicates: int = 3,
    existing_seeds: Optional[Dict[ConditionKey, Set[int]]] = None,
) -> List[ExperimentConfiguration]:
    """Renumber a batch of configurations so compared conditions share seeds.

    Why the agents' own seed choices are discarded
    ----------------------------------------------
    ``random_seed`` controls model initialisation and batch shuffling only -
    the train/val/test split is fixed per dataset (``DatasetProfile.split_seed``,
    see ``tools/dataset/splitting.py``). It is therefore a *nuisance* factor,
    and the cleanest design reuses the same draws in every condition (the
    "common random numbers" technique): whatever difference remains between two
    conditions is then attributable to the factor that was varied, not to the
    conditions having been initialised differently. Allocating a fresh seed
    range per condition - which both agents used to do - throws that away for
    nothing, since no seed *value* carries meaning of its own.

    The comparison stays an independent two-sample t-test
    (``tools/statistical_analyzer.py``); matching seeds only removes an
    avoidable source of between-condition variation, it does not change the test.

    Behaviour
    ---------
    Each distinct condition in ``configs`` is given
    ``max(min_replicates, distinct seeds it was proposed with)`` replicates,
    numbered 1, 2, 3, ... . ``existing_seeds`` maps a condition to the seeds
    it has ALREADY been run with in this session, so topping an existing
    condition up continues the sequence (4, 5, 6) instead of re-running seeds
    whose results are already on record. Deterministic: the same input always
    produces the same output.
    """
    prior = existing_seeds or {}
    groups: Dict[ConditionKey, List[ExperimentConfiguration]] = {}
    for config in configs:
        groups.setdefault(condition_key(config), []).append(config)

    out: List[ExperimentConfiguration] = []
    for key, group in groups.items():
        n_replicates = max(min_replicates, len({c.random_seed for c in group}))
        already = prior.get(key, set())
        seed = _FIRST_SEED
        for i in range(n_replicates):
            while seed in already:
                seed += 1
            template = group[i] if i < len(group) else group[0]
            out.append(template.model_copy(update={"random_seed": seed}))
            seed += 1
    return out
