"""
backend/tools/anomaly_detector.py
====================================
AnomalyDetector: rule-based anomaly detection with template explanations.

Three detection rules (unchanged in count/spirit from the original design;
see DESIGN_REVIEW_CHANGES.md for why a 4th "configuration integrity" rule
was rejected, and its "Architecture Revision" entry for why validation
collapse changed from a flat MNIST threshold to a class-count-aware test):

- outlier_detection    : val_loss > 3 leave-one-out std devs from the mean
                         of the *rest* of its (dataset, model_type,
                         hyperparameters) group.
- loss_divergence      : training loss increased over the final 20% of
                         epochs (skipped when ``initial_train_loss`` is
                         absent — i.e. ``linear_baseline``, which has no
                         epoch loop).
- validation_collapse  : for classification experiments, accuracy is not
                         statistically distinguishable from random-chance
                         performance for the dataset's n_classes.

Every experiment that produced metrics is examined, including ones currently
marked "anomalous" — the detector is stateless and reports what holds *now*,
which is what lets the validation node withdraw a flag that no longer applies.
Only "failed" experiments (no ``metrics`` at all) are skipped.

Leave-one-out mean/std for outlier detection, and why it needs a floor
----------------------------------------------------------------------
The naive "include all points" z-score is mathematically bounded below
3-sigma for small group sizes (max achievable is ``sqrt(n-1)``, below 3 for
any ``n <= 9``). Each point's mean/std is instead computed from the *rest* of
its group (leave-one-out / Grubbs'-test style), so an outlier can never
inflate its own baseline.

Leave-one-out alone, however, over-fires at the *bottom* end. With a group of
3, "the rest" is 2 points, whose std is just half their gap — so 3 std devs is
only 1.5x that gap and almost any third value clears it. A real run produced
two "4.2 sigma" and "4.9 sigma" flags for val_loss values of 0.505 against a
mean of 0.498: a 1.5% deviation, comfortably inside the spread the same
condition showed once it had 16 replicates. Those false flags then excluded
their experiments from the analysis, shrinking the baseline further, and gave
the Recommender an anomaly it kept trying to "resolve" until the cycle cap.

Two guards fix this:

- ``_MIN_GROUP_FOR_OUTLIER`` (5): below this the leave-one-out std is not a
  usable dispersion estimate at all, so the rule abstains rather than guesses.
- ``_MIN_RELATIVE_DEVIATION`` (10%): the value must also differ from the group
  mean by a materially large fraction of it. A z-score says "unusual for this
  group"; this says "unusual by an amount worth a human's attention", which is
  what an anomaly report claiming "numerical instability" should mean.

Both are deliberately conservative: a missed marginal outlier costs far less
than a false one, which actively corrupts the statistics and the loop.

Re-evaluation (anomaly lifecycle)
----------------------------------
The detector is pure - it reports what currently holds for the experiments it
is given, with no memory. The validation node calls it with *every* completed
experiment in the session each cycle and reconciles the result against the
stored flags (see ``state_machine/nodes.py``), so a flag raised early against a
thin group is withdrawn once the group grows. That is what makes "no unresolved
anomalies" a condition the Recommender can actually reach.

Dataset-aware grouping (architecture revision)
--------------------------------------------------
Grouping uses the shared ``models/condition.condition_key`` - the same
definition of "condition" the statistical analyzer uses, so the two tools can
never disagree about which experiments are replicates of one another. It
includes ``dataset_id`` (val_loss scale is dataset-specific) and
``preprocessing.normalize`` (it changes the model being fitted).

Class-count-aware validation collapse — one-sided z-test (architecture revision)
------------------------------------------------------------------------------------
The original rule ("MNIST accuracy < 20%") was implicitly tied to a
10-class dataset (20% = 2x the 10% random-chance baseline). Generalizing
that "2x baseline" multiplier naively to ``2 / n_classes`` breaks for
binary classification: baseline=50%, so the threshold becomes 100% —
flagging every non-perfect model. Instead, this uses a one-sided
two-proportion z-test against the chance baseline:

    z = (accuracy - 1/n_classes) / sqrt((1/n_classes)(1 - 1/n_classes) / n_val_samples)

flagged when ``z < 1.645`` (95% one-sided confidence the model is not doing
better than random guessing). This correctly reproduces the original
MNIST-shaped case (10 classes, large n_val, accuracy=0.11 -> not
significantly above the 0.10 baseline -> still flagged) while not
misfiring on binary classification (baseline=0.5: an accuracy near 0.5 with
a large enough validation set legitimately fails to clear z=1.645, while an
accuracy of, say, 0.75 comfortably clears it).

``n_classes`` and ``n_val_samples`` are recorded in ``metrics`` by both
trainers (see ``backend/tools/trainers.py``) so this rule stays computable
without the detector touching the dataset profile or the database.

Requirements
------------
5.1  Identify outliers within each (dataset, configuration) group
5.2  Check for training loss divergence
5.3  Class-count-aware check for validation accuracy near chance level
5.6  Provide a natural language (template) explanation
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import numpy as np

from backend.models.anomaly import AnomalyReport
from backend.models.condition import ConditionKey, condition_key
from backend.models.experiment import ExperimentResult
from backend.tools.anomaly_templates import ANOMALY_TEMPLATES

logger = logging.getLogger(__name__)

# abs(value - mean) > _OUTLIER_STD_THRESHOLD * std  =>  candidate outlier
_OUTLIER_STD_THRESHOLD = 3.0
# Minimum replicates in a condition before the outlier rule will fire at all.
# With fewer, the leave-one-out std is not a usable dispersion estimate (see
# the module docstring) and the rule produces false positives.
_MIN_GROUP_FOR_OUTLIER = 5
# The deviation must also be this fraction of the group mean. Guards against
# "statistically unusual but practically identical" flags on tight groups.
_MIN_RELATIVE_DEVIATION = 0.10
# Below this, treat std as zero (avoids flagging floating-point noise as an
# outlier when a group's values are all effectively identical).
_MIN_STD_FOR_OUTLIER_CHECK = 1e-9
# One-sided z critical value at 95% confidence (see module docstring).
_VALIDATION_COLLAPSE_Z_THRESHOLD = 1.645

_ConfigKey = ConditionKey


def _config_group_key(experiment: ExperimentResult) -> _ConfigKey:
    """Group replicates of one condition together.

    Delegates to the shared ``models/condition.condition_key`` so the detector
    and the statistical analyzer group experiments identically. ``random_seed``
    is a top-level field on ``ExperimentConfiguration`` rather than a
    hyperparameter, so replicates of a condition land in the same group.
    """
    return condition_key(experiment.config)


class AnomalyDetector:
    """Rule-based anomaly detection with template-generated explanations."""

    def detect_anomalies(
        self, experiments: List[ExperimentResult]
    ) -> List[AnomalyReport]:
        """Run all 3 detection rules over ``experiments``.

        An experiment can trigger more than one rule (e.g. an outlier that
        is also a validation collapse) — each triggered rule produces its
        own ``AnomalyReport``.

        Requirements: 5.1, 5.2, 5.3, 5.6
        """
        anomalies: List[AnomalyReport] = []
        # Every experiment that produced metrics, INCLUDING ones currently
        # flagged "anomalous". This is what makes re-evaluation possible: a run
        # excluded from its own group's baseline could never be un-flagged, and
        # excluding it also biased the baseline for everything else. Only
        # "failed" runs (no metrics at all) are skipped.
        completed = [e for e in experiments if e.status != "failed" and e.metrics]

        anomalies.extend(self._detect_outliers(completed))
        anomalies.extend(self._detect_loss_divergence(completed))
        anomalies.extend(self._detect_validation_collapse(completed))

        return anomalies

    # ------------------------------------------------------------------
    # Rule 1: outlier detection
    # ------------------------------------------------------------------

    def _detect_outliers(
        self, experiments: List[ExperimentResult]
    ) -> List[AnomalyReport]:
        groups: Dict[_ConfigKey, List[ExperimentResult]] = {}
        for exp in experiments:
            groups.setdefault(_config_group_key(exp), []).append(exp)

        anomalies: List[AnomalyReport] = []
        for group in groups.values():
            values = [
                exp.metrics["val_loss"] for exp in group if exp.metrics  # type: ignore[index]
            ]
            if len(values) < _MIN_GROUP_FOR_OUTLIER:
                # Too few replicates for the leave-one-out std to mean anything
                # (see module docstring) - abstain rather than guess.
                continue

            for i, exp in enumerate(group):
                rest = np.array(values[:i] + values[i + 1:], dtype=float)
                mean = float(np.mean(rest))
                std = float(np.std(rest))
                if std < _MIN_STD_FOR_OUTLIER_CHECK:
                    continue  # the rest of the group is effectively uniform

                value = values[i]
                deviation = abs(value - mean)
                z_score = (value - mean) / std
                # Both guards must hold: statistically extreme for this group,
                # AND materially different in absolute terms.
                if (
                    deviation > _OUTLIER_STD_THRESHOLD * std
                    and deviation > _MIN_RELATIVE_DEVIATION * abs(mean)
                ):
                    anomalies.append(
                        AnomalyReport(
                            experiment_id=exp.experiment_id,
                            rule="outlier_detection",
                            explanation=ANOMALY_TEMPLATES["outlier_detection"].format(
                                metric="val_loss",
                                value=value,
                                z_score=z_score,
                                mean=mean,
                            ),
                            severity="warning",
                        )
                    )
        return anomalies

    # ------------------------------------------------------------------
    # Rule 2: loss divergence
    # ------------------------------------------------------------------

    def _detect_loss_divergence(
        self, experiments: List[ExperimentResult]
    ) -> List[AnomalyReport]:
        anomalies: List[AnomalyReport] = []
        for exp in experiments:
            metrics = exp.metrics or {}
            if "initial_train_loss" not in metrics:
                # No epoch history to compare (e.g. linear_baseline, which
                # fits via a single closed-form/solver call, not epochs).
                continue

            initial_loss = metrics["initial_train_loss"]
            final_loss = metrics["train_loss"]
            if final_loss > initial_loss:
                anomalies.append(
                    AnomalyReport(
                        experiment_id=exp.experiment_id,
                        rule="loss_divergence",
                        explanation=ANOMALY_TEMPLATES["loss_divergence"].format(
                            initial_loss=initial_loss,
                            final_loss=final_loss,
                        ),
                        severity="warning",
                    )
                )
        return anomalies

    # ------------------------------------------------------------------
    # Rule 3: validation collapse (class-count-aware one-sided z-test)
    # ------------------------------------------------------------------

    def _detect_validation_collapse(
        self, experiments: List[ExperimentResult]
    ) -> List[AnomalyReport]:
        anomalies: List[AnomalyReport] = []
        for exp in experiments:
            if exp.task_type != "classification":
                # Regression experiments have no "accuracy" key at all (see
                # ExperimentResult docstring) — this rule doesn't apply.
                continue
            metrics = exp.metrics or {}
            accuracy = metrics.get("accuracy")
            n_classes = metrics.get("n_classes")
            n_val_samples = metrics.get("n_val_samples")
            if accuracy is None or not n_classes or not n_val_samples:
                continue

            baseline = 1.0 / n_classes
            se = math.sqrt(baseline * (1.0 - baseline) / n_val_samples)
            z_score = (accuracy - baseline) / se if se > 0 else float("inf")

            if z_score < _VALIDATION_COLLAPSE_Z_THRESHOLD:
                anomalies.append(
                    AnomalyReport(
                        experiment_id=exp.experiment_id,
                        rule="validation_collapse",
                        explanation=ANOMALY_TEMPLATES["validation_collapse"].format(
                            accuracy=accuracy,
                            baseline=baseline,
                            n_classes=int(n_classes),
                            z_score=z_score,
                        ),
                        severity="critical",
                    )
                )
        return anomalies
