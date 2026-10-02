"""
backend/tools/experiment_runner.py
=====================================
``run_experiment``: train one configuration and return an ``ExperimentResult``
that is never an exception.

This is also the whole of result validation. A run is ``failed`` when it
crashed or produced a non-finite number (NaN/Inf loss or scores); every other
run is ``ok`` and counts as evidence - including runs whose numbers look
unusual, because dropping those would bias the statistics.
"""

from __future__ import annotations

import logging
import math

from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.trainers import TRAINERS

logger = logging.getLogger(__name__)


def run_experiment(
    config: ExperimentConfiguration,
    profile: DatasetProfile,
    session_id: str,
    round: int,
) -> ExperimentResult:
    def failed(error: str) -> ExperimentResult:
        logger.warning("experiment failed (%s seed=%d): %s", config.model_type, config.random_seed, error)
        return ExperimentResult(
            session_id=session_id, round=round, config=config, status="failed", error=error
        )

    try:
        out = TRAINERS[config.model_type](config, profile)
    except Exception as exc:  # noqa: BLE001 - any training failure becomes a failed run
        return failed(f"{type(exc).__name__}: {exc}")

    numbers = [*out.metrics.values(), *out.val_scores, *out.test_scores]
    if not all(math.isfinite(v) for v in numbers):
        return failed("training produced a non-finite value (NaN/Inf)")

    return ExperimentResult(
        session_id=session_id,
        round=round,
        config=config,
        status="ok",
        metrics=out.metrics,
        val_scores=out.val_scores,
        test_scores=out.test_scores,
    )
