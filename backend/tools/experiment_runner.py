"""
backend/tools/experiment_runner.py
=====================================
ExperimentRunner: dispatches ExperimentConfiguration (+ a resolved
DatasetProfile) to the right trainer, turns exceptions into failed
ExperimentResults, and retries transient failures.

Retry policy
------------
Mirrors the shape of ``backend/tools/state_manager.py``'s ``_retry_db``:
tenacity, max 3 attempts, exponential backoff (1s, 2s, 4s...). The
difference is *what* gets retried — here it is gated by ``_is_transient``,
since permanent errors (invalid config, shape mismatches) must fail
immediately, not burn 3 attempts before reporting failure.

Requirements
------------
3.1  Executes configurations against a resolved DatasetProfile
3.2  Detailed error message on invalid configuration
3.9  run_batch executes sequentially, continuing past failures
10.1 Retries transient errors up to 3 times with exponential backoff
10.2 Does NOT retry permanent errors
10.3 Marks failed and logs error details when retries are exhausted
"""

from __future__ import annotations

import logging
from typing import List

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.tools.trainers import train_linear_baseline, train_mlp

logger = logging.getLogger(__name__)

_TRANSIENT_SUBSTRINGS = (
    "out of memory",
    "cuda error",
    "temporarily unavailable",
    "connection reset",
)


def _is_transient(exc: BaseException) -> bool:
    """Best-effort classification of an exception as transient vs. permanent.

    Transient (retryable): CUDA out-of-memory, generic OS-level I/O hiccups
    (``OSError``, ``TimeoutError``).

    Permanent (not retried): everything else, notably ``ValueError`` /
    ``TypeError`` (invalid configuration) and PyTorch shape-mismatch
    ``RuntimeError``s, which are bugs in the configuration, not flukes —
    retrying them would just fail the same way 3 times.
    """
    if isinstance(exc, (OSError, TimeoutError)):
        return True
    message = str(exc).lower()
    return any(substring in message for substring in _TRANSIENT_SUBSTRINGS)


_retry_transient = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


class ExperimentRunner:
    """Deterministic experiment execution for ``mlp`` and ``linear_baseline``,
    against any profiled dataset.

    Does not touch the database — callers (tests today, a future LangGraph
    execution node) are responsible for resolving ``config.dataset_id``
    into a ``DatasetProfile`` (via ``StateManager.get_dataset``) and for
    persisting the returned ``ExperimentResult``.
    """

    _TRAINERS = {
        "mlp": train_mlp,
        "linear_baseline": train_linear_baseline,
    }

    def run_experiment(
        self,
        config: ExperimentConfiguration,
        dataset_profile: DatasetProfile,
        session_id: str = "",
    ) -> ExperimentResult:
        """Execute one experiment, never raising.

        Dispatches to the trainer matching ``config.model_type``. Transient
        failures (see ``_is_transient``) are retried up to 3 times with
        exponential backoff; any other exception — or a transient one that
        exhausts its retries — is caught and returned as
        ``ExperimentResult(status="failed", error=...)``.

        Parameters
        ----------
        config:
            Complete experiment specification.
        dataset_profile:
            The dataset referenced by ``config.dataset_id``, already
            resolved by the caller. This class never looks it up itself.
        session_id:
            Passed through into the returned ``ExperimentResult``.

        Requirements: 3.1, 3.2, 3.9, 10.1, 10.2, 10.3
        """
        trainer = self._TRAINERS.get(config.model_type)
        if trainer is None:
            # Not reachable via a validated ExperimentConfiguration (model_type
            # is a Literal), but guarded defensively for direct/malformed calls.
            return ExperimentResult(
                session_id=session_id,
                config=config,
                task_type=dataset_profile.task_type,
                metrics=None,
                status="failed",
                error=f"Unsupported model_type: {config.model_type!r}",
            )

        try:
            run_with_retry = _retry_transient(trainer)
            return run_with_retry(config, dataset_profile, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 - intentionally broad: any
            # training failure must become a failed result, not crash the loop.
            logger.error(
                "Experiment failed: model_type=%s dataset_id=%s seed=%d error=%s",
                config.model_type, config.dataset_id, config.random_seed, exc,
            )
            return ExperimentResult(
                session_id=session_id,
                config=config,
                task_type=dataset_profile.task_type,
                metrics=None,
                status="failed",
                error=str(exc),
            )

    def run_batch(
        self,
        configs: List[ExperimentConfiguration],
        dataset_profile: DatasetProfile,
        session_id: str = "",
    ) -> List[ExperimentResult]:
        """Execute multiple experiments sequentially, in order.

        A failure in one experiment does not stop the batch — each config
        gets its own ``run_experiment`` call, and results are returned in
        the same order as ``configs`` (mixing "success" and "failed" as
        appropriate).

        Requirements: 3.9
        """
        return [
            self.run_experiment(cfg, dataset_profile, session_id=session_id)
            for cfg in configs
        ]
