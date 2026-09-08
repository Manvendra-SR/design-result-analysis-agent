"""
backend/agents/recommender.py
===============================
Recommender_Agent: the "brain" that closes the adaptive loop. Given the
research question, every experiment run so far, the scipy-computed
statistical comparisons, and any detected anomalies, it decides whether to
run more experiments (and which) or to conclude.

Strict LLM / deterministic separation (Requirement 12.3)
-------------------------------------------------------
This agent receives statistics as pre-computed structured input and only
*interprets* them. It never calls scipy/numpy - ``backend/agents/`` imports
none of them. The p-values, effect sizes and confidence intervals in the
prompt were produced by ``StatisticalAnalyzer`` (Phase 3).

Output validation
-----------------
The same guard rails as the Planner apply to any ``recommended_experiments``:
each is rebuilt through ``ExperimentConfiguration`` (running the Phase 2
hyperparameter validator), and ``dataset_id`` is forced to the dataset the
existing experiments already use (all experiments in a session share one
dataset). ``action`` must be exactly ``"run_more_experiments"`` (which
requires a non-empty recommendation list) or ``"conclude"`` (which forces
the list empty).

Also like the Planner, the recommended batch is **deterministically
repaired** so every distinct condition it proposes carries >=
``_MIN_SEEDS_PER_CONDITION`` distinct seeds - exact ``(condition, seed)``
duplicates are dropped first, then short conditions are topped up on fresh
seeds. This stops the recommender re-proposing the same seed, and stops a
2-seed condition from reaching the statistical-analysis node.

Structured-output retry
-----------------------
The LLM call goes through ``request_structured``: a reply that parses as
JSON but is semantically wrong (``action: null``, missing
``evidence_summary``) is fed back to the model to fix, up to a bounded
number of attempts, before the agent raises ``RecommendationError``.

Requirements
------------
7.1  Retrieve all experiments in the current session (caller passes them in)
7.2  Analyse statistical results, anomalies, history to find knowledge gaps
7.3  Natural-language explanation of current evidence + reasoning
7.4  Recommend specific Experiment_Configurations
7.5  Insufficient evidence -> recommend more replicates / seeds
7.6  Anomaly detected -> recommend rerun with a different seed
7.7  Promising value -> recommend exploring nearby values
7.8  Actionable recommendation while not yet conclusive
7.9  Sufficient evidence -> recommend concluding
12.3 Receives statistics as input; does NOT compute them
15.2 Uses the LLM backend (Groq) via the LLMClient protocol
15.8 Does not proceed on malformed LLM output
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import ValidationError

from backend.agents._common import StructuredOutputError, request_structured
from backend.agents.client_factory import create_llm_client
from backend.agents.llm_client import LLMClient, LLMError
from backend.agents.output_schemas import (
    RECOMMENDATION_SCHEMA,
    strip_null_hyperparameters,
)
from backend.agents.parsing import JSONParseError
from backend.models.anomaly import AnomalyReport
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.recommendation import Recommendation
from backend.models.statistics import StatisticalComparison

logger = logging.getLogger(__name__)

_VALID_ACTIONS = ("run_more_experiments", "conclude")
_MIN_SEEDS_PER_CONDITION = 3  # matches planner._MIN_SEEDS_PER_CONDITION
# Slightly higher than the planner: the recommender writes a paragraph of
# reasoning, but structure/action still matter more than prose style.
_RECOMMENDER_OPTIONS: Dict[str, Any] = {"temperature": 0.3}

_StatsInput = Union[
    Dict[str, StatisticalComparison],
    Sequence[StatisticalComparison],
    None,
]

_SYSTEM_PROMPT = """\
You are an adaptive ML experiment recommender. You are given a research \
question, the experiments run so far, statistical comparisons that have \
ALREADY been computed with scipy, and any anomalies that were detected. \
Decide what to do next.

You do NOT compute statistics. Cite the p-values, effect sizes and \
confidence intervals you are given - do not recalculate or second-guess \
them.

Reasoning patterns:
- Anomalies detected -> recommend re-running those configurations with new \
random seeds.
- High within-condition variance, or fewer than 5 successful replicates per \
condition -> recommend more seeds for the existing conditions.
- A promising trend in one hyperparameter -> recommend nearby values to \
locate the optimum.
- A high proportion of failed experiments -> recommend adjusting the \
configuration to reduce the failure rate.
- Sufficient evidence (a significant result at p < 0.05 with at least 5 \
replicates per condition and no unresolved anomalies) -> conclude.

Every recommended configuration must set dataset_id to the id given below, \
use model_type "mlp" or "linear_baseline", and keep mlp hyperparameters in \
range (dropout 0-1, learning_rate > 0, batch_size > 0, hidden_size > 0, \
epochs > 0). linear_baseline takes "hyperparameters": {}.

When action is "run_more_experiments", recommend AT MOST 6 configurations \
(one or two conditions is normal); the system adds seed replicates itself, \
so you need only ONE configuration per condition. Keep the explanation and \
evidence_summary to a few sentences each.

Respond with ONLY a JSON object of this shape:
{
  "action": "run_more_experiments" | "conclude",
  "recommended_experiments": [ { "dataset_id": "...", "model_type": "mlp", "hyperparameters": {...}, "preprocessing": {"normalize": false}, "random_seed": 7 } ],
  "explanation": "<plain-language reasoning that cites specific experiments and statistics>",
  "evidence_summary": "<one short paragraph summarising the accumulated evidence>"
}
Use "recommended_experiments": [] when action is "conclude"."""


class RecommendationError(ValueError):
    """The Recommender_Agent could not produce a valid recommendation.

    Raised on unusable LLM output, an invalid ``action``, a
    ``run_more_experiments`` recommendation with no configurations, or
    recommended configurations that fail validation.
    """


class RecommenderAgent:
    """LLM-based agent that drives the adaptive loop from accumulated evidence.

    Parameters
    ----------
    llm_client:
        An ``LLMClient``. Defaults to the provider from ``LLM_PROVIDER``
        (``create_llm_client``). Injected in tests.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client or create_llm_client()

    def recommend_next(
        self,
        research_question: str,
        experiments: List[ExperimentResult],
        statistical_results: _StatsInput = None,
        anomalies: Optional[List[AnomalyReport]] = None,
    ) -> Recommendation:
        """Analyse the evidence and recommend the next action.

        Parameters
        ----------
        research_question:
            The original question driving the session.
        experiments:
            Every experiment run in the session so far (success, failed,
            anomalous). Must be non-empty.
        statistical_results:
            Pre-computed comparisons from ``StatisticalAnalyzer`` - either a
            ``{name: StatisticalComparison}`` mapping or a plain sequence.
        anomalies:
            Anomaly reports from ``AnomalyDetector`` for this session.

        Returns
        -------
        Recommendation

        Raises
        ------
        RecommendationError
            If there are no experiments to reason about, or the LLM output
            is unusable / invalid.
        LLMError
            If the LLM backend cannot be reached.
        """
        if not experiments:
            raise RecommendationError(
                "Cannot recommend next experiments: no experiments have been run yet."
            )

        dataset_id = experiments[0].config.dataset_id
        evidence = self._build_evidence(experiments, statistical_results, anomalies or [])
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f'Research question: "{research_question.strip()}"\n\n'
                    f"dataset_id for any recommended experiments: {dataset_id}\n\n"
                    f"Evidence (JSON):\n"
                    f"{json.dumps(evidence, separators=(',', ':'), default=str)}\n\n"
                    "Recommend the next action now."
                ),
            },
        ]

        try:
            return request_structured(
                self._llm,
                messages,
                build=lambda parsed: self._build_recommendation(parsed, dataset_id),
                schema=RECOMMENDATION_SCHEMA,
                options=_RECOMMENDER_OPTIONS,
            )
        except (JSONParseError, StructuredOutputError, ValidationError) as exc:
            raise RecommendationError(
                f"Recommender LLM did not produce a usable recommendation: {exc}"
            ) from exc
        except LLMError:
            raise

    # ------------------------------------------------------------------
    # Evidence serialisation (input to the prompt)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence(
        experiments: List[ExperimentResult],
        statistical_results: _StatsInput,
        anomalies: List[AnomalyReport],
    ) -> Dict[str, Any]:
        # Kept compact on purpose - the prompt grows with every cycle and
        # hosted models meter tokens per minute. Redundant-across-rows fields
        # (task_type, experiment_id) and default preprocessing are dropped;
        # metrics are rounded.
        def _row(e: ExperimentResult) -> Dict[str, Any]:
            row: Dict[str, Any] = {
                "model": e.config.model_type,
                "hp": e.config.hyperparameters,
                "seed": e.config.random_seed,
                "status": e.status,
                "metrics": (
                    {k: round(v, 4) for k, v in e.metrics.items()}
                    if e.metrics
                    else None
                ),
            }
            if e.config.preprocessing.normalize:
                row["normalize"] = True
            return row

        exp_rows = [_row(e) for e in experiments]
        task_type = next((e.task_type for e in experiments if e.task_type), None)

        if statistical_results is None:
            stats_out: Any = {}
        elif isinstance(statistical_results, dict):
            stats_out = {
                name: comp.model_dump(mode="json")
                for name, comp in statistical_results.items()
            }
        else:
            stats_out = [comp.model_dump(mode="json") for comp in statistical_results]

        return {
            "task_type": task_type,
            "experiments": exp_rows,
            "statistical_comparisons": stats_out,
            "anomalies": [
                {"rule": a.rule, "severity": a.severity, "explanation": a.explanation}
                for a in anomalies
            ],
            "counts": {
                "total": len(experiments),
                "success": sum(1 for e in experiments if e.status == "success"),
                "failed": sum(1 for e in experiments if e.status == "failed"),
                "anomalous": sum(1 for e in experiments if e.status == "anomalous"),
            },
        }

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _build_recommendation(
        self, parsed: Dict[str, Any], dataset_id: str
    ) -> Recommendation:
        action = parsed.get("action")
        if action not in _VALID_ACTIONS:
            raise StructuredOutputError(
                f"'action' is {action!r}; must be one of {_VALID_ACTIONS}"
            )

        explanation = parsed.get("explanation")
        evidence_summary = parsed.get("evidence_summary")
        if not isinstance(explanation, str) or not explanation.strip():
            raise StructuredOutputError("response is missing an 'explanation' string")
        if not isinstance(evidence_summary, str) or not evidence_summary.strip():
            raise StructuredOutputError(
                "response is missing an 'evidence_summary' string"
            )

        recommended: List[ExperimentConfiguration] = []
        if action == "run_more_experiments":
            raw_recs = parsed.get("recommended_experiments") or []
            if not isinstance(raw_recs, list) or not raw_recs:
                raise StructuredOutputError(
                    "action is 'run_more_experiments' but 'recommended_experiments' "
                    "is empty"
                )
            recommended = self._validate_configs(raw_recs, dataset_id)
            recommended = self._repair_seeds(recommended)

        rec = Recommendation(
            action=action,  # type: ignore[arg-type]
            recommended_experiments=recommended,
            explanation=explanation.strip(),
            evidence_summary=evidence_summary.strip(),
        )
        logger.info(
            "Recommender -> action=%s, %d recommended experiment(s)",
            rec.action, len(rec.recommended_experiments),
        )
        return rec

    @staticmethod
    def _validate_configs(
        raw_recs: List[Any], dataset_id: str
    ) -> List[ExperimentConfiguration]:
        configs: List[ExperimentConfiguration] = []
        for i, item in enumerate(raw_recs):
            if not isinstance(item, dict):
                raise StructuredOutputError(
                    f"recommended experiment #{i} is not a JSON object: {item!r}"
                )
            item = strip_null_hyperparameters(dict(item))
            supplied_id = item.get("dataset_id")
            if supplied_id and supplied_id != dataset_id:
                logger.warning(
                    "Recommender set dataset_id=%r on experiment #%d; overriding with %r",
                    supplied_id, i, dataset_id,
                )
            item["dataset_id"] = dataset_id
            try:
                configs.append(ExperimentConfiguration(**item))
            except (ValidationError, TypeError) as exc:
                raise StructuredOutputError(
                    f"recommended experiment #{i} failed validation: {exc}"
                ) from exc
        return configs

    @staticmethod
    def _condition_key(config: ExperimentConfiguration) -> Any:
        return (
            config.model_type,
            config.preprocessing.normalize,
            tuple(sorted(config.hyperparameters.items())),
        )

    def _repair_seeds(
        self, configs: List[ExperimentConfiguration]
    ) -> List[ExperimentConfiguration]:
        """Dedup exact ``(condition, seed)`` repeats, then top every proposed
        condition up to >= ``_MIN_SEEDS_PER_CONDITION`` distinct seeds on fresh
        seed values. Deterministic. Mirrors ``ExperimentPlannerAgent._repair_seeds``.
        """
        # 1. drop exact (condition, seed) duplicates, keep first occurrence
        seen: set = set()
        deduped: List[ExperimentConfiguration] = []
        for c in configs:
            marker = (self._condition_key(c), c.random_seed)
            if marker in seen:
                logger.warning(
                    "Recommender proposed a duplicate config (seed=%d); dropping it",
                    c.random_seed,
                )
                continue
            seen.add(marker)
            deduped.append(c)

        # 2. top up short conditions
        groups: Dict[Any, List[ExperimentConfiguration]] = {}
        for c in deduped:
            groups.setdefault(self._condition_key(c), []).append(c)

        used_seeds = {c.random_seed for c in deduped}
        next_seed = (max(used_seeds) if used_seeds else 41) + 1
        out = list(deduped)
        for group in groups.values():
            seeds = {c.random_seed for c in group}
            if len(seeds) >= _MIN_SEEDS_PER_CONDITION:
                continue
            template = group[0]
            before = len(seeds)
            while len(seeds) < _MIN_SEEDS_PER_CONDITION:
                while next_seed in used_seeds:
                    next_seed += 1
                used_seeds.add(next_seed)
                seeds.add(next_seed)
                out.append(template.model_copy(update={"random_seed": next_seed}))
                next_seed += 1
            logger.info(
                "Recommender: topped up a proposed condition from %d to %d distinct seeds",
                before, _MIN_SEEDS_PER_CONDITION,
            )
        return out
