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
15.2 Uses the Ollama HTTP API
15.8 Does not proceed on malformed LLM output
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import ValidationError

from backend.agents._common import request_json_object
from backend.agents.llm_client import LLMError, OllamaClient
from backend.agents.parsing import JSONParseError
from backend.models.anomaly import AnomalyReport
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.recommendation import Recommendation
from backend.models.statistics import StatisticalComparison

logger = logging.getLogger(__name__)

_VALID_ACTIONS = ("run_more_experiments", "conclude")
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
        Ollama adapter. Defaults to a fresh ``OllamaClient``. Injected in tests.
    """

    def __init__(self, llm_client: Optional[OllamaClient] = None) -> None:
        self._llm = llm_client or OllamaClient()

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
            If the Ollama server cannot be reached.
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
                    f"Evidence (JSON):\n{json.dumps(evidence, indent=2, default=str)}\n\n"
                    "Recommend the next action now."
                ),
            },
        ]

        try:
            parsed = request_json_object(self._llm, messages, options=_RECOMMENDER_OPTIONS)
        except JSONParseError as exc:
            raise RecommendationError(
                f"Recommender LLM did not return valid JSON after a repair attempt: {exc}"
            ) from exc
        except LLMError:
            raise

        return self._build_recommendation(parsed, dataset_id)

    # ------------------------------------------------------------------
    # Evidence serialisation (input to the prompt)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence(
        experiments: List[ExperimentResult],
        statistical_results: _StatsInput,
        anomalies: List[AnomalyReport],
    ) -> Dict[str, Any]:
        exp_rows = [
            {
                "experiment_id": e.experiment_id[:8],
                "model_type": e.config.model_type,
                "hyperparameters": e.config.hyperparameters,
                "preprocessing": e.config.preprocessing.model_dump(),
                "random_seed": e.config.random_seed,
                "status": e.status,
                "task_type": e.task_type,
                "metrics": e.metrics,
            }
            for e in experiments
        ]

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
            "experiments": exp_rows,
            "statistical_comparisons": stats_out,
            "anomalies": [a.model_dump(mode="json") for a in anomalies],
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
            raise RecommendationError(
                f"Recommender returned an invalid action {action!r}; "
                f"expected one of {_VALID_ACTIONS}."
            )

        explanation = parsed.get("explanation")
        evidence_summary = parsed.get("evidence_summary")
        if not isinstance(explanation, str) or not explanation.strip():
            raise RecommendationError("Recommender response is missing an 'explanation'.")
        if not isinstance(evidence_summary, str) or not evidence_summary.strip():
            raise RecommendationError(
                "Recommender response is missing an 'evidence_summary'."
            )

        recommended: List[ExperimentConfiguration] = []
        if action == "run_more_experiments":
            raw_recs = parsed.get("recommended_experiments") or []
            if not isinstance(raw_recs, list) or not raw_recs:
                raise RecommendationError(
                    "action is 'run_more_experiments' but no recommended "
                    "experiments were provided."
                )
            recommended = self._validate_configs(raw_recs, dataset_id)

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
                raise RecommendationError(
                    f"Recommended experiment #{i} is not a JSON object: {item!r}"
                )
            item = dict(item)
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
                raise RecommendationError(
                    f"Recommended experiment #{i} failed validation: {exc}"
                ) from exc
        return configs
