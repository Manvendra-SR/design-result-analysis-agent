"""
backend/agents/planner.py
===========================
Experiment_Planner_Agent: turns a natural-language research question about a
profiled dataset into a validated batch of controlled experiment
configurations.

What the LLM does vs. what this module does
-------------------------------------------
The LLM chooses *which* factor to vary and *what* values to try. This module
is the deterministic guard rail around that choice:

- rebuilds every configuration through ``ExperimentConfiguration`` so the
  Phase 2 hyperparameter-range validator runs (dropout 0-1, lr > 0, ...);
- forces ``dataset_id`` to the real dataset (a small local model cannot be
  trusted to echo a UUID, and there is exactly one valid dataset per
  session), logging a warning if the model supplied a different one;
- **guarantees Requirement 2.3 by deterministic repair**: any condition the
  LLM gave fewer than 3 distinct random seeds is topped up with extra
  configs on fresh, unused seeds (rather than rejecting the whole plan, which
  a 7B model triggers routinely). See ``_repair_seeds``.
- enforces that the plan compares at least two conditions (Requirement 2.2
  - "vary one factor at a time" only makes sense with >= 2 levels).

Structured-output retry
-----------------------
The LLM call goes through ``request_structured`` (``agents/_common.py``):
parse + validate + build happen inside a bounded retry loop, so a reply that
is valid JSON but *semantically* wrong (missing ``explanation``, an
out-of-range hyperparameter) is fed back to the model to fix before the
agent gives up with ``PlanValidationError``.

"Exactly one factor at a time" is requested in the prompt but not machine
-enforced - deciding which of several changing keys is "the factor" is
ambiguous, and a strict check would reject legitimate plans.

Unanswerable questions
----------------------
The prompt instructs the model to reply ``{"error": "..."}`` when the
question cannot be addressed with ``mlp`` / ``linear_baseline`` on this
dataset. The agent raises ``PlanningError`` in that case (Requirement 2.6),
which the Phase 6 API layer turns into an HTTP 400 - mirroring how
``ingest_csv`` raises ``DatasetValidationError``.

Requirements
------------
2.1  Parse the question, identify hyperparameters to vary
2.2  One factor varied at a time
2.3  >= 3 random seeds per configuration
2.4  dataset/model/hyperparameters/seed specified explicitly
2.5  Validate the question is answerable with available models + task type
2.6  Return an error explaining the limitation when it is not
2.7  Configurations designed to run quickly (small models)
15.1 Uses the LLM backend (Groq) via the LLMClient protocol
15.8 Does not proceed on malformed LLM output (bounded validation retry, then error)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from backend.agents._common import StructuredOutputError, request_structured
from backend.agents.client_factory import create_llm_client
from backend.agents.llm_client import LLMClient, LLMError
from backend.agents.output_schemas import (
    EXPERIMENT_PLAN_SCHEMA,
    strip_null_hyperparameters,
)
from backend.agents.parsing import JSONParseError
from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentPlan

logger = logging.getLogger(__name__)

_MIN_SEEDS_PER_CONDITION = 3
_MIN_CONDITIONS = 2
# Low temperature: the planner's value is valid, well-formed configs, not
# creative prose.
_PLANNER_OPTIONS: Dict[str, Any] = {"temperature": 0.2}

_SYSTEM_PROMPT = """\
You are an ML experiment designer. Given a research question about a specific \
tabular dataset and that dataset's profile, produce a set of controlled \
experiment configurations that answer the question.

Rules:
1. Vary exactly ONE factor at a time. Hold every other hyperparameter fixed \
across the whole plan.
2. Include at least THREE different random_seed values for every distinct \
configuration, for statistical validity.
3. Produce at least two distinct conditions (for example two dropout values, \
or mlp vs linear_baseline).
4. Prefer small, fast models: hidden_size <= 128, epochs <= 30.
5. Every configuration MUST set dataset_id to the exact id given below.

Available models:
- "mlp": feed-forward network. Hyperparameters (optional, sensible defaults \
applied when omitted): hidden_size (integer > 0), dropout (float 0-1), \
learning_rate (float > 0), batch_size (integer > 0), epochs (integer > 0).
- "linear_baseline": logistic or linear regression, chosen automatically by \
the dataset's task type. NO tunable hyperparameters - use \
"hyperparameters": {}.

Preprocessing: {"normalize": true|false} fits a StandardScaler on the \
training split only. This is the one preprocessing option you may vary.

If the question cannot be answered with these two models on this dataset \
(for example it is not about this dataset, or it needs a model type that is \
not available), respond with EXACTLY this and nothing else:
{"error": "<one sentence explaining why>"}

Otherwise respond with ONLY a JSON object of this shape:
{
  "experiments": [
    {
      "dataset_id": "<the id given below>",
      "model_type": "mlp",
      "hyperparameters": {"dropout": 0.0, "learning_rate": 0.001, "batch_size": 32, "hidden_size": 64, "epochs": 20},
      "preprocessing": {"normalize": false},
      "random_seed": 42
    }
  ],
  "explanation": "<why this design answers the research question>"
}"""


class PlanningError(ValueError):
    """The research question cannot be turned into a valid experiment plan.

    Raised when the LLM declares the question unanswerable, when it fails to
    return usable JSON, or when the generated configurations fail validation
    (see the ``PlanValidationError`` subclass).
    """


class PlanValidationError(PlanningError):
    """The LLM returned a plan, but the configurations violate a hard rule.

    Examples: an out-of-range hyperparameter, fewer than 3 seeds for a
    condition, or fewer than 2 conditions.
    """


class ExperimentPlannerAgent:
    """LLM-based agent that translates research questions into experiment plans.

    Parameters
    ----------
    llm_client:
        An ``LLMClient``. Defaults to the provider from ``LLM_PROVIDER``
        (``create_llm_client``). Injected in tests.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client or create_llm_client()

    def plan_experiments(
        self, research_question: str, dataset_profile: DatasetProfile
    ) -> ExperimentPlan:
        """Generate a validated experiment plan for ``research_question``.

        Parameters
        ----------
        research_question:
            Natural-language question about model behaviour on this dataset.
        dataset_profile:
            The profile of the already-ingested dataset the session is
            scoped to.

        Returns
        -------
        ExperimentPlan

        Raises
        ------
        PlanningError
            If the question is unanswerable or the LLM output is unusable.
        PlanValidationError
            If the generated configurations violate a hard rule that repair
            cannot fix (fewer than 2 conditions, an out-of-range value).
        LLMError
            If the LLM backend cannot be reached.
        """
        if not research_question or not research_question.strip():
            raise PlanningError("Research question is empty.")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(research_question, dataset_profile)},
        ]

        def _build(parsed: Dict[str, Any]) -> ExperimentPlan:
            # A genuine "cannot be answered" reply is terminal - not retried.
            if isinstance(parsed.get("error"), str) and parsed["error"].strip():
                raise PlanningError(parsed["error"].strip())

            raw_experiments = parsed.get("experiments")
            explanation = parsed.get("explanation") or ""
            if not isinstance(raw_experiments, list) or not raw_experiments:
                raise StructuredOutputError(
                    "response has no non-empty 'experiments' array"
                )
            if not isinstance(explanation, str) or not explanation.strip():
                raise StructuredOutputError("response is missing an 'explanation' string")

            configs = self._validate_configs(raw_experiments, dataset_profile)
            self._require_min_conditions(configs)
            configs = self._repair_seeds(configs)
            return ExperimentPlan(experiments=configs, explanation=explanation.strip())

        try:
            plan = request_structured(
                self._llm,
                messages,
                build=_build,
                schema=EXPERIMENT_PLAN_SCHEMA,
                options=_PLANNER_OPTIONS,
            )
        except (JSONParseError, StructuredOutputError, ValidationError) as exc:
            raise PlanValidationError(
                f"Planner LLM did not produce a usable plan: {exc}"
            ) from exc
        except LLMError:
            raise  # transport failure - let it surface unchanged

        logger.info(
            "Planner produced %d configurations across %d conditions for dataset %s",
            plan.total_count,
            len(self._group_by_condition(plan.experiments)),
            dataset_profile.dataset_id,
        )
        return plan

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(research_question: str, profile: DatasetProfile) -> str:
        lines = [
            f'Research question: "{research_question.strip()}"',
            "",
            "Dataset profile:",
            f"- task_type: {profile.task_type}",
            f'- target column: "{profile.target_column}"',
            f"- feature columns: {len(profile.feature_columns)} "
            f"({len(profile.numeric_columns)} numeric, "
            f"{len(profile.categorical_columns)} categorical)",
            f"- model input_dim after one-hot encoding: {profile.n_features}",
            f"- rows: {profile.n_rows}",
        ]
        if profile.task_type == "classification":
            lines.append(f"- n_classes: {profile.n_classes}")
            if profile.class_distribution:
                lines.append(f"- class distribution: {profile.class_distribution}")
        lines += [
            "",
            f"dataset_id (use this exact value on every configuration): {profile.dataset_id}",
            "",
            "Design the experiments now.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_configs(
        self, raw_experiments: List[Any], profile: DatasetProfile
    ) -> List[ExperimentConfiguration]:
        configs: List[ExperimentConfiguration] = []
        for i, item in enumerate(raw_experiments):
            if not isinstance(item, dict):
                raise StructuredOutputError(
                    f"experiment #{i} is not a JSON object: {item!r}"
                )
            # strict-schema replies carry every mlp hyperparameter as a key,
            # with null for the ones the model did not choose - drop those.
            item = strip_null_hyperparameters(dict(item))
            supplied_id = item.get("dataset_id")
            if supplied_id and supplied_id != profile.dataset_id:
                logger.warning(
                    "Planner set dataset_id=%r on experiment #%d; overriding with %r",
                    supplied_id, i, profile.dataset_id,
                )
            item["dataset_id"] = profile.dataset_id

            try:
                configs.append(ExperimentConfiguration(**item))
            except (ValidationError, TypeError) as exc:
                raise StructuredOutputError(
                    f"experiment #{i} failed validation: {exc}"
                ) from exc

        return configs

    def _require_min_conditions(
        self, configs: List[ExperimentConfiguration]
    ) -> None:
        """A plan with < 2 conditions has nothing to compare - retryable."""
        groups = self._group_by_condition(configs)
        if len(groups) < _MIN_CONDITIONS:
            raise StructuredOutputError(
                f"plan must compare at least {_MIN_CONDITIONS} conditions "
                f"(one factor varied at >= 2 levels); got {len(groups)}"
            )

    def _repair_seeds(
        self, configs: List[ExperimentConfiguration]
    ) -> List[ExperimentConfiguration]:
        """Guarantee >= ``_MIN_SEEDS_PER_CONDITION`` distinct seeds per condition.

        Deterministic: for any short condition, clone its first config onto
        fresh seeds (the smallest integers above every seed already used in
        the plan, skipping collisions). A plan the LLM already got right
        passes through unchanged.
        """
        groups = self._group_by_condition(configs)
        used_seeds = {c.random_seed for c in configs}
        next_seed = (max(used_seeds) if used_seeds else 41) + 1
        repaired = list(configs)

        for key, group in groups.items():
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
                repaired.append(template.model_copy(update={"random_seed": next_seed}))
                next_seed += 1
            logger.info(
                "planner: topped up condition %s from %d to %d distinct seeds",
                self._describe_condition(key), before, _MIN_SEEDS_PER_CONDITION,
            )
        return repaired

    @staticmethod
    def _group_by_condition(
        configs: List[ExperimentConfiguration],
    ) -> Dict[Tuple[Any, ...], List[ExperimentConfiguration]]:
        """Group configs by everything except random_seed (a 'condition')."""
        groups: Dict[Tuple[Any, ...], List[ExperimentConfiguration]] = {}
        for c in configs:
            key = (
                c.model_type,
                c.preprocessing.normalize,
                tuple(sorted(c.hyperparameters.items())),
            )
            groups.setdefault(key, []).append(c)
        return groups

    @staticmethod
    def _describe_condition(key: Tuple[Any, ...]) -> str:
        model_type, normalize, hp = key
        hp_str = ", ".join(f"{k}={v}" for k, v in hp) or "no hyperparameters"
        return f"({model_type}, normalize={normalize}, {hp_str})"
