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
none of them. The p-values, effect sizes, confidence intervals and replicate
counts in the prompt were all produced by ``StatisticalAnalyzer`` (Phase 3).

Cumulative evidence, explicitly attributed
-------------------------------------------
The agent sees the whole investigation, not just the latest cycle - that is
what makes the loop adaptive. What it previously *lacked* was attribution:
experiments and anomalies arrived with no cycle on them, so in a cycle that
flagged nothing it would still open with "two runs were flagged as anomalous",
which read as corruption next to a per-cycle header showing zero. Every
experiment and anomaly now carries its cycle, anomalies are split into open
(still hold) and resolved (withdrawn - history, not a to-do), and the prompt
requires earlier-cycle references to be labelled as such.

The agent is also told where it is in the cycle budget, so the final cycle is
a deliberate conclusion rather than a silent cut-off by the safety cap.

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
``_MIN_SEEDS_PER_CONDITION`` seeds, all of them matched to the seeds the other
conditions use (1, 2, 3, ...) - see ``models/condition.assign_matched_seeds``.
Exact ``(condition, seed)`` duplicates are dropped first. Because the agent is
told which experiments already exist, a condition that has already been run
with seeds 1-3 is topped up with 4, 5, 6 rather than repeating runs whose
results are already on record.

Controlled comparisons - guidance, not a machine rule
-------------------------------------------------------
The prompt asks the agent to carry the established configuration forward and
change only the factor under test, and - when a second change is genuinely
warranted - to name it and justify it in ``explanation``. This mirrors the
Planner's "vary one factor at a time" rule, which the Recommender previously
was never given at all: it re-specified a full configuration every cycle with
nothing anchoring it to the design, and a real run answered "does an MLP beat
a linear baseline?" by quietly switching to a different MLP (four fields
changed at once) and never re-running the baseline.

Like the Planner's rule, this is deliberately **not** enforced in code. No
check counts how many fields differ from a previous cycle and no
recommendation is rejected for changing several at once - deciding which of
several changed keys is "the factor" is ambiguous, and a strict check would
reject the legitimate case where several parameters must move together. The
agent keeps the freedom; the prompt tells it what a controlled comparison is
and asks it to show its reasoning when it departs from one.

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
from backend.models.condition import ConditionKey, assign_matched_seeds, condition_key
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.recommendation import Recommendation
from backend.models.statistics import AnalysisSnapshot, StatisticalComparison

logger = logging.getLogger(__name__)

_VALID_ACTIONS = ("run_more_experiments", "conclude")
_MIN_SEEDS_PER_CONDITION = 3  # matches planner._MIN_SEEDS_PER_CONDITION
# Slightly higher than the planner: the recommender writes a paragraph of
# reasoning, but structure/action still matter more than prose style.
_RECOMMENDER_OPTIONS: Dict[str, Any] = {"temperature": 0.3}

_StatsInput = Union[
    AnalysisSnapshot,
    Dict[str, StatisticalComparison],
    Sequence[StatisticalComparison],
    None,
]

_SYSTEM_PROMPT = """\
You are an adaptive ML experiment recommender. You are given a research \
question, every experiment run so far, statistical comparisons that have \
ALREADY been computed with scipy, per-condition summary statistics, and the \
anomaly flags. Decide what to do next.

You do NOT compute statistics. Cite the p-values, effect sizes, confidence \
intervals and counts you are given - do not recalculate or second-guess them, \
and do not estimate replicate counts yourself: use the numbers in \
"condition_summaries" and "counts".

VOCABULARY (do not confuse these):
- an "experiment" is ONE complete training run of one configuration;
- a "condition" is a configuration without its random seed;
- a "replicate" is one experiment within a condition, differing only by seed;
- "epochs" is how many passes over the training data happen INSIDE a single \
experiment - it is never a count of experiments.

EVIDENCE IS CUMULATIVE. You are shown the whole investigation, not just the \
latest cycle. Every experiment and anomaly carries the cycle it belongs to. \
When you refer to something from an earlier cycle, SAY SO explicitly (for \
example "the run flagged in cycle 1"), so a reader looking at the current \
cycle is not misled.

ANOMALIES have a lifecycle. Detection re-runs over all evidence every cycle, \
so a flag raised against a thin group can later be withdrawn. Only entries in \
"open_anomalies" still hold. Entries in "resolved_anomalies" are HISTORY - \
they have already been dealt with, and you must NOT ask for more experiments \
to resolve them again.

CONTROLLED COMPARISONS. The investigation is trying to answer ONE research \
question, and a comparison only answers it when the conditions being compared \
differ in the factor under test and are otherwise alike. So when you propose \
a new configuration, start from the configuration already being investigated \
and change the factor you are actually testing; hold the other \
hyperparameters, the model_type and the normalize setting at the values the \
existing conditions use. Copy them across explicitly rather than omitting \
them - a key you leave out is not "unchanged", it is a different \
configuration.

You MAY change more than one parameter at once when there is a specific \
scientific reason - for example a learning rate that no longer suits a much \
larger hidden layer, or a configuration that has to change to make a fair \
comparison possible at all. This is a judgement you are trusted to make. When \
you do it, say so plainly in your "explanation": name every parameter you \
changed and why that change was necessary, so a reader can see it was a \
deliberate design decision rather than drift. Never silently redesign the \
configuration you are comparing against - if the established comparison still \
lacks evidence, extend that comparison instead of replacing it.

Reasoning patterns:
- Open anomalies -> recommend re-running those configurations with new seeds.
- High within-condition variance, or fewer than 5 successful replicates in a \
condition that is not deterministic -> recommend more seeds for it.
- A condition marked "deterministic": its result does not depend on the \
random seed, so extra seeds add NO information. Never ask for more replicates \
of a deterministic condition; if you need more evidence there, vary a \
hyperparameter instead.
- "skipped_comparisons" are knowledge gaps - read the reason and recommend \
experiments that would close it.
- A promising trend in one hyperparameter -> recommend nearby values.
- A high proportion of failed experiments -> adjust the configuration.
- Sufficient evidence -> conclude. That means: a comparison that answers the \
question (significant at p < 0.05, or a tight confidence interval around no \
difference), at least 5 successful replicates in each non-deterministic \
condition, and no open anomalies.

CYCLE BUDGET. "cycles_remaining" tells you how many further cycles the system \
will allow. When it is 0 this is the LAST cycle: you MUST answer "conclude" \
and give the best answer the evidence supports, stating plainly what is still \
uncertain. Do not ask for experiments that can never run.

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


def _as_snapshot(statistical_results: _StatsInput) -> AnalysisSnapshot:
    """Normalise whatever the caller passed into an ``AnalysisSnapshot``.

    The state machine always passes a full snapshot. Tests and scripts that
    only have a mapping or list of comparisons are still accepted - they just
    carry no skips or condition summaries.
    """
    if statistical_results is None:
        return AnalysisSnapshot()
    if isinstance(statistical_results, AnalysisSnapshot):
        return statistical_results
    if isinstance(statistical_results, dict):
        return AnalysisSnapshot(comparisons=list(statistical_results.values()))
    return AnalysisSnapshot(comparisons=list(statistical_results))


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
        current_cycle: Optional[int] = None,
        max_cycles: Optional[int] = None,
    ) -> Recommendation:
        """Analyse the evidence and recommend the next action.

        Parameters
        ----------
        research_question:
            The original question driving the session.
        experiments:
            Every experiment run in the session so far (success, failed,
            anomalous). Must be non-empty. This is deliberately cumulative -
            each row carries its ``cycle`` so the agent can attribute what it
            cites to the cycle it came from.
        statistical_results:
            Pre-computed output from ``StatisticalAnalyzer``. Normally an
            ``AnalysisSnapshot`` (comparisons + skipped pairs + per-condition
            summaries); a bare mapping or sequence of comparisons is also
            accepted for callers that have nothing else.
        anomalies:
            Anomaly reports for this session, open and resolved. They are
            split in the prompt so the agent never chases a flag that has
            already been withdrawn.
        current_cycle:
            The 1-based cycle this recommendation closes.
        max_cycles:
            The safety cap. Together with ``current_cycle`` this tells the
            agent how many cycles remain, so on the last one it can conclude
            deliberately instead of being cut off mid-investigation.

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
        evidence = self._build_evidence(
            experiments,
            statistical_results,
            anomalies or [],
            current_cycle=current_cycle,
            max_cycles=max_cycles,
        )
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
                build=lambda parsed: self._build_recommendation(
                    parsed, dataset_id, self._seeds_already_run(experiments)
                ),
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
        current_cycle: Optional[int] = None,
        max_cycles: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Serialise the cumulative evidence for the prompt.

        Kept compact on purpose - it grows with every cycle and hosted models
        meter tokens per minute - but every fact the agent is expected to
        reason about is *attributed*:

        - each experiment carries the ``cycle`` that produced it, so the agent
          can say "flagged in cycle 1" instead of speaking timelessly;
        - anomalies are split into open (still hold) and resolved (history,
          already dealt with), each with the cycle it was raised/withdrawn in;
        - ``condition_summaries`` carry the replicate counts, so the agent
          never has to count rows out of the JSON - it used to get those
          counts wrong in its prose;
        - ``cycle_budget`` says how many cycles remain, so the last cycle can
          be a deliberate conclusion rather than an abrupt cut-off.
        """
        # Redundant-across-rows fields (task_type, experiment_id) and default
        # preprocessing are dropped; metrics are rounded.
        def _row(e: ExperimentResult) -> Dict[str, Any]:
            row: Dict[str, Any] = {
                "cycle": e.cycle,
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

        # Anomalies name their experiment by (cycle, seed, model) rather than a
        # UUID the agent cannot match to anything in the experiment rows.
        by_id = {e.experiment_id: e for e in experiments}

        def _anomaly(a: AnomalyReport) -> Dict[str, Any]:
            exp = by_id.get(a.experiment_id)
            out: Dict[str, Any] = {
                "rule": a.rule,
                "severity": a.severity,
                "explanation": a.explanation,
                "detected_in_cycle": a.detected_cycle,
            }
            if exp is not None:
                out["experiment"] = {
                    "cycle": exp.cycle,
                    "model": exp.config.model_type,
                    "seed": exp.config.random_seed,
                }
            if not a.is_open:
                out["withdrawn_in_cycle"] = a.resolved_cycle
            return out

        exp_rows = [_row(e) for e in experiments]
        task_type = next((e.task_type for e in experiments if e.task_type), None)
        analysis = _as_snapshot(statistical_results)

        evidence: Dict[str, Any] = {
            "task_type": task_type,
            "evidence_scope": (
                "cumulative - every experiment run in this investigation so far"
            ),
            "experiments": exp_rows,
            "condition_summaries": [
                s.model_dump(mode="json") for s in analysis.condition_summaries
            ],
            "statistical_comparisons": [
                c.model_dump(mode="json") for c in analysis.comparisons
            ],
            "skipped_comparisons": [s.model_dump(mode="json") for s in analysis.skipped],
            "open_anomalies": [_anomaly(a) for a in anomalies if a.is_open],
            "resolved_anomalies": [_anomaly(a) for a in anomalies if not a.is_open],
            "counts": {
                "total": len(experiments),
                "success": sum(1 for e in experiments if e.status == "success"),
                "failed": sum(1 for e in experiments if e.status == "failed"),
                "anomalous": sum(1 for e in experiments if e.status == "anomalous"),
            },
        }

        if current_cycle is not None and max_cycles is not None:
            evidence["cycle_budget"] = {
                "current_cycle": current_cycle,
                "max_cycles": max_cycles,
                "cycles_remaining": max(0, max_cycles - current_cycle),
            }
        return evidence

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    @staticmethod
    def _seeds_already_run(
        experiments: List[ExperimentResult],
    ) -> Dict[ConditionKey, set]:
        """Which seeds each condition has already been run with in this session.

        Lets ``assign_matched_seeds`` extend a condition (4, 5, 6) instead of
        re-proposing seeds whose results are already recorded.
        """
        seen: Dict[ConditionKey, set] = {}
        for e in experiments:
            seen.setdefault(condition_key(e.config), set()).add(e.config.random_seed)
        return seen

    def _build_recommendation(
        self,
        parsed: Dict[str, Any],
        dataset_id: str,
        existing_seeds: Optional[Dict[ConditionKey, set]] = None,
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
            recommended = self._repair_seeds(recommended, existing_seeds)

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

    def _repair_seeds(
        self,
        configs: List[ExperimentConfiguration],
        existing_seeds: Optional[Dict[ConditionKey, set]] = None,
    ) -> List[ExperimentConfiguration]:
        """Drop exact ``(condition, seed)`` repeats, then renumber every
        proposed condition onto matched seeds (1, 2, 3, ... continuing past the
        seeds that condition has already been run with), topping each up to at
        least ``_MIN_SEEDS_PER_CONDITION`` replicates. Deterministic. Mirrors
        ``ExperimentPlannerAgent._repair_seeds``.
        """
        seen: set = set()
        deduped: List[ExperimentConfiguration] = []
        for c in configs:
            marker = (condition_key(c), c.random_seed)
            if marker in seen:
                logger.warning(
                    "Recommender proposed a duplicate config (seed=%d); dropping it",
                    c.random_seed,
                )
                continue
            seen.add(marker)
            deduped.append(c)

        repaired = assign_matched_seeds(
            deduped,
            min_replicates=_MIN_SEEDS_PER_CONDITION,
            existing_seeds=existing_seeds,
        )
        logger.info(
            "Recommender: %d proposed config(s) -> %d on matched seeds",
            len(configs), len(repaired),
        )
        return repaired
