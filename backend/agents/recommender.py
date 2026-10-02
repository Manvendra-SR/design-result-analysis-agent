"""
backend/agents/recommender.py
===============================
Recommender: two narrow LLM jobs.

``decide`` - after a round, look at the validation results and choose between
    *explore*: try 1-3 new levels of the SAME factor (e.g. "0.2 beat 0.0 and
               0.5, so try 0.1 and 0.3"), or
    *conclude*: the levels tried are enough to answer the question.
  That judgement - is another region of the factor worth a round? - is the
  open-ended part. Everything mechanical is code: the factor cannot change,
  levels are validated, already-tried levels are rejected, and the round
  budget is enforced by the loop (the LLM is not even called on the last round).

``interpret`` - turn the final, code-computed numbers into a short
  plain-language explanation. It cites numbers; it never computes them.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from backend.agents.llm import request_json
from backend.agents.planner import parse_level
from backend.models.investigation import Analysis, Decision, Plan, Report, format_value

MAX_NEW_LEVELS = 3

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["explore", "conclude"]},
        "new_levels": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["action", "new_levels", "rationale"],
}

INTERPRETATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"interpretation": {"type": "string"}},
    "required": ["interpretation"],
}

_DECIDE_PROMPT = f"""\
You steer a controlled ML experiment. One factor is being varied; you see how \
every level tried so far performs on the VALIDATION split, each compared with \
the reference level (difference and 95% bootstrap confidence interval, computed \
by code - do not recompute them).

Choose:
- "explore": propose 1-{MAX_NEW_LEVELS} NEW levels of the SAME factor, when the \
results suggest an untried region could change the answer (e.g. the best level \
sits at the edge of the range tried, or between two levels). Levels are strings, \
e.g. "0.3" or "true".
- "conclude": the levels tried already answer the question, or nothing \
untried is worth a round.

Do not ask for more seeds or replicates - those are fixed by the system. \
Keep "rationale" to 2-3 sentences that cite the numbers you were given."""

_INTERPRET_PROMPT = """\
You explain the result of a controlled ML experiment to a practitioner in \
3-5 sentences. All numbers were computed by code: cite them, never invent or \
recompute any. Levels were explored on the validation split; the final \
comparison (best level vs reference) was run once on a held-out test split, \
so it is not biased by that exploration. Say what the answer to the research \
question is, how confident the confidence interval allows us to be, and one \
honest caveat (e.g. results are for this dataset and training split only)."""


class Recommender:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def decide(self, question: str, plan: Plan, analysis: Analysis, round: int, max_rounds: int) -> Decision:
        evidence = {
            "research_question": question,
            "factor": plan.factor,
            "reference": plan.label(plan.reference),
            "levels_tried": [format_value(v) for v in plan.levels],
            "metric": f"{analysis.metric} ({'higher' if analysis.higher_is_better else 'lower'} is better)",
            "validation_rows": analysis.n_rows,
            "majority_class_rate": analysis.majority_rate,
            "conditions": [c.model_dump(mode="json", exclude={"level"}) for c in analysis.conditions],
            "comparisons_vs_reference": [c.model_dump(mode="json", exclude={"level"}) for c in analysis.comparisons],
            "rounds_left_after_this": max_rounds - round,
        }
        messages = [
            {"role": "system", "content": _DECIDE_PROMPT},
            {"role": "user", "content": json.dumps(evidence, indent=1)},
        ]

        def build(reply: Dict[str, Any]) -> Decision:
            rationale = reply["rationale"].strip()
            if reply["action"] == "conclude":
                return Decision(round=round, action="conclude", rationale=rationale)
            new = [parse_level(plan.factor, v) for v in reply["new_levels"]]
            if not 1 <= len(new) <= MAX_NEW_LEVELS:
                raise ValueError(f"explore needs 1-{MAX_NEW_LEVELS} new levels, got {len(new)}")
            tried = {format_value(v) for v in plan.levels}
            repeats = [format_value(v) for v in new if format_value(v) in tried]
            if repeats:
                raise ValueError(f"levels {repeats} were already tried; propose untried levels or conclude")
            Plan.model_validate({**plan.model_dump(), "levels": [*plan.levels, *new]})  # range/type checks
            return Decision(round=round, action="explore", new_levels=new, rationale=rationale)

        return request_json(self._llm, messages, DECISION_SCHEMA, build=build, temperature=0.3)

    def interpret(self, question: str, plan: Plan, report: Report) -> str:
        facts = {
            "research_question": question,
            "factor": plan.factor,
            "design_rationale": plan.rationale,
            "result": report.model_dump(mode="json", exclude={"interpretation"}),
        }
        messages = [
            {"role": "system", "content": _INTERPRET_PROMPT},
            {"role": "user", "content": json.dumps(facts, indent=1)},
        ]
        return request_json(
            self._llm, messages, INTERPRETATION_SCHEMA,
            build=lambda reply: reply["interpretation"].strip(), temperature=0.3,
        )
