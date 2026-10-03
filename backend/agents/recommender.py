"""
backend/agents/recommender.py
===============================
Recommender: two narrow LLM jobs.

``decide`` - after a round, look at the validation evidence and choose between
    *refine*:   1-3 new values of ONE knob of ONE parent candidate (e.g. "the
                forest overfits - its train-validation gap is 25 points - so try
                min_samples_leaf 5 and 20"). In effect mode the knob is the
                plan's factor and the parent is the reference.
    *conclude*: nothing untried is likely to change the answer.
  That judgement - which experiment could change the conclusion? - is the
  open-ended part. Everything mechanical is code: only contenders may be
  refined, only on a knob of their own family, values must be in range, a
  configuration is never run twice, and the round budget and the "settled"
  rule are enforced by the loop (the LLM is not even called then).

``interpret`` - turn the final, code-computed numbers into a short
  plain-language explanation. It cites numbers; it never computes them.

Context stays bounded: each call is stateless and sends one compact summary
row per candidate (never runs, rows or earlier conversation), and the number
of candidates is bounded by the round budget.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from backend.agents.llm import request_json
from backend.agents.planner import parse_level
from backend.models.experiment import FAMILIES
from backend.models.investigation import MAX_NEW_CANDIDATES, Analysis, Decision, Plan, Report

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["refine", "conclude"]},
        "parent": {"type": ["string", "null"]},
        "knob": {"type": ["string", "null"]},
        "values": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["action", "parent", "knob", "values", "rationale"],
}

INTERPRETATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"interpretation": {"type": "string"}},
    "required": ["interpretation"],
}

_DECIDE_PROMPT = f"""\
You steer an ML experiment. You see every candidate configuration tried so far \
on the VALIDATION split: its mean metric, its score on the training split and \
the gap between them (positive = overfitting, near zero = possibly underfitting), \
and paired differences with 95% bootstrap CIs as [diff, low, high] against the \
reference ("vs_ref"), the current leader ("vs_leader") and the candidate it was \
derived from ("vs_parent"). All numbers were computed by code - do not recompute them.

Choose:
- "refine": the experiment most likely to change the answer. Give a "parent" \
(selection mode: one of the contenders), ONE "knob" from "refinable", and 1-{MAX_NEW_CANDIDATES} \
untried "values" as strings, within the ranges given. In effect mode the parent is \
the reference and the knob is the factor (both may be null).
- "conclude": the evidence already answers the question, or nothing untried could \
change it.

Do not ask for more seeds or replicates - those are fixed by the system. \
Keep "rationale" to 2-3 sentences that cite the numbers you were given."""

_INTERPRET_PROMPT = """\
You explain the result of an ML experiment to a practitioner in 3-5 sentences. \
All numbers were computed by code: cite them, never invent or recompute any. \
Candidates were explored on the validation split; the winner was then compared \
once on a held-out test split (with the reference, and in selection mode with the \
best model of another family), so those comparisons are not biased by the \
exploration. "inconclusive" means the models are statistically tied on this data. \
Say what the answer to the research question is, how confident the confidence \
intervals allow us to be, and one honest caveat: results hold for the candidates \
tried (see "effort") on this dataset and training split only, and are not a claim \
that the best possible model was found."""


def _r(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


def evidence(question: str, plan: Plan, analysis: Analysis, round: int, max_rounds: int) -> Dict[str, Any]:
    """The compact, bounded summary the Recommender sees: one row per candidate."""
    diffs: Dict[str, Dict[str, List[float]]] = {}
    for c in analysis.comparisons:
        key = {"reference": "vs_ref", "leader": "vs_leader", "parent": "vs_parent"}[c.anchor]
        diffs.setdefault(c.a, {})[key] = [_r(c.diff), _r(c.ci_low), _r(c.ci_high)]  # type: ignore[list-item]
    rows = []
    for s in analysis.conditions:
        row = {"id": s.id, "family": s.family, "parent": s.parent, "change": s.change, "val": _r(s.mean),
               "seed_std": _r(s.seed_std), "train": _r(s.train_metric), "gap": _r(s.gap),
               "failed_runs": s.n_failed or None, **diffs.get(s.id, {})}
        if plan.mode == "selection":
            row["contender"] = s.contender
        rows.append({k: v for k, v in row.items() if v is not None})

    if plan.mode == "selection":
        families = {plan.get(c).model_type for c in analysis.contenders}
        refinable = {
            f: {k: [FAMILIES[f].knobs[k].low, FAMILIES[f].knobs[k].high] for k in FAMILIES[f].refinable}
            for f in sorted(families) if FAMILIES[f].refinable
        }
    else:
        knobs = FAMILIES[plan.base.model_type].knobs  # type: ignore[union-attr]
        if plan.factor in knobs:
            options: Any = [knobs[plan.factor].low, knobs[plan.factor].high]
        elif plan.factor == "model_type":
            options = [f for f in FAMILIES if f not in {c.model_type for c in plan.candidates}]
        else:
            options = [True, False]
        refinable = {plan.factor: options}
    return {
        "question": question,
        "mode": plan.mode,
        "metric": f"{analysis.metric} ({'higher' if analysis.higher_is_better else 'lower'} is better)",
        "val_rows": analysis.n_rows,
        "majority_rate": _r(analysis.majority_rate),
        "reference": plan.reference,
        "leader": analysis.leader,
        "round": round,
        "rounds_left": max_rounds - round,
        "candidates": rows,
        "refinable": refinable,
    }


def compact(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"))


class Recommender:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def decide(self, question: str, plan: Plan, analysis: Analysis, round: int, max_rounds: int) -> Decision:
        messages = [
            {"role": "system", "content": _DECIDE_PROMPT},
            {"role": "user", "content": compact(evidence(question, plan, analysis, round, max_rounds))},
        ]
        contenders = analysis.contenders if plan.mode == "selection" else None

        def build(reply: Dict[str, Any]) -> Decision:
            rationale = reply["rationale"].strip()
            if reply["action"] == "conclude":
                return Decision(round=round, action="conclude", rationale=rationale)
            knob = plan.factor if plan.mode == "effect" else reply["knob"]
            parent = plan.reference if plan.mode == "effect" else reply["parent"]
            if not knob:
                raise ValueError("refine needs a knob")
            values = [parse_level(knob, v) for v in reply["values"]]
            plan.refine(parent, knob, values, round + 1, contenders)  # every rule; raises ValueError
            return Decision(round=round, action="refine", parent=parent, knob=knob, values=values,
                            rationale=rationale)

        return request_json(self._llm, messages, DECISION_SCHEMA, build=build, temperature=0.3)

    def interpret(self, question: str, plan: Plan, report: Report) -> str:
        facts = {
            "research_question": question,
            "mode": plan.mode,
            "design_rationale": plan.rationale,
            "result": report.model_dump(mode="json", exclude={"interpretation"}),
        }
        messages = [
            {"role": "system", "content": _INTERPRET_PROMPT},
            {"role": "user", "content": compact(facts)},
        ]
        return request_json(
            self._llm, messages, INTERPRETATION_SCHEMA,
            build=lambda reply: reply["interpretation"].strip(), temperature=0.3, reasoning_effort="low",
        )
