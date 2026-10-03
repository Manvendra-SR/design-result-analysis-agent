"""
backend/agents/planner.py
===========================
Planner: research question + dataset profile -> ``Plan``.

This is the step that genuinely needs language understanding: deciding what
kind of question a free-text question is, and turning it into a design.

effect     "Does X help?" - which factor, which levels, which level is the
           natural reference, and the configuration they share.
selection  "Which model is best?" - which model families are worth comparing.
           Code starts each one at its registry defaults (a fair start) and
           makes the simplest one the reference.

The LLM fills a small schema; it never writes raw configurations or seeds.
``Plan`` construction then enforces the rules in code - a real factor or
family, 2-5 distinct levels, values in range - and a reply that breaks one is
sent back to the model once to fix.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.agents.llm import request_json
from backend.models.dataset import DatasetProfile
from backend.models.experiment import ALL_KNOBS, FAMILIES
from backend.models.investigation import FACTORS, FAMILY_ORDER, BaseSetup, Level, Plan

MAX_INITIAL_LEVELS = 5
MAX_CLASSES_SHOWN = 10


class PlanningError(Exception):
    """The question cannot be answered with the available models on this dataset."""


def parse_level(factor: str, raw: str) -> Level:
    """Turn the LLM's string level into a typed value (ValueError if impossible)."""
    text = raw.strip()
    if factor == "model_type":
        return text
    if factor == "normalize":
        if text.lower() not in ("true", "false"):
            raise ValueError(f"normalize levels must be 'true' or 'false', got {raw!r}")
        return text.lower() == "true"
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{factor} levels must be numbers, got {raw!r}") from None


_HP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {k: {"type": ["number", "null"]} for k in ALL_KNOBS},
    "required": list(ALL_KNOBS),
}

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "error": {"type": ["string", "null"]},
        "mode": {"type": "string", "enum": ["effect", "selection"]},
        "families": {"type": "array", "items": {"type": "string", "enum": FAMILY_ORDER}},
        "factor": {"type": ["string", "null"], "enum": [*FACTORS, None]},
        "levels": {"type": "array", "items": {"type": "string"}},
        "reference": {"type": ["string", "null"]},
        "base": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model_type": {"type": "string", "enum": FAMILY_ORDER},
                "normalize": {"type": ["boolean", "null"]},
                "hyperparameters": _HP_SCHEMA,
            },
            "required": ["model_type", "normalize", "hyperparameters"],
        },
        "rationale": {"type": "string"},
    },
    "required": ["error", "mode", "families", "factor", "levels", "reference", "base", "rationale"],
}


def _knob_lines() -> str:
    return "\n".join(
        f"- {family}: " + (", ".join(
            f"{k} ({v.low:g}-{v.high:g})" for k, v in spec.knobs.items()) or "no hyperparameters")
        for family, spec in FAMILIES.items()
    )


_SYSTEM_PROMPT = f"""\
You design an ML experiment that answers a research question about one \
tabular dataset. First decide which kind of question it is:

- "selection": which model is best / should be used. List 2-4 "families" worth \
comparing. Code starts each at its defaults and makes the simplest the reference. \
Leave factor/levels/reference empty (null / []).
- "effect": does ONE thing help or matter (dropout, normalization, depth, one model \
vs another...). Choose ONE "factor", 2-{MAX_INITIAL_LEVELS} "levels" (as strings) \
and the "reference" level the others are compared against - usually the default \
or simplest choice (e.g. dropout "0", normalize "false", "linear_baseline"). \
Everything else is held fixed in "base"; leave "families" empty.

Model families and their hyperparameters (ranges):
{_knob_lines()}
Factors: "model_type" (levels are family names), "normalize" ("true"/"false"), \
or a hyperparameter of the base family.

Rules:
1. Keep "base" small and fast (hidden_size <= 128, epochs <= 30). Hyperparameters \
you do not set are null and take their defaults. normalize null = the family default.
2. Seeds and replicates are handled by the system - do not mention them.

If the question cannot be answered with these models on this dataset, set "error" \
to one sentence explaining why (other fields may be empty). Otherwise set "error" \
to null and explain the design in "rationale" (2-3 sentences)."""


class Planner:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def plan(self, question: str, profile: DatasetProfile) -> Plan:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Research question: {question.strip()}\n\n{describe_dataset(profile)}"},
        ]
        return request_json(self._llm, messages, PLAN_SCHEMA, build=_build_plan)


def _build_plan(reply: Dict[str, Any]) -> Plan:
    if reply.get("error"):
        raise PlanningError(reply["error"])
    rationale = reply["rationale"].strip()
    if reply["mode"] == "selection":
        return Plan.selection(reply["families"], rationale)
    factor = reply["factor"]
    if not factor or reply["reference"] is None:
        raise ValueError("an effect plan needs a factor and a reference level")
    levels = [parse_level(factor, v) for v in reply["levels"]]
    if len(levels) > MAX_INITIAL_LEVELS:
        raise ValueError(f"use at most {MAX_INITIAL_LEVELS} levels, got {len(levels)}")
    base = reply["base"]
    return Plan.effect(
        factor=factor,
        levels=levels,
        reference=parse_level(factor, reply["reference"]),
        base=BaseSetup(
            model_type=base["model_type"],
            normalize=base["normalize"],
            hyperparameters={k: v for k, v in base["hyperparameters"].items() if v is not None},
        ),
        rationale=rationale,
    )


def describe_dataset(profile: DatasetProfile) -> str:
    """Counts only - never rows or column names. Bounded: at most MAX_CLASSES_SHOWN classes."""
    lines = [
        "Dataset:",
        f"- task: {profile.task_type}, target column {profile.target_column!r}",
        f"- {profile.n_rows} rows, {len(profile.feature_columns)} feature columns "
        f"({len(profile.numeric_columns)} numeric, {len(profile.categorical_columns)} categorical)",
    ]
    if profile.task_type == "classification":
        dist = sorted((profile.class_distribution or {}).items(), key=lambda kv: -kv[1])
        shown = dict(dist[:MAX_CLASSES_SHOWN])
        more = f" and {len(dist) - MAX_CLASSES_SHOWN} more" if len(dist) > MAX_CLASSES_SHOWN else ""
        lines.append(f"- {profile.n_classes} classes, distribution {shown}{more}")
    return "\n".join(lines)
