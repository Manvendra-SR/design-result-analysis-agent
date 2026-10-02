"""
backend/agents/planner.py
===========================
Planner: research question + dataset profile -> ``Plan``.

This is the step that genuinely needs language understanding: deciding which
factor a free-text question is about, which levels are worth trying, and which
level is the natural reference. The LLM fills a small schema
(factor / levels / reference / base configuration); it never writes raw
configurations or seeds. ``Plan`` validation then enforces the rules in code -
a real factor, 2-5 distinct levels, values in range - and a reply that breaks
one is sent back to the model once to fix.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.agents.llm import request_json
from backend.models.dataset import DatasetProfile
from backend.models.experiment import MLP_DEFAULTS
from backend.models.investigation import BaseSetup, Factor, Level, Plan

MAX_INITIAL_LEVELS = 5
FACTORS = list(Factor.__args__)  # type: ignore[attr-defined]


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
    "properties": {k: {"type": ["number", "null"]} for k in MLP_DEFAULTS},
    "required": list(MLP_DEFAULTS),
}

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "error": {"type": ["string", "null"]},
        "factor": {"type": "string", "enum": FACTORS},
        "levels": {"type": "array", "items": {"type": "string"}},
        "reference": {"type": "string"},
        "base": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model_type": {"type": "string", "enum": ["mlp", "linear_baseline"]},
                "normalize": {"type": "boolean"},
                "hyperparameters": _HP_SCHEMA,
            },
            "required": ["model_type", "normalize", "hyperparameters"],
        },
        "rationale": {"type": "string"},
    },
    "required": ["error", "factor", "levels", "reference", "base", "rationale"],
}

_SYSTEM_PROMPT = f"""\
You design a controlled ML experiment that answers a research question about \
one tabular dataset. You choose ONE factor to vary; everything else is held fixed.

Factors you may vary:
- "model_type": levels "mlp" and/or "linear_baseline" (logistic/linear regression)
- "normalize": levels "true"/"false" (StandardScaler fit on the training split)
- an mlp hyperparameter: "hidden_size" (1-512), "dropout" (0 to <1), \
"learning_rate" (0-1], "batch_size" (1-4096), "epochs" (1-100)

Rules:
1. Pick the single factor the question is about, and 2-{MAX_INITIAL_LEVELS} levels of it, as strings.
2. "reference" is the level the others are compared against - usually the \
default or simplest choice (e.g. dropout "0", normalize "false", "linear_baseline").
3. "base" is the configuration shared by every level. Keep it small and fast \
(hidden_size <= 128, epochs <= 30). Hyperparameters you do not set may be null \
(defaults: {MLP_DEFAULTS}). The factor's own value in "base" is ignored.
4. A hyperparameter factor requires base model_type "mlp".
5. Seeds and replicates are handled by the system - do not mention them.

If the question cannot be answered by varying one of these factors on this \
dataset, set "error" to one sentence explaining why (other fields may be empty). \
Otherwise set "error" to null and explain the design in "rationale" (2-3 sentences)."""


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
    factor = reply["factor"]
    levels = [parse_level(factor, v) for v in reply["levels"]]
    if len(levels) > MAX_INITIAL_LEVELS:
        raise ValueError(f"use at most {MAX_INITIAL_LEVELS} levels, got {len(levels)}")
    base = reply["base"]
    return Plan(
        factor=factor,
        levels=levels,
        reference=parse_level(factor, reply["reference"]),
        base=BaseSetup(
            model_type=base["model_type"],
            normalize=base["normalize"],
            hyperparameters={k: v for k, v in base["hyperparameters"].items() if v is not None},
        ),
        rationale=reply["rationale"].strip(),
    )


def describe_dataset(profile: DatasetProfile) -> str:
    lines = [
        "Dataset:",
        f"- task: {profile.task_type}, target column {profile.target_column!r}",
        f"- {profile.n_rows} rows, {len(profile.feature_columns)} feature columns "
        f"({len(profile.numeric_columns)} numeric, {len(profile.categorical_columns)} categorical)",
    ]
    if profile.task_type == "classification":
        lines.append(f"- {profile.n_classes} classes, distribution {profile.class_distribution}")
    return "\n".join(lines)
