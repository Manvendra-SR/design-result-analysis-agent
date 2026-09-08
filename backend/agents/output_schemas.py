"""
backend/agents/output_schemas.py
==================================
JSON Schemas for the Planner / Recommender LLM outputs, in the strict subset
that Groq's ``response_format: {"type": "json_schema", "strict": true}``
accepts (every object closed with ``additionalProperties: false``, every
property listed in ``required``, nullable expressed as a type union).

These are passed to ``GroqClient.chat_json(..., schema=...)`` as a strict
``response_format`` so generation is constrained to exactly this shape.

They mirror ``ExperimentConfiguration`` / ``ExperimentPlan`` /
``Recommendation`` with two accommodations the agents clean up afterwards:

- ``dataset_id`` is a required string in the schema (the prompt tells the
  model to echo it, and strict mode with ``additionalProperties: false``
  would otherwise *reject* the generation), but the agent overwrites whatever
  the model put there with the session's real dataset id;
- ``hyperparameters`` lists every mlp key as nullable; the agent drops the
  nulls before building ``ExperimentConfiguration`` (so ``linear_baseline``
  ends up with ``{}`` and mlp keeps only the values the model set).
"""

from __future__ import annotations

from typing import Any, Dict

_HYPERPARAMETERS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dropout": {"type": ["number", "null"]},
        "learning_rate": {"type": ["number", "null"]},
        "batch_size": {"type": ["integer", "null"]},
        "hidden_size": {"type": ["integer", "null"]},
        "epochs": {"type": ["integer", "null"]},
    },
    "required": ["dropout", "learning_rate", "batch_size", "hidden_size", "epochs"],
}

EXPERIMENT_CONFIG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        # required so strict generation validates; the agent overwrites it.
        "dataset_id": {"type": "string"},
        "model_type": {"type": "string", "enum": ["mlp", "linear_baseline"]},
        "hyperparameters": _HYPERPARAMETERS_SCHEMA,
        "preprocessing": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"normalize": {"type": "boolean"}},
            "required": ["normalize"],
        },
        "random_seed": {"type": "integer"},
    },
    "required": [
        "dataset_id",
        "model_type",
        "hyperparameters",
        "preprocessing",
        "random_seed",
    ],
}

EXPERIMENT_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        # non-null only when the question cannot be answered with these models
        "error": {"type": ["string", "null"]},
        "experiments": {"type": "array", "items": EXPERIMENT_CONFIG_SCHEMA},
        "explanation": {"type": "string"},
    },
    "required": ["error", "experiments", "explanation"],
}

RECOMMENDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["run_more_experiments", "conclude"]},
        "recommended_experiments": {
            "type": "array",
            "items": EXPERIMENT_CONFIG_SCHEMA,
        },
        "explanation": {"type": "string"},
        "evidence_summary": {"type": "string"},
    },
    "required": [
        "action",
        "recommended_experiments",
        "explanation",
        "evidence_summary",
    ],
}


def strip_null_hyperparameters(config_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``hyperparameters`` keys whose value is ``null``.

    The strict schema forces every mlp hyperparameter to be present; the
    model sets the ones it does not want to ``null``. ``ExperimentConfiguration``
    expects a plain ``Dict[str, float]``, so the nulls are removed here.
    """
    out = dict(config_obj)
    hp = out.get("hyperparameters")
    if isinstance(hp, dict):
        out["hyperparameters"] = {k: v for k, v in hp.items() if v is not None}
    return out
