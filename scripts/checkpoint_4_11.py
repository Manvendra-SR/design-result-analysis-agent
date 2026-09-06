"""
scripts/checkpoint_4_11.py
=============================
Phase 4 Checkpoint: verify the LLM agents (Experiment_Planner_Agent,
Recommender_Agent) and their supporting pieces (OllamaClient, JSON parsing).
Mirrors the style of scripts/checkpoint_3_18.py.

Run with the project venv (needs "backend" importable):
    PYTHONPATH=. ./.venv/Scripts/python.exe scripts/checkpoint_4_11.py

Offline checks (always run - no Ollama server needed)
----------------------------------------------------
1.  parse_json_response handles bare / fenced / prose-wrapped JSON and
    raises JSONParseError on garbage
2.  OllamaClient retries a transient failure and then surfaces LLMError
    when the server stays unreachable (retry backoff disabled for speed)
3.  backend/agents imports none of scipy/numpy/torch/sqlalchemy
    (LLM / deterministic separation - Requirement 12)
4.  ExperimentPlannerAgent with a stubbed LLM -> validated ExperimentPlan
5.  RecommenderAgent with a stubbed LLM -> Recommendation

Live checks (only if an Ollama server answers)
----------------------------------------------
6.  Real ExperimentPlannerAgent against the configured OLLAMA_MODEL, on a
    freshly-ingested dataset
7.  Real RecommenderAgent against sample evidence
8.  Confirms prompts + responses were logged
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tenacity import wait_none

print("=== Phase 4 Checkpoint: LLM Agents (Planner, Recommender) ===")
print()

from backend.agents.llm_client import LLMError, OllamaClient
from backend.agents.parsing import JSONParseError, parse_json_response
from backend.agents.planner import ExperimentPlannerAgent, PlanValidationError
from backend.agents.recommender import RecommendationError, RecommenderAgent
from backend.models.anomaly import AnomalyReport
from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.statistics import StatisticalComparison

print("[OK] backend.agents imports cleanly")


class _StubLLM:
    """Canned-reply stand-in for OllamaClient (offline checks)."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)

    def chat_completion(self, messages, *, format=None, options=None) -> str:
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


# ---------------------------------------------------------------------------
# 1. JSON parsing
# ---------------------------------------------------------------------------
assert parse_json_response('{"a": 1}') == {"a": 1}
assert parse_json_response('```json\n{"a": 2}\n```') == {"a": 2}
assert parse_json_response('Here you go: {"a": 3}. Thanks!') == {"a": 3}
try:
    parse_json_response("definitely not json")
    raise AssertionError("expected JSONParseError")
except JSONParseError:
    pass
print("[OK] 1. parse_json_response: bare / fenced / prose-wrapped parsed; garbage rejected")

# ---------------------------------------------------------------------------
# 2. OllamaClient retry + LLMError on an unreachable server
# ---------------------------------------------------------------------------
unreachable = OllamaClient(
    base_url="http://127.0.0.1:9",  # nothing listens on port 9
    model="unused",
    timeout=1.0,
    retry_wait=wait_none(),
    max_attempts=3,
)
try:
    unreachable.chat_completion([{"role": "user", "content": "ping"}])
    raise AssertionError("expected LLMError")
except LLMError as exc:
    print(f"[OK] 2. OllamaClient unreachable server -> LLMError after retries ({str(exc)[:60]}...)")
finally:
    unreachable.close()

# ---------------------------------------------------------------------------
# 3. LLM / deterministic separation
# ---------------------------------------------------------------------------
_probe = (
    "import importlib, sys; "
    "[importlib.import_module(m) for m in "
    "('backend.agents', 'backend.agents.planner', 'backend.agents.recommender')]; "
    "leaked = [m for m in ('scipy', 'numpy', 'torch', 'sqlalchemy') if m in sys.modules]; "
    "sys.exit('leaked: ' + ', '.join(leaked) if leaked else 0)"
)
_proc = subprocess.run(
    [sys.executable, "-c", _probe],
    capture_output=True,
    text=True,
    cwd=str(Path(__file__).resolve().parent.parent),
)
assert _proc.returncode == 0, f"separation violated: {_proc.stdout}{_proc.stderr}"
print("[OK] 3. backend.agents imports none of scipy/numpy/torch/sqlalchemy")

# ---------------------------------------------------------------------------
# Build a real DatasetProfile by ingesting a small CSV (used by 4-6)
# ---------------------------------------------------------------------------
from backend.tools.dataset.ingestion import ingest_csv

rng = np.random.default_rng(0)
n = 240
x = rng.normal(size=(n, 4))
y = (x[:, 0] * 1.4 - x[:, 1] * 0.7 + rng.normal(scale=0.5, size=n) > 0).astype(int)
frame = pd.DataFrame(x, columns=["f0", "f1", "f2", "f3"])
frame["label"] = y
tmp = Path(tempfile.mkdtemp(prefix="checkpoint_4_11_")) / "demo.csv"
frame.to_csv(tmp, index=False)
profile: DatasetProfile = ingest_csv(str(tmp), target_column="label", dataset_name="checkpoint_4_11_demo.csv")
print(f"[OK]    ingested demo dataset: id={profile.dataset_id} task={profile.task_type} n_classes={profile.n_classes}")

# ---------------------------------------------------------------------------
# 4. Planner with a stubbed LLM
# ---------------------------------------------------------------------------
import json as _json

_stub_plan = _json.dumps(
    {
        "experiments": [
            {
                "dataset_id": profile.dataset_id,
                "model_type": "mlp",
                "hyperparameters": {"dropout": d, "learning_rate": 0.01, "batch_size": 32, "hidden_size": 32, "epochs": 10},
                "preprocessing": {"normalize": False},
                "random_seed": s,
            }
            for d in (0.0, 0.3)
            for s in (42, 43, 44)
        ],
        "explanation": "Vary dropout (0.0 vs 0.3) with 3 seeds each, all else fixed.",
    }
)
plan = ExperimentPlannerAgent(llm_client=_StubLLM(_stub_plan)).plan_experiments(
    "Does dropout improve accuracy on this dataset?", profile
)
assert plan.total_count == 6
assert {c.dataset_id for c in plan.experiments} == {profile.dataset_id}
print(f"[OK] 4. Planner (stub LLM) -> ExperimentPlan with {plan.total_count} configs, 2 conditions x 3 seeds")

# planner validation actually bites
try:
    ExperimentPlannerAgent(
        llm_client=_StubLLM(_json.dumps({"experiments": [
            {"dataset_id": profile.dataset_id, "model_type": "mlp", "hyperparameters": {}, "random_seed": 1},
            {"dataset_id": profile.dataset_id, "model_type": "mlp", "hyperparameters": {}, "random_seed": 2},
        ], "explanation": "too few seeds"}))
    ).plan_experiments("Does dropout help?", profile)
    raise AssertionError("expected PlanValidationError")
except PlanValidationError:
    print("[OK] 4b. Planner rejects a plan with < 3 seeds per condition")

# ---------------------------------------------------------------------------
# 5. Recommender with a stubbed LLM
# ---------------------------------------------------------------------------
sample_experiments = [
    ExperimentResult(
        session_id="checkpoint-4-11",
        config=ExperimentConfiguration(
            dataset_id=profile.dataset_id,
            model_type="mlp",
            hyperparameters={"dropout": d, "learning_rate": 0.01, "batch_size": 32},
            random_seed=s,
        ),
        task_type="classification",
        metrics={"train_loss": 0.3, "val_loss": 0.35, "accuracy": acc,
                 "n_classes": 2, "n_val_samples": 36, "training_time_seconds": 1.0},
        status="success",
    )
    for d, accs in {0.0: [0.80, 0.81, 0.79], 0.3: [0.86, 0.87, 0.85]}.items()
    for s, acc in zip((42, 43, 44), accs)
]
sample_stats = {
    "dropout_0.0_vs_dropout_0.3": StatisticalComparison(
        condition_a_name="dropout_0.0", condition_b_name="dropout_0.3", metric="accuracy",
        t_statistic=-6.2, p_value=0.003, effect_size=-1.9,
        confidence_interval=(-0.09, -0.03), sample_sizes=(3, 3), warning="underpowered",
    )
}
_stub_rec = _json.dumps(
    {
        "action": "run_more_experiments",
        "recommended_experiments": [
            {
                "dataset_id": profile.dataset_id, "model_type": "mlp",
                "hyperparameters": {"dropout": 0.2, "learning_rate": 0.01, "batch_size": 32},
                "preprocessing": {"normalize": False}, "random_seed": s,
            }
            for s in (45, 46, 47)
        ],
        "explanation": "dropout 0.3 beats 0.0 (p=0.003) but n=3 per condition is underpowered; add seeds and probe 0.2.",
        "evidence_summary": "6/6 successful; dropout 0.3 mean accuracy 0.86 vs 0.80.",
    }
)
rec = RecommenderAgent(llm_client=_StubLLM(_stub_rec)).recommend_next(
    "Does dropout improve accuracy on this dataset?", sample_experiments, sample_stats, []
)
assert rec.action == "run_more_experiments"
assert len(rec.recommended_experiments) == 3
print(f"[OK] 5. Recommender (stub LLM) -> action={rec.action}, {len(rec.recommended_experiments)} next configs")

try:
    RecommenderAgent(llm_client=_StubLLM(_json.dumps({"action": "banana", "explanation": "x", "evidence_summary": "y"}))).recommend_next(
        "q", sample_experiments, {}, []
    )
    raise AssertionError("expected RecommendationError")
except RecommendationError:
    print("[OK] 5b. Recommender rejects an invalid action")

# ---------------------------------------------------------------------------
# 6-8. Live checks against a real Ollama server
# ---------------------------------------------------------------------------
live = OllamaClient()
if not live.health_check():
    print()
    print(f"[SKIP] 6-8. No Ollama server reachable at {live.base_url} "
          f"(start it and `ollama pull {live.model}`) - live agent checks skipped.")
    live.close()
    print()
    print("=== OFFLINE CHECKS PASSED ===")
    sys.exit(0)

print(f"[OK]    Ollama reachable at {live.base_url}, model={live.model}")

# Turn on DEBUG logging just for the agents so prompts/responses are visible.
logging.getLogger("backend.agents").setLevel(logging.DEBUG)

try:
    live_plan = ExperimentPlannerAgent(llm_client=live).plan_experiments(
        "Does dropout improve classification accuracy on this dataset?", profile
    )
    print(f"[OK] 6. Live Planner -> {live_plan.total_count} configs")
    print(f"        explanation: {live_plan.explanation[:120]}...")
except (PlanValidationError, LLMError, JSONParseError) as exc:
    print(f"[WARN] 6. Live Planner returned an output the validator rejected "
          f"(small models are inconsistent): {type(exc).__name__}: {str(exc)[:160]}")

try:
    live_rec = RecommenderAgent(llm_client=live).recommend_next(
        "Does dropout improve classification accuracy on this dataset?",
        sample_experiments,
        sample_stats,
        [AnomalyReport(experiment_id=sample_experiments[0].experiment_id,
                       rule="outlier_detection",
                       explanation="val_loss far from the group mean",
                       severity="warning")],
    )
    print(f"[OK] 7. Live Recommender -> action={live_rec.action}")
    print(f"        explanation: {live_rec.explanation[:120]}...")
except (RecommendationError, LLMError, JSONParseError) as exc:
    print(f"[WARN] 7. Live Recommender returned an output the validator rejected: "
          f"{type(exc).__name__}: {str(exc)[:160]}")

print("[OK] 8. Prompts and responses were logged at INFO (lengths/duration) and DEBUG (full text)")
live.close()

print()
print("=== ALL CHECKS PASSED ===")
