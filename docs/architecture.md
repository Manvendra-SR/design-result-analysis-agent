# Architecture

Text companion to [`architecture.html`](architecture.html) (interactive click-through of the same flows) and [`pipeline_flow.tex`](pipeline_flow.tex) (the algorithm explained from scratch: a mental model, then Selection Mode and Effect Mode each end to end, with real runs). Everything here matches the code on the current branch; file paths are given so each claim can be checked.

**In one sentence:** an LLM turns a research question into a small experiment design, deterministic code derives, trains and statistically compares the candidate configurations on the validation split, the LLM chooses which one-knob refinement of a model still in contention could change the answer (within hard limits enforced by code), and the answer comes from comparisons fixed in advance and run once on a held-out test split. LLMs handle language and judgement; code handles every rule and every number.

## Components

| Component | Kind | Where | Responsibility |
|---|---|---|---|
| React UI | client | `frontend/` | Upload a CSV, ask a question, start; polls `GET /api/sessions/{id}` every 2 s while running |
| FastAPI | API | `backend/api/` | 9 endpoints under `/api` + `/health`; runs an investigation as a background task |
| CSV store | persistence | `data/uploads/<id>/data.csv` | The cleaned CSV (rows with missing values dropped); read with a fixed 70/15/15 split |
| PostgreSQL | persistence | `backend/database/` | 3 tables (`datasets`, `sessions`, `experiments`) behind `Repository` |
| Investigation loop | orchestration | `backend/state_machine/graph.py` | LangGraph: `plan_experiments → run_experiments → analyze → decide → (run_experiments \| finalize)` |
| Planner | LLM | `backend/agents/planner.py` | Question + dataset counts → an *effect* plan or a *selection* plan |
| Recommender | LLM | `backend/agents/recommender.py` | After a round: `refine` one knob of one candidate, or `conclude`; at the end: `interpret` |
| Family registry | deterministic | `backend/models/experiment.py` | `FAMILIES`: per model family its knobs (range, default), seeded?, default preprocessing, refinable knobs |
| Plan / candidates | deterministic | `backend/models/investigation.py` | `Plan.effect`, `Plan.selection`, `Plan.refine`: every configuration is derived by code |
| Experiment runner | deterministic | `backend/tools/trainers.py`, `experiment_runner.py` | Trains one configuration; per-row validation/test scores; crash or NaN/Inf → `failed` |
| Statistics | deterministic | `backend/tools/stats.py` | Paired bootstrap CIs, leader, contenders, settled rule, final selection, test comparisons |
| Groq | LLM provider | `backend/agents/llm.py` | `openai/gpt-oss-120b`, strict JSON-schema output, retries, output cap, usage logging |

## Investigation modes

The Planner classifies the question; code builds the candidates.

| | **effect** — "does X help?" | **selection** — "which model is best?" |
|---|---|---|
| LLM fills | `factor`, 2–5 `levels`, `reference` level, `base` configuration | 2–4 `families` |
| Code derives | one candidate per level: the base with only the factor changed; the reference is the root `c1`, every other level is its child | one root per family at its registry defaults and default preprocessing; the simplest chosen family (order linear → tree → forest → MLP) is the reference `c1` |
| Factors / families | `model_type`, `normalize`, or any knob of the base family | `linear_baseline`, `decision_tree`, `random_forest`, `mlp` |
| Refinement | new levels of the **same** factor (parent = reference) | one knob of **one contender**, from that family's refinable knobs |
| Validation views | each candidate vs the reference | vs the reference, vs the leader, vs its parent |
| Test comparisons | best non-reference candidate vs reference (95%) | that, plus vs the best candidate of another family (97.5% each) |

## Model families (`FAMILIES`)

| Family | Trainer | Knobs (range, default) | Seeded | Default `normalize` | Refinable |
|---|---|---|---|---|---|
| `linear_baseline` | LogisticRegression (`max_iter=1000`) / LinearRegression | — | no (1 run) | true | — |
| `decision_tree` | DecisionTree, `random_state=0` | `max_depth` 1–30 (8), `min_samples_leaf` 1–200 (1) | no (1 run) | false | both |
| `random_forest` | RandomForest, 200 trees, `random_state=seed` | `max_depth` 2–40 (library default), `min_samples_leaf` 1–100 (1), `max_features` 0.1–1.0 (library default) | yes | false | all three |
| `mlp` | `TabularMLP`: one hidden layer, ReLU, Adam | `hidden_size` 1–512 (64), `dropout` [0, 1) (0), `learning_rate` (0, 1] (0.001), `batch_size` 16–4096 (32), `epochs` 1–50 (20) | yes | true | all but `batch_size` |

## Candidates and lineage

A `Candidate` is one configuration without a seed: `id`, `label`, `model_type`, `hyperparameters`, `normalize`, and its lineage — `parent`, the one `knob` changed and its `value`, and the `round` it was added in. Runs are mapped back to candidates by `ExperimentConfiguration.key()` (the configuration minus the seed), which also rejects any configuration that was already run. `Plan` validation enforces the lineage: in effect mode every non-reference candidate is the reference with only the factor changed; in selection mode each family appears once as a root and every child changes one refinable knob of a same-family parent.

## Execution

`configs_for` expands each new candidate into `N_SEEDS` runs (seeds 0..2) for seeded families, one run otherwise. `run_experiment` trains one configuration on `Dataset.load_split` — the same rows for every run (`split_seed` = 42), one-hot encoding fixed at ingestion, `StandardScaler` fit on the training split only when `normalize` is true. Each run returns the validation metric, the same metric on the training split, `train_loss`, training time, and **one score per validation and test row** (1/0 correct for classification, squared error for regression). A run is `failed` only if it crashed or produced NaN/Inf; every other run is evidence. Each result is stored as it finishes.

## Statistics (validation, every round)

`stats.analyze(plan, results, profile, "val")`, recomputed from the stored runs — never stored:

- Per candidate: per-row scores averaged over its successful seeds; mean, seed spread (stability, not uncertainty), training-split metric and the train–validation gap (positive = overfitting).
- Paired bootstrap of `d_i = score_a(i) − score_b(i)` over the validation rows (2,000 resamples, percentile 95% CI); verdict *better* / *worse* when the CI excludes zero, else *inconclusive*.
- Pairs: each candidate vs the reference; in selection mode also each candidate vs the **leader** (best mean) and each refined child vs its **parent**.
- **Contenders** (selection mode): candidates whose CI against the leader does not lie entirely on the worse side of zero.

## The adaptive loop (`decide`)

Code runs first: if the round is `MAX_ROUNDS` → conclude (`budget`); if, in selection mode, every contender belongs to one family → conclude (`settled`). Only otherwise is the Recommender called. It receives one compact JSON object: the question, mode, metric, number of validation rows, majority-class rate, reference and leader ids, round and rounds left, one row per candidate (family, parent, change, validation mean, seed spread, train metric, gap, `[diff, low, high]` vs reference / leader / parent, contender flag) and the refinable knob ranges. It replies `{action, parent, knob, values, rationale}`. `Plan.refine` then checks every rule — effect mode keeps its factor and parent; selection mode refines only a contender, only on a refinable knob of its own family; 1–3 values, each in range; no configuration twice — and a violation is sent back to the model once. Accepted values become new candidates for the next round; earlier runs remain evidence.

## Final evaluation and report (`finalize`)

1. **Select, from validation only:** winner *W* = best non-reference candidate; in selection mode, runner-up *R* = best candidate whose family is neither *W*'s nor the reference's.
2. **Test, read once:** *W* vs reference (primary) and, if *R* exists, *W* vs *R* (secondary), each at confidence 1 − 0.05/k for k = 1 or 2 comparisons (Bonferroni: 95% or 97.5%).
3. **Report:** absolute test scores of *W* and the reference with bootstrap CIs, both comparisons, *W*'s validation-to-test drop, candidates per family (`effort`), rounds run, who stopped the loop (`agent` / `budget` / `settled`), majority-class rate, and a deterministic headline. If the reference itself led on validation, the headline says that its strongest alternative was tested. The Recommender's `interpret` call adds a few sentences that cite these numbers; if it fails, the report is saved without them.

## LLM interaction

All three calls go through `request_json`: a fresh two-message conversation (no history), strict JSON-schema output, a business-rule validator whose `ValueError` is fed back to the model once, transport retries for timeouts / 429 (honouring `Retry-After`) / 5xx, `max_completion_tokens` = 2,048, `reasoning_effort` (medium for plan and decide, low for interpret; only sent to gpt-oss models) and `usage` logged per response. The planner sees dataset **counts** only (task, target name, row and column counts, at most 10 classes) — never rows or column names. No call ever receives per-row scores or test results before finalize.

## Hard limits

| Limit | Value | Enforced by |
|---|---|---|
| Rounds | `MAX_ROUNDS` = 3 | `decide_node` concludes on the last round without an LLM call |
| New candidates per decision | 1–3 (`MAX_NEW_CANDIDATES`) | `Plan.refine` |
| Initial levels / families | 2–5 / 2–4 | `Planner`, `Plan.selection` |
| Runs per candidate | `N_SEEDS` = 3 (seeded), 1 otherwise | `configs_for` |
| One run's cost | MLP `epochs` ≤ 50, `batch_size` ≥ 16 | `FAMILIES` range checks |
| LLM calls | ≤ `MAX_ROUNDS` + 1 = 4 | plan + ≤ 2 decisions + interpret |
| Runs per investigation | ≤ 26 selection (8 + 2 × 3 × 3), ≤ 33 effect ((5 + 2 × 3) × 3) | the limits above |
| LLM context | ≈ 1,000 input tokens per call measured, bounded by the candidate count | stateless, compact payloads |

## Persistence and API

| Stored | Recomputed on read | In memory only |
|---|---|---|
| dataset profile; session status/error, plan (with all candidates), decisions, report; every run with its per-row scores | validation analysis (every `GET`), round count | LangGraph state during one run |

`POST /api/datasets`, `GET /api/datasets`, `GET|DELETE /api/datasets/{id}` (409 while in use unless `?cascade=true`); `POST /api/sessions`, `GET /api/sessions`, `GET|DELETE /api/sessions/{id}`; `POST /api/sessions/{id}/run` → 202 (409 if running or done; a failed session is reset first). `GET /api/sessions/{id}` returns `SessionDetail`: plan, decisions, runs (without per-row scores), the validation analysis and the report. Failures (`LLMError`, `PlanningError`, an invalid refinement, nothing to compare) mark the session `failed` with the error text; at startup any session left `running` is marked failed. The JSON columns need no migration, but plans from before the candidate model no longer validate — reset with `python -m backend.database.init_db --reset`.

## GPU vs CPU

`trainers.DEVICE` is `cuda` when available, else `cpu`; only the MLP uses it, with identical code path, seeds and batch order. A default MLP run (hidden 64, batch 32, 20 epochs) on the 10,500-row `diabetes_risk.csv` is launch-bound, so the GPU buys little at batch 32: ≈ 26–36 s per run measured with CUDA. The sklearn families run on the CPU (forest ≈ 1.6–3.2 s, tree and logistic ≈ 0.2 s).
