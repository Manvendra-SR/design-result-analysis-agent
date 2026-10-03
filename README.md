# Adaptive ML Experiment Agent

An agentic system that answers a research question about your own tabular dataset. Upload a CSV and ask something like *"Does dropout help here?"* or *"Which model performs best on this data?"*. The agent designs a controlled experiment, runs it, decides which experiment could still change the answer, and reports the answer with confidence intervals from a held-out test set.

## How it works

```
User: dataset + question
        │
        ▼
1. PLAN (LLM)     -> effect question ("does X help?"): one factor, its levels, a reference level
        │            selection question ("which model?"): 2-4 model families at their defaults,
        │            the simplest one as reference. Code derives every configuration.
        ▼
┌─► 2. RUN        train each new candidate (3 seeds for random forest / MLP, 1 otherwise)
│       ▼
│   3. CHECK      crashed / NaN runs are excluded and shown; everything else is evidence
│       ▼
│   4. ANALYZE    VALIDATION split, paired bootstrap CIs: each candidate vs the reference,
│       │         vs the current leader, and vs its parent; contenders = not clearly worse
│       ▼         than the leader
│   5. DECIDE     last round -> conclude; all contenders one family -> conclude (code).
└──── refine         Otherwise the LLM refines ONE knob of ONE contender (1-3 values) or
        ▼ conclude   concludes; code validates every proposal
6. FINALIZE       pick the winner (and the best other family) on validation, then unseal
        ▼         the TEST split once: winner vs reference [+ vs runner-up], Bonferroni
Report: deterministic headline + CIs, plus an LLM-written interpretation
```

Model families: logistic / linear regression (`linear_baseline`), decision tree, random forest and a one-hidden-layer MLP. Their knobs, ranges, defaults and default preprocessing live in one registry (`FAMILIES` in `backend/models/experiment.py`).

**The split of responsibilities is the main design decision.** LLMs do the parts that need language understanding: turning a free-text question into an experiment design, judging whether another round is worth running, and explaining the result. Deterministic code does everything that has to be correct: training, statistics, validation of the LLM's proposals, the round budget and every number in the report.

## Statistical method

- **What counts as a sample:** an evaluation row, not a training run. Each run scores every validation and test row (1/0 for classification, squared error for regression). Scores are averaged over a candidate's seeds, then two candidates are compared row by row: `d_i = score_a(i) − score_b(i)`.
- **Uncertainty:** a **paired bootstrap** CI on `mean(d)`, with the rows resampled 2,000 times. The spread across seeds is reported separately, as *stability*.
- **Verdict:** "better" or "worse" when the whole CI lies on one side of zero, otherwise "inconclusive".
- **Validation (exploratory, steers the loop):** each candidate vs the reference; in selection mode also vs the current leader (a candidate whose CI still includes zero is a *contender*, and only contenders may be refined) and a refined candidate vs its parent (the effect of the one knob it changed). The train-validation gap is shown so the agent can tell overfitting from underfitting.
- **Test (confirmatory, read once):** the winner is the best non-reference candidate on validation; in selection mode the runner-up is the best candidate from another family. Both are fixed **before** test is read. The test split then answers winner vs reference and, if there is a runner-up, winner vs runner-up, each at 1 − 0.05/k confidence (k = 1 or 2: 95% or 97.5%, Bonferroni). The report also gives the winner's absolute test score with a CI, its validation-to-test drop (selection optimism made visible) and how many candidates each family received.
- **What it claims:** the winner was selected fairly among the candidates tried, and its held-out performance and comparisons are unbiased by the exploration. It does not claim the best possible model was found.
- **Baseline fairness:** the same split and metric for every candidate; families start at registry defaults with their own default preprocessing (normalized for MLP and linear models; trees are scale-invariant).
- **Sanity floor:** the majority-class rate is shown next to every classification result.
- **Limitation (deliberate):** results condition on one training split. Capturing retraining variability would need k-fold cross-validation, which is out of scope.

## Hard limits

All bounds are code constants, not LLM restraint:

| Limit | Value | Effect |
|---|---|---|
| Rounds | `MAX_ROUNDS` = 3 | the last round always concludes without an LLM call |
| New candidates per round | 3 | one parent, one knob, 1-3 values |
| Runs per candidate | `N_SEEDS` = 3 (random forest, MLP), 1 (linear, tree) | |
| MLP cost per run | epochs ≤ 50, batch size ≥ 16 | no single run can become expensive |
| LLM calls | ≤ `MAX_ROUNDS` + 1 = 4 | plan + ≤ 2 decisions + interpretation |
| Runs per investigation | ≤ 26 (selection), ≤ 33 (effect) | e.g. 8 in round 1 + 2 × 3 × 3 |
| LLM context | ≈ 600–1,000 input tokens per call measured (about 2× on the one validation retry) | stateless calls, one compact row per candidate (so bounded by the candidate count), no rows, no history; output capped at 2,048 tokens, `usage` logged |

## Documentation

| Document | What it covers |
|---|---|
| [`docs/pipeline_flow.tex`](docs/pipeline_flow.tex) | Start here: a one-page mental model, the shared setup, then Selection Mode and Effect Mode each end to end (seeds → runs, paired bootstrap, contenders, refinement, the sealed test set), with real runs on `diabetes_risk.csv` (compile with `pdflatex` or `tectonic`) |
| [`docs/architecture.md`](docs/architecture.md) | Components, investigation modes, model families, the adaptive loop, persistence, API, hard limits |
| [`docs/architecture.html`](docs/architecture.html) | Interactive click-through of the same flows (open in a browser) |
| [`docs/statistics_explained.html`](docs/statistics_explained.html) | The statistics: per-row scores, paired bootstrap, leader and contenders, the sealed test set |

## Example questions for `diabetes_risk.csv`

`diabetes_risk.csv` (in the repository root) has 15,000 patients; 10,500 remain after ingestion drops rows with a missing value. Upload it with target column **`diabetes_risk`** — three classes, `Low` / `Moderate` / `High`, so the task is inferred as **classification** (metric: accuracy; majority-class floor 60%). The other 18 columns are features: `patient_id`, `age`, `gender`, `city`, `bmi`, `family_history_diabetes`, `physical_activity_level`, `diet_type`, `smoking_status`, `alcohol_consumption`, `hours_sleep_per_night`, `stress_level`, `fasting_blood_sugar`, `hba1c_level`, `blood_pressure_systolic`, `blood_pressure_diastolic`, `waist_circumference_cm`, `income_bracket` (categoricals are one-hot encoded).

Type any of these as the research question:

| # | Research question | What the agent does |
|---|---|---|
| 1 | `Which model performs best for predicting diabetes risk on this dataset?` | **Selection** over all four families (logistic regression as reference, decision tree, random forest, MLP); refines contenders, then tests the winner against logistic regression and against the best other family |
| 2 | `Is a random forest more accurate than logistic regression for predicting diabetes risk?` | Two-model comparison (the planner may treat it as a two-family selection or a `model_type` effect); logistic regression is the reference |
| 3 | `Does normalizing the features improve the MLP's accuracy?` | **Effect** of `normalize` (`false` reference vs `true`) on an MLP base |
| 4 | `Does adding dropout help the MLP predict diabetes risk?` | **Effect** of `dropout` (e.g. 0 as reference vs 0.2, 0.5); may refine with more dropout levels |
| 5 | `Does the maximum depth of a decision tree matter for predicting diabetes risk?` | **Effect** of `max_depth` on a `decision_tree` base; may refine with depths either side of the best |
| 6 | `Does a larger minimum leaf size (min_samples_leaf) make the random forest more accurate?` | **Effect** of `min_samples_leaf` on a `random_forest` base |

Questions the current agent **cannot** answer: anything about an individual feature (*"Does BMI matter?"*, feature selection or engineering), models outside the four families (XGBoost, SVM, CNNs …), or regression on this target. The planner answers those with a planning error, shown on the session.

## Tech stack

| Area | Tools |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy, Pydantic |
| Agent orchestration | LangGraph |
| LLM | Groq (`gpt-oss-120b`), strict JSON-schema structured output |
| ML / stats | PyTorch (CUDA when available), scikit-learn, pandas, numpy |
| Database | PostgreSQL (3 tables: datasets, sessions, experiments) |
| Frontend | React, TypeScript, Vite, react-query |
| Testing | pytest, vitest |

## Getting started

**Prerequisites:** Python 3.12+, Node.js, PostgreSQL, and a Groq API key (free at console.groq.com).

```powershell
# 1. Backend setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # for CPU-only PyTorch, change cu128 to cpu in requirements.txt

# 2. Configure: create a PostgreSQL database, then
copy .env.example .env                 # set DATABASE_URL and GROQ_API_KEY

# 3. Create tables and start the API
python -m backend.database.init_db     # add --reset to drop old investigations (datasets are kept)
python -m backend.api.main             # http://localhost:8000/docs

# 4. Start the frontend (new terminal)
cd frontend
npm install
npm run dev                            # http://localhost:5173
```

Optional settings in `.env`: `MAX_ROUNDS` (default 3) and `N_SEEDS` (default 3).

## Tests

```powershell
pytest                     # in-memory SQLite, stubbed LLM - no database or API key needed
cd frontend; npm run test
```

## API

Interactive docs are at `/docs`. All endpoints are under `/api`:

| Endpoint | Purpose |
|---|---|
| `POST /datasets`, `GET /datasets`, `GET/DELETE /datasets/{id}` | Upload, profile and manage CSVs |
| `POST /sessions`, `GET /sessions`, `DELETE /sessions/{id}` | Create, list and delete investigations |
| `POST /sessions/{id}/run` | Start the investigation in the background (202) |
| `GET /sessions/{id}` | Plan, decisions, runs, live validation analysis and the final report |

## Design decisions

| Decision | Why |
|---|---|
| The LLM never computes a statistic | Numbers are reproducible and can't be hallucinated |
| The planner fills a small schema (mode, factor / levels / reference or families), not raw configs | "One change at a time" is enforced by code, not requested in a prompt |
| Every refinement is a parent with one knob changed, and only contenders are refined | Each new result is attributable, and the budget goes where it can change the answer |
| Test comparisons fixed from validation before test is read, Bonferroni for two | Adaptive exploration on validation can't inflate the claims |
| Evaluation rows, not seeds, are the statistical sample | Captures the real uncertainty; a seed-independent baseline needs no special case |
| Unusual-but-valid runs are never excluded | Dropping them would bias the statistics; only crashes and NaN/Inf are invalid |
| Statistics are recomputed, not stored | One source of truth (the experiments); nothing to keep in sync |
| Investigations run as a background task | The API responds immediately; the UI polls one endpoint |

## Project structure

```
backend/
  agents/          # llm.py (Groq client), planner.py, recommender.py
  api/             # FastAPI app, routes (datasets, sessions), schemas, errors
  database/        # ORM models, repository, init_db
  models/          # Pydantic models: dataset, experiment, investigation (plan/decision/report)
  state_machine/   # graph.py - the LangGraph loop
  tools/           # dataset ingestion, trainers, experiment runner, stats
frontend/          # React + TypeScript dashboard
tests/unit/        # pytest suite
docs/              # pipeline_flow.tex, architecture (md + interactive html), statistics explainer
```
