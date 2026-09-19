# Adaptive ML Experiment Agent

An agentic AI system that investigates a research question about your own tabular dataset. You upload a CSV and ask a question (e.g. "Does a neural network beat a linear baseline here?"). The system then plans experiments, runs them, checks for anomalies, runs statistical tests, and decides what to try next. It repeats this loop on its own until it has enough evidence to answer.

## How it works

```
React Frontend  ->  FastAPI Backend  ->  LangGraph State Machine
                                              |
                       +----------------------+---------------------+
                       |  LLM agents                                |  Deterministic tools
                       |  - Planner (designs experiments)           |  - Trainers (MLP, linear baseline)
                       |  - Recommender (continue or conclude?)     |  - Statistical analyzer (scipy)
                       +--------------------------------------------+  - Anomaly detector
                                              |                         - State manager
                                         PostgreSQL
```

The adaptive loop:

**plan -> run experiments -> validate/detect anomalies -> analyze statistics -> recommend -> (repeat or conclude)**

A key design choice: **the LLM decides *what* to test, but never computes numbers.** All training, statistics and anomaly detection are plain deterministic code, so results are reproducible and can't be hallucinated.

## Key features

- **Closed-loop agent**: one API call runs the whole investigation, with a configurable cycle cap.
- **Bring your own dataset**: CSVs are profiled automatically (task type, columns, fixed train/val/test split).
- **Sound statistics**: replicates share the same seeds across conditions, t-tests handle deterministic baselines, and skipped comparisons are recorded with a reason.
- **Anomaly handling**: flags are re-checked every cycle and can be withdrawn, so a bad flag can't block a conclusion.
- **Honest outcomes**: a run stopped by the cycle cap is shown as "stopped at limit", not as a conclusion.
- **Reliable LLM output**: strict JSON-schema output plus validate-and-retry. Supports Groq (default) and Google Gemini via one setting.
- **Crash recovery**: progress is saved after every step, so an interrupted run resumes where it stopped.
- **Live dashboard**: shows which stage the agent is in, per-cycle history, statistics and the final recommendation.

## Statistical methods

All computed with scipy/numpy in `backend/tools/`:

- **Independent two-sample t-test** (`scipy.stats.ttest_ind`) to compare two experiment conditions, reporting the t-statistic and p-value.
- **One-sample t-test** (`ttest_1samp`) when one condition is deterministic (e.g. linear regression gives identical results for every seed), treating it as a known constant. If both conditions are deterministic, no test is run and the pair is recorded as skipped with a reason.
- **Cohen's d** (using pooled standard deviation) as the effect size, so results show practical as well as statistical significance.
- **95% confidence intervals** for the difference in means, from the t-distribution.
- **"Underpowered" warning** when a condition has fewer than 5 successful replicates.
- **Leave-one-out z-score outlier detection** on validation loss: each run is compared to the mean and std of the *other* replicates in its condition, and flagged only if it is more than 3 std away, more than 10% away from the mean, and the condition has at least 5 replicates.
- **One-sided z-test against random-chance accuracy** (`1 / n_classes`) to flag a "validation collapse", a classifier not significantly better than guessing (z < 1.645).
- **Training loss divergence check**: flags a run whose final training loss ended above its initial loss.
- **Matched random seeds** (1, 2, 3, ...) across conditions with at least 3 replicates each, so the only difference between conditions is the factor being varied.

## Tech stack

| Area | Tools |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy, Pydantic |
| Agent orchestration | LangGraph |
| LLMs | Groq (`gpt-oss-120b`) or Gemini, with structured output |
| ML / stats | PyTorch, scikit-learn, scipy, pandas, numpy |
| Database | PostgreSQL |
| Frontend | React, TypeScript, Vite, react-query |
| Testing | pytest, vitest |

## Getting started

**Prerequisites:** Python 3.13, Node.js, PostgreSQL, and a Groq API key (free at console.groq.com).

```powershell
# 1. Backend setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # for CPU-only PyTorch, change cu128 to cpu in requirements.txt

# 2. Configure: create a PostgreSQL database, then
copy .env.example .env                 # set DATABASE_URL and GROQ_API_KEY

# 3. Create tables and start the API
python -m backend.database.init_db
python -m backend.api.main             # http://localhost:8000/docs

# 4. Start the frontend (new terminal)
cd frontend
npm install
npm run dev                            # http://localhost:5173
```

To use Gemini instead, set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`.

> There is no migration tool by design. After a schema change, drop the tables and re-run `init_db`.

## Tests

```powershell
pytest tests/unit/ -v      # in-memory SQLite, no database or API key needed
cd frontend; npm run test
```

## API overview

Interactive docs are at `/docs`. Main endpoints (all under `/api`):

| Endpoint | Purpose |
|---|---|
| `POST /datasets` | Upload and profile a CSV |
| `POST /sessions` | Start an investigation (question + dataset) |
| `POST /sessions/{id}/run-cycle` | Run the full adaptive loop until it concludes |
| `GET /sessions/{id}/cycles` | Per-cycle history (experiments, stats, recommendation) |
| `GET /sessions/{id}/recommendation` | Final recommendation |

## Design decisions

| Decision | Why |
|---|---|
| Statistics are computed by scipy, never by the LLM | Reproducible, no hallucinated numbers |
| PostgreSQL is the single source of truth (graph state is just session id + current node) | No duplicated state; easy crash recovery |
| Fixed data split per dataset | Seed-only differences reflect model randomness, not different validation rows |
| Same seeds used in every compared condition | Makes "vary one factor" actually true |
| MLP reports its best-epoch validation metrics | Avoids measuring the epoch budget instead of the configuration |
| Termination reason stored separately from the recommendation | A capped run is never presented as a real conclusion |
| Follow-up questions create new sessions | Evidence from one question never leaks into another |

## Project structure

```
backend/
  agents/          # Planner and Recommender (LLM)
  api/             # FastAPI app, routes, schemas
  database/        # schema.sql, ORM models, init script
  models/          # Pydantic data models
  state_machine/   # LangGraph loop (nodes, graph, executor)
  tools/           # Trainers, statistics, anomaly detection, dataset ingestion
frontend/          # React + TypeScript dashboard
tests/unit/        # Unit tests (in-memory SQLite, stubbed LLM and trainers)
```
