# Adaptive ML Experiment Agent

An agentic system that answers a research question about your own tabular dataset. Upload a CSV and ask something like *"Does dropout help here?"* or *"Is an MLP actually better than logistic regression on this data?"*. The agent designs a controlled experiment, runs it, decides whether more of it is worth running, and reports the answer with a confidence interval from a held-out test set.

## How it works

```
User: dataset + question
        │
        ▼
1. PLAN (LLM)     -> one factor, the levels to try, the reference level, a fixed base config
        │            (code expands this into runs, so "vary one factor" holds by construction)
        ▼
┌─► 2. RUN        train each new configuration (3 seeds per MLP level, 1 for the baseline)
│       ▼
│   3. CHECK      crashed / NaN runs are excluded and shown; everything else is evidence
│       ▼
│   4. ANALYZE    each level vs the reference on the VALIDATION split (paired bootstrap CI)
│       ▼
│   5. DECIDE     last round -> conclude; otherwise the LLM explores new levels of the
└──── explore        same factor or concludes (code validates every proposal)
        ▼ conclude
6. FINALIZE       unseal the TEST split once: best validation level vs reference
        ▼
Report: deterministic headline + CI, plus an LLM-written interpretation
```

**The split of responsibilities is the main design decision.** LLMs do the parts that need language understanding: turning a free-text question into an experiment design, judging whether another round is worth running, and explaining the result. Deterministic code does everything that has to be correct: training, statistics, validation of the LLM's proposals, the round budget and every number in the report.

## Statistical method

- **What is compared:** each level of the factor against the reference level, on the same evaluation rows.
- **What counts as a sample:** an evaluation row, not a training run. Each run scores every validation and test row (1/0 for classification, squared error for regression). Scores are averaged over a condition's seeds, then compared row by row with the reference: `d_i = score_level(i) − score_reference(i)`.
- **Uncertainty:** a 95% **paired bootstrap** CI on `mean(d)`, with the rows resampled 2,000 times. This reflects how much the difference could move on other data from the same population, which is the dominant source of uncertainty here. The spread across seeds is reported separately, as *stability*.
- **Verdict:** "better" or "worse" when the whole CI lies on one side of zero, otherwise "inconclusive".
- **Train / validation / test:**
  - train fits the models;
  - validation drives every decision inside the loop, and those results are labelled exploratory;
  - test is sealed until the end and used **once**, for a single comparison chosen beforehand (best validation level vs reference). Because of that, exploring adaptively doesn't inflate the final result, and no multiple-comparison correction is needed.
- **Baseline fairness:** both models use the same split, preprocessing, metric and fixed training budget. `linear_baseline` is seed-independent, so it runs once.
- **Sanity floor:** the majority-class rate is shown next to every classification result.
- **Limitation (deliberate):** results condition on one training split. Capturing retraining variability would need k-fold cross-validation, which is out of scope.

## Documentation

| Document | What it covers |
|---|---|
| [`ARCHITECTURE_DEEP_DIVE.md`](ARCHITECTURE_DEEP_DIVE.md) | A guided walk through one investigation, stage by stage, with the real code |
| [`docs/architecture.html`](docs/architecture.html) | Interactive click-through of the same flows (open in a browser); [`docs/architecture.md`](docs/architecture.md) is the text companion |
| [`docs/statistics_explained.html`](docs/statistics_explained.html) | The statistics: per-row scores, paired bootstrap, the sealed test set |
| [`docs/SIMPLIFICATION_PLAN.md`](docs/SIMPLIFICATION_PLAN.md) | The audit of the previous design that led to this one |

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
| The planner fills a small schema (factor / levels / reference), not raw configs | "One factor at a time" is enforced by code, not requested in a prompt |
| Evaluation rows, not seeds, are the statistical sample | Captures the real uncertainty; a seed-independent baseline needs no special case |
| Test split sealed until one final comparison | Adaptive exploration on validation can't bias the answer |
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
docs/              # architecture diagram, statistics explainer, simplification plan
```
