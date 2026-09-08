# Adaptive ML Experiment Agent

An intelligent experimentation system that demonstrates genuine agentic AI
behaviour through a **closed-loop adaptive workflow**. A researcher uploads
their own tabular dataset and submits a research question about it; the
system profiles the dataset, plans experiments, runs them, detects anomalies,
performs statistical analysis, and recommends the next experiment - cycling
until sufficient evidence is gathered to answer the question. The dataset is
a first-class input: the system is not built around a predefined dataset.

---

## Architecture overview

```
React Frontend  (Phase 7)
      |  REST API
FastAPI Backend  (Phase 6)
      |
LangGraph State Machine  (Phase 5)
      |
+-------------+----------------------+
|  LLM Agents |  Deterministic Tools |
|  - Planner  |  - Experiment Runner |
|  - Recommender  | - Statistical Analyzer |
|             |  - Anomaly Detector  |
|             |  - State Manager     |
+-------------+----------------------+
      |  SQLAlchemy
Local PostgreSQL  <->  pgAdmin
```

**LLM stack**: pluggable via `LLM_PROVIDER` — [Groq](https://groq.com/) Cloud `openai/gpt-oss-120b` (default) or [Google Gemini](https://ai.google.dev/) `gemini-3.8-flash`, both with structured JSON-schema output
**Statistical stack**: scipy, pandas, numpy, scikit-learn
**DB**: PostgreSQL (4 tables: datasets, sessions, experiments, anomalies)
**Models**: `mlp` (generic feed-forward network) and `linear_baseline` (logistic/linear regression), chosen automatically by the uploaded dataset's inferred task type

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python 3.13 | Application runtime |
| PostgreSQL (local installation) | Database server |
| pgAdmin 4 | Database inspection / management |
| Groq API key | LLM inference (needed from Phase 4 onward) — free at console.groq.com |

---

## Setup - Step by step

### 1. Navigate to project and create virtual environment

```powershell
cd design-result-analysis-agent
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

> **PyTorch note**: The `requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cu128`
> which installs the CUDA 12.8 build of PyTorch. If you need a CPU-only build,
> replace `cu128` with `cpu` in `requirements.txt` before running the above command.

### 3. Create the database in pgAdmin

1. Open **pgAdmin 4** and connect to your local PostgreSQL server.
2. Right-click **Databases** -> **Create** -> **Database**.
3. Set **Database** name: `mlexperiments` (or any name you prefer).
4. Optionally create a dedicated login role:
   - Right-click **Login/Group Roles** -> **Create** -> **Login/Group Role**.
   - Set name `mluser`, password `yourpassword`, enable *Can login*.
5. Grant the role ownership of the new database if needed.

### 4. Configure environment variables

```powershell
copy .env.example .env
```

Edit `.env` with your actual values:

```env
# PostgreSQL - adjust user, password, host, port, and db name to match your setup
DATABASE_URL=postgresql://mluser:yourpassword@localhost:5432/mlexperiments

# Groq - needed from Phase 4 onward (free key at https://console.groq.com)
GROQ_API_KEY=gsk_...
```

### 5. Initialise the database schema

```powershell
python -m backend.database.init_db
```

Expected output:
```
INFO  Connecting to database...
INFO  Running schema.sql...
INFO  Database initialised successfully. Tables: datasets, sessions, experiments, anomalies
```

> **Schema changes**: this project has no migration tool (no Alembic), by choice during
> development. `schema.sql` only handles fresh creation (`CREATE TABLE IF NOT EXISTS`), so after
> any change to it - the `datasets` table, `sessions.dataset_id`, the Phase 5
> `sessions.pending_configs` / `sessions.latest_analysis` columns, the
> `sessions.run_phase` / `sessions.run_error` columns (run-state tracking) - drop and recreate the
> database, then re-run `init_db`:
> ```powershell
> # in pgAdmin or psql:  DROP TABLE anomalies, experiments, sessions, datasets CASCADE;
> python -m backend.database.init_db
> ```

### 6. Verify in pgAdmin

Navigate to: **mlexperiments -> Schemas -> public -> Tables**

You should see four tables:
- `datasets` - uploaded dataset metadata/profile (task type, columns, split seed)
- `sessions` - research sessions, each scoped to one dataset
- `experiments` - individual experiment runs
- `anomalies` - detected anomalies linked to experiments

Right-click each table -> **Properties** to inspect columns, indexes, and
foreign-key constraints.

---

## Running tests

Unit tests use **in-memory SQLite** - no live PostgreSQL and no Groq key
required (the LLM agents and the training runner are stubbed):

```powershell
pytest tests/unit/ -v
```

Integration and end-to-end tests (Phase 8) use your local PostgreSQL via
`DATABASE_URL`.

Per-phase checkpoint scripts do a real end-to-end run and skip gracefully
when a live dependency is missing:

```powershell
$env:PYTHONPATH="."; python scripts/checkpoint_5_11.py   # state machine (autonomous loop)
$env:PYTHONPATH="."; python scripts/checkpoint_6_10.py   # API (all 11 endpoints)
```

---

## Running the API

```powershell
python -m backend.api.main          # binds API_HOST:API_PORT (default 127.0.0.1:8000)
# or: uvicorn backend.api.app:app --reload
```

Interactive docs at `http://localhost:8000/docs`. Endpoints (all under `/api`):

| Method | Path | Purpose |
|---|---|---|
| POST | `/datasets` | Ingest a CSV (`{filename, csv_content, target_column, task_type_override?}`) -> `DatasetProfile` |
| GET | `/datasets`, `/datasets/{id}` | List / fetch dataset profiles |
| DELETE | `/datasets/{id}` | Delete a dataset (DB row + `data/uploads/<id>/`). 409 if any investigation still references it, unless `?cascade=true` (which also deletes those investigations). |
| POST | `/sessions` | Create a session (`{research_question, dataset_id}`) |
| GET | `/sessions`, `/sessions/{id}` | List / fetch sessions (detail includes `plan_explanation`) |
| DELETE | `/sessions/{id}` | Delete an investigation and its experiments + anomalies (cascade). The dataset is left untouched. |
| POST | `/sessions/{id}/run-cycle` | **Run the whole adaptive investigation.** The graph loops plan -> execute -> validate -> analyze -> recommend -> execute -> ... autonomously until the Recommender concludes or the `MAX_ADAPTIVE_CYCLES` cap. Returns when `status == "concluded"`. If the session crashed mid-run, resumes it and continues. |
| GET | `/sessions/{id}/cycles` | Per-cycle investigation history: for each adaptive cycle, its experiments, anomalies, statistical comparisons, and the recommendation (with the agent's reasoning for continuing / stopping). This is what a UI renders so the autonomous loop is not a black box. |
| GET | `/sessions/{id}/experiments?status=` | Experiments for a session (each tagged with the `cycle` that produced it) |
| GET | `/experiments/{id}` | One experiment |
| GET | `/sessions/{id}/recommendation` | The **final** recommendation (404 before the first run) |

Every error response has the same shape: `{"error", "message", "details"}`.

### The adaptive loop

`POST /run-cycle` is autonomous: one call runs the entire closed-loop
investigation. The LangGraph graph has a real `recommending -> executing`
conditional edge, so cycles repeat inside a single `graph.invoke()` until
the Recommender returns `action == "conclude"`. `MAX_ADAPTIVE_CYCLES`
(env, default 6) is the safety cap - if the Recommender never concludes, the
`recommending` node forces a conclusion and records why in the
recommendation's `evidence_summary`. There is no per-cycle human approval
gate; the human inspects the completed investigation via `GET /cycles`.

Crash recovery is a separate mechanism: every node persists
`sessions.current_node` before returning, and a router on the graph's
`START` resumes an interrupted session at that node (then it keeps looping
to conclusion).

---

## Running the frontend (Phase 7)

React + TypeScript dashboard (Vite) — a light, colour-coded UI. It only reads
the API and computes no statistics or workflow state of its own. The landing
page is a two-step flow: **upload a CSV** (drag & drop / browse → preview →
choose the target column → ingest) then **start an investigation** (research
question + dataset). CSV upload is read in the browser and posted as text to
`POST /api/datasets` (no multipart, no backend change).

```powershell
cd frontend
npm install
npm run dev          # http://localhost:5173  (dev proxy /api -> http://localhost:8000)
```

The backend (`python -m backend.api.main`) must be running. Point the proxy
elsewhere with `VITE_API_TARGET`.

Checks: `npm run typecheck`, `npm run build`, `npm run test` (vitest + React
Testing Library).

**Live active-agent view.** `AdaptiveLoopVisualizer` highlights the workflow
stage named by `sessions.current_node`, polled from `GET /api/sessions/{id}`.
The highlight (a cyan glow) *pulses* only while a run is genuinely in
progress — the UI derives a real 4-state value (`idle` / `running` / `failed`
/ `concluded`) from the backend's `run_phase` column plus the local run
mutation (`frontend/src/lib/runState.ts`), so:
- a brand-new session shows **Not running**, not "planning stage active";
- a failed run shows a **stalled/failed** state with the persisted error and a
  Resume button — it never keeps showing "running";
- a page refresh restores the correct state from `run_phase`;
- on `concluded` the whole loop shows complete and polling stops.
The highlight still moves only when the backend's persisted node moves — no
frontend timer. `POST /run-cycle` runs the whole investigation in one request;
the dashboard fires it in the background and lets the poll drive the visuals.
See `frontend/README.md` for the divergences from the original Phase 7 task
list.

---

## Project structure

```
design-result-analysis-agent/
+-- backend/
|   +-- agents/          # LLM agents (Planner, Recommender) - Phase 4
|   +-- api/             # FastAPI app, routes, schemas, error handlers - Phase 6
|   |   +-- app.py       # create_app() + module-level `app` for uvicorn
|   |   +-- routes/      # sessions, experiments, datasets routers
|   +-- database/
|   |   +-- schema.sql   # DDL for all 4 tables
|   |   +-- init_db.py   # Schema initialisation script
|   |   +-- connection.py # SQLAlchemy engine + retry
|   |   +-- models.py    # ORM models
|   +-- models/          # Pydantic data models - Phase 2 (incl. DatasetProfile)
|   +-- state_machine/   # LangGraph adaptive loop - Phase 5
|   |   +-- nodes.py     # planning/executing/validating/analyzing/recommending
|   |   +-- graph.py     # compiled StateGraph (resumable from any node)
|   |   +-- executor.py  # execute_cycle(session_id, context) -> CycleResult
|   +-- tools/
|   |   +-- dataset/     # CSV ingestion, preprocessing, splitting (Dataset facade)
|   |   +-- ...          # ExperimentRunner, trainers, StatisticalAnalyzer, AnomalyDetector, StateManager
|   +-- config.py        # Environment variable loading + logging
+-- frontend/            # React + TypeScript dashboard (Vite) - Phase 7
|   +-- src/
|   |   +-- components/  # SessionView + AdaptiveLoopVisualizer, ExperimentTable, etc.
|   |   +-- hooks/       # react-query hooks (polling lives here)
|   |   +-- services/    # axios API client
|   |   +-- types/       # TS mirrors of the Pydantic models
+-- tests/
|   +-- unit/            # Fast in-memory SQLite / no-live-dependency tests
|   +-- integration/     # PostgreSQL integration tests - Phase 8
+-- data/uploads/        # Ingested dataset files (gitignored)
+-- .env.example         # Environment template
+-- requirements.txt     # Python dependencies
+-- README.md
```

---

## LLM setup (Phase 4 onward)

Only the **Planner** and **Recommender** agents call an LLM; anomaly
explanations are template-based.

The backend is chosen by **`LLM_PROVIDER`** (`groq` — default — or `gemini`);
both go through the same `LLMClient` interface and the same output schemas, so
switching is a `.env` change with no code edit. Only the selected provider's
key is needed; no key → the agents fail fast with a clear message (HTTP 502).

**Groq (default):**

1. Get a free key at https://console.groq.com
2. `.env`: `GROQ_API_KEY=...` (that's the only required line — `GROQ_MODEL`
   defaults to `openai/gpt-oss-120b`, `GROQ_BASE_URL` to the standard endpoint).

**Gemini:**

1. Get a key at https://aistudio.google.com/apikey
2. `.env`: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`. `GEMINI_MODEL`
   defaults to `gemini-3.8-flash` and accepts any model your key can use; the
   strict schemas are adapted to Gemini's `responseSchema` automatically.

`gpt-oss-120b` is used with **strict `json_schema` structured output** — the
Planner / Recommender output schemas (`backend/agents/output_schemas.py`)
constrain generation so `action` cannot be null, required fields cannot be
missing, and types are enforced. This is what kills the "unparseable
recommendation" failure at the source. The provider clients (`GroqClient` /
`GeminiClient`) and these schemas are the only LLM code — swapping models is a
`GROQ_MODEL` / `GEMINI_MODEL` change, swapping providers an `LLM_PROVIDER` one.

> **Free-tier rate limit.** Groq's free tier caps `gpt-oss-120b` at ~8000
> tokens/minute, which a multi-cycle investigation exceeds. `GroqClient`
> honours the `Retry-After` header and rides it out (a 6-cycle run then takes
> a few minutes of mostly waiting). For a fast run, upgrade to the **Dev tier**
> (free — just add a payment method at console.groq.com/settings/billing), or
> set `GROQ_MODEL=llama-3.3-70b-versatile` (higher free-tier limit).

### Backstop reliability

Beyond the schema, both agents run their LLM call through
`agents/_common.request_structured`: the reply is parsed **and validated
against the Pydantic model inside a bounded retry loop** (3 attempts), so
anything a schema can't express — a hyperparameter out of range,
`run_more_experiments` with an empty list — is fed back to the model to
correct before the agent gives up. Validation is never weakened. The Planner
and Recommender additionally **repair the experiment set deterministically**
so every condition carries at least 3 distinct random seeds (see below),
rather than rejecting a plan the model got slightly wrong.

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| PostgreSQL over SQLite | Production-representative; robust transactions |
| LangGraph state = session_id + current_node only | PostgreSQL is single source of truth; no state duplication |
| Groq + gpt-oss-120b default, Gemini optional | Reliable structured output; one HTTP adapter per provider behind `LLMClient`, provider set via `LLM_PROVIDER`, model via `GROQ_MODEL` / `GEMINI_MODEL` |
| scipy for all statistics | Deterministic, reproducible, never LLM-generated |
| Template-based anomaly explanations | Faster and cheaper than per-anomaly LLM calls |
| 4 DB tables (no stats/recommendations tables) | Stats computed on-demand; latest recommendation stored as JSON blob |
| Dataset-first architecture | Datasets are ingested/profiled once, independent of any experiment; MNIST and synthetic regression were removed rather than kept as special cases |
| Fixed per-dataset train/val/test split | Experiments differing only in `random_seed` compare model/training randomness, not different rows landing in validation |
