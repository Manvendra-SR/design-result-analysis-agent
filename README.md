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

**LLM stack**: [Ollama](https://ollama.com/) + [Qwen](https://ollama.com/library/qwen2.5)
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
| Ollama | LLM inference server (needed from Phase 4 onward) |

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

# Ollama - only needed from Phase 4 onward
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
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
> `sessions.pending_configs` / `sessions.latest_analysis` columns, etc. - drop and recreate the
> database, then re-run `init_db`.

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

Unit tests use **in-memory SQLite** - no live PostgreSQL and no Ollama server
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
| POST | `/sessions` | Create a session (`{research_question, dataset_id}`) |
| GET | `/sessions`, `/sessions/{id}` | List / fetch sessions (detail includes `plan_explanation`) |
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

1. Install Ollama: https://ollama.com/download
2. Pull the Qwen model:
   ```powershell
   ollama pull qwen2.5:7b
   ```
3. Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env`.

The LLM adapter (`backend/agents/llm_client.py`) is kept separate from agent
logic - changing the model is a `.env` change, not a code change.

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| PostgreSQL over SQLite | Production-representative; robust transactions |
| LangGraph state = session_id + current_node only | PostgreSQL is single source of truth; no state duplication |
| Ollama + Qwen | Local inference, no API key required, model-agnostic adapter |
| scipy for all statistics | Deterministic, reproducible, never LLM-generated |
| Template-based anomaly explanations | Faster and cheaper than per-anomaly LLM calls |
| 4 DB tables (no stats/recommendations tables) | Stats computed on-demand; latest recommendation stored as JSON blob |
| Dataset-first architecture | Datasets are ingested/profiled once, independent of any experiment; MNIST and synthetic regression were removed rather than kept as special cases |
| Fixed per-dataset train/val/test split | Experiments differing only in `random_seed` compare model/training randomness, not different rows landing in validation |
