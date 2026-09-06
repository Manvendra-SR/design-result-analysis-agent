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

> **Schema changes**: this project has no migration tool (no Alembic). If you already had an older
> database initialised before the dataset-first revision, drop and recreate it before re-running
> `init_db` - `schema.sql` only handles fresh creation (`CREATE TABLE IF NOT EXISTS`), not altering
> existing tables to add the new `datasets` table / `sessions.dataset_id` column.

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

Unit tests use **in-memory SQLite** - no live PostgreSQL required:

```powershell
pytest tests/unit/ -v
```

Integration and end-to-end tests (Phase 8) use your local PostgreSQL via
`DATABASE_URL`.

---

## Project structure

```
design-result-analysis-agent/
+-- backend/
|   +-- agents/          # LLM agents (Planner, Recommender) - Phase 4
|   +-- api/             # FastAPI app and routes - Phase 6
|   +-- database/
|   |   +-- schema.sql   # DDL for all 4 tables
|   |   +-- init_db.py   # Schema initialisation script
|   |   +-- connection.py # SQLAlchemy engine + retry
|   |   +-- models.py    # ORM models
|   +-- models/          # Pydantic data models - Phase 2 (incl. DatasetProfile)
|   +-- state_machine/   # LangGraph graph - Phase 5
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
