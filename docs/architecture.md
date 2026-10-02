# Architecture — companion notes

Text companion to [`architecture.html`](architecture.html), the interactive click-through of the same flows (open it in a browser, pick a scenario, press Space). Everything here matches the code on the current branch; file paths are given so each step can be checked.

**One-sentence summary:** an LLM designs a one-factor experiment, code trains and evaluates it, the LLM decides whether another round is worth running, and the answer comes from a single comparison on a held-out test set — LLMs handle language, code handles every number.

## Components on the canvas

| Node | Role | Where | What it does |
|---|---|---|---|
| React UI | client | `frontend/` | Upload, ask, start; polls `GET /api/sessions/{id}` every 2 s while running |
| FastAPI | orchestration | `backend/api/` | 9 endpoints; starts investigations as background tasks |
| CSV store | persistence | `data/uploads/` | The uploaded CSV; read with a fixed 70/15/15 split |
| PostgreSQL | persistence | `backend/database/` | 3 tables (datasets, sessions, experiments) behind `Repository` |
| Investigation loop | orchestration | `backend/state_machine/graph.py` | LangGraph: plan_experiments → run_experiments → analyze → decide → finalize |
| Planner agent | LLM | `backend/agents/planner.py` | Question → factor, levels, reference, base config |
| Recommender agent | LLM | `backend/agents/recommender.py` | explore / conclude after each round; final interpretation |
| Experiment runner | deterministic | `backend/tools/trainers.py`, `experiment_runner.py` | Trains MLP (CUDA or CPU) / linear baseline; crash or NaN → `failed` |
| Stats | deterministic | `backend/tools/stats.py` | Paired bootstrap CI of each level vs the reference |
| Groq Cloud | LLM provider | `backend/agents/llm.py` | `openai/gpt-oss-120b`, strict JSON-schema output |

## Flow 1 — Setup (dataset, then investigation)

1. **UI → API** `POST /api/datasets` with the CSV as text and the target column.
2. **API → CSV store** `ingest_csv` validates, infers the task type, one-hot encodes categoricals, fixes `split_seed`, copies the file to `data/uploads/<id>/`.
3. **API → PostgreSQL** the `DatasetProfile` goes into `datasets`.
4. **UI → API** `POST /api/sessions` with the question and dataset id.
5. **API → PostgreSQL** a `sessions` row: status `pending`, no plan, decisions or report.

## Flow 2 — Start, plan, run round 1

1. **UI → API** `POST /api/sessions/{id}/run` → **202**. 409 if running or done; a failed session is reset first.
2. **API → PostgreSQL** status `running`, set *before* responding.
3. **API → loop** `run_investigation` runs as a background task; graph state lives in memory.
4. **Loop → Planner** `plan(question, profile)`.
5. **Planner → Groq** strict-schema call (temperature 0.2).
6. **Groq → Planner** `{factor, levels, reference, base, rationale}`. Levels arrive as strings and are parsed per factor; a broken rule is fed back once.
7. **Planner → loop** code expands the plan with `configs_for`: base config with only the factor changed; seeds 0..N-1 per MLP level, one run for `linear_baseline`.
8. **Loop → PostgreSQL** `save_plan`.
9. **Loop → runner** one `run_experiment` per configuration (GPU/CPU mode differs only here).
10. **Runner → CSV store** `Dataset.load_split` — same rows for every run.
11. **Runner → loop** `ExperimentResult` with `status` ok/failed, metrics, and per-row `val_scores` / `test_scores`.
12. **Loop → PostgreSQL** each result stored as it finishes.

## Flow 3 — Analyze, decide, explore

1. **Loop → Stats** `analyze(plan, results, profile, "val")`.
2. **Stats → loop** per-level mean and seed spread; each level vs reference with diff, 95% CI, verdict.
3. **Loop → Recommender** `decide(...)` — *not called on the last round*; the budget concludes.
4. **Recommender → Groq** `{action, new_levels, rationale}`.
5. **Groq → Recommender** code checks: same factor, in range, not already tried.
6. **Recommender → loop** conditional edge: `explore` → back to `run_experiments` with the new levels; `conclude` → `finalize`.
7. **Loop → PostgreSQL** `append_decision`, and the plan with new levels.
8. **Loop → runner** round 2 trains only the new levels; earlier runs remain evidence.

## Flow 4 — Finalize

1. **Loop → Stats** the best non-reference level on validation vs the reference, on the **test** split — the only time test scores are read.
2. **Stats → loop** the comparison plus a deterministic headline sentence.
3. **Loop → Recommender** `interpret(...)`.
4. **Recommender → Groq** a few sentences citing the numbers. If this fails, the report is saved without it.
5. **Loop → PostgreSQL** `save_report`, status `done`.

## Flow 5 — UI reads state

1. **UI → API** `GET /api/sessions/{id}` (every 2 s while running).
2. **API → PostgreSQL** session row + experiments.
3. **API → Stats** the validation analysis is recomputed on every request — statistics are never stored.
4. **API → UI** `SessionDetail` (plan, decisions, runs, analysis, report). Per-row scores are excluded.

## Flow 6 — Failure

1. A node raises: `LLMError` (Groq unreachable after transport retries), `PlanningError` (no factor can answer the question), or nothing successful to compare.
2. **Loop → PostgreSQL** status `failed` with the error text; nothing is raised from the background task.
3. **UI → API** the user presses *Start over* → `POST /run`.
4. **API → PostgreSQL** `reset_session` discards the old runs, plan, decisions and report, then the run starts fresh. At startup the API marks any session left `running` by a restart as `failed`.

## Mode differences (GPU vs CPU)

| | GPU | CPU |
|---|---|---|
| `trainers.DEVICE` | `cuda` | `cpu` |
| Runner node | PyTorch · CUDA / sklearn | PyTorch · CPU / sklearn |
| Batching | training set on the GPU, batches by index, one sync per epoch | same code path |
| Batch order / seeds | identical (CPU-side shuffle generator) | identical |
| Speed, hidden 512 / batch 256, 20 epochs | ~3.3 s | ~9.3 s |
| Speed, default hidden 64 / batch 32 | ~24 s (launch-bound) | ~24 s |

Everything else — the loop, the statistics, the LLM calls — is identical in both modes.

## What is stored, and what is not

| Stored | Recomputed | In memory only |
|---|---|---|
| dataset profile, session status/error, plan, decisions, report, every run with its per-row scores | validation analysis (every request), round count | LangGraph state during one run |

The rule: store only what is non-recomputable and shown to the user.

## Using the diagram in a walkthrough

- **Flow 2** carries the main design argument: the planner fills a form, and code makes "one factor at a time" a guarantee.
- **Flow 3** shows where the agent's judgement actually matters (explore vs conclude) and where it is deliberately constrained.
- **Flow 4** is the statistical punchline: adaptive exploration on validation, one confirmatory comparison on test.
- Toggle **GPU/CPU** on Flow 2, step 9, to show the only part of the system the device affects.
