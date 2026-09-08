# Frontend — Adaptive ML Experiment Agent (Phase 7)

React + TypeScript dashboard for the adaptive experimentation loop. It reads
the Phase 6 FastAPI backend and never computes statistics or workflow state of
its own — everything shown is what the backend reports.

## Design

Light, calm, colour-coded. The landing page is a two-step flow — **1. upload a
CSV** (drag & drop / browse → preview → pick target column → ingest), then
**2. start an investigation** (research question + dataset). Colour carries
meaning: blue = primary/info, cyan = the active agent, green = done/success,
orange = warning/anomaly, red = error, violet = LLM-written text. Monospace is
reserved for values the backend computed.

The dataset dropdown in step 2 lists every previously-ingested dataset (from
`GET /api/datasets`), defaulting to the one just uploaded. Old demo datasets
created by the `scripts/checkpoint_*.py` runs show up there too — they are real
database rows, not frontend fixtures.

To clear them: each row in **Your investigations** has a 🗑 button
(`DELETE /api/sessions/{id}` — removes the session and its experiments), and
**Upload dataset → Manage uploaded datasets** lists every dataset with a delete
button (`DELETE /api/datasets/{id}`; if the dataset is still used by
investigations the UI asks whether to delete those too, via `?cascade=true`).

## Run

```powershell
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api -> http://localhost:8000)
```

The backend must be running (`python -m backend.api.main`). Point the proxy
elsewhere with `VITE_API_TARGET`.

## Checks

```powershell
npm run typecheck    # tsc --noEmit
npm run build        # typecheck + vite build
npm run test         # vitest run
```

## How the live "active agent" view works

`AdaptiveLoopVisualizer` shows the six workflow stages (planning → executing →
validating → analyzing → recommending → concluded). The active stage is
`session.current_node` from `GET /api/sessions/{id}`. There is **no** frontend
timer or scripted sequence — the highlight moves only when the backend's
persisted `current_node` moves. `POST /run-cycle` runs the whole investigation
in one request; the dashboard fires it in the background and lets the poll
drive the visuals.

### Run state (idle / running / failed / concluded)

`session.status` only means active-vs-concluded, so `lib/runState.ts` derives a
real 4-state value from the backend's **`run_phase`** column plus the local run
mutation:

| state | when | UI |
|---|---|---|
| `concluded` | `status === "concluded"` | loop complete, polling stops |
| `running` | `run.isPending` **or** `run_phase === "running"` | active stage **pulses**, running banner |
| `failed` | `run_phase === "failed"` or `run.isError` | stalled marker, persisted `run_error`, **Resume** button |
| `idle` | otherwise | current stage highlighted (no pulse), **Run**/**Resume** button |

So a brand-new session shows **Not running** (never "planning stage active"), a
failed run never keeps showing "running", and a page refresh restores the
correct state because `run_phase` is persisted. The session-detail poll runs
fast (1.5 s in `executing`, 2.5 s otherwise) while `run_phase === "running"`,
slow (8 s) otherwise, and stops at `concluded`.

## Divergences from the original Phase 7 tasks

The Phase 7 task list predates the Phase 5/6 "autonomous loop" revision:

- **No per-cycle approval.** One `POST /run-cycle` runs every cycle to a
  conclusion, so there is a single **Run Investigation** action (it also
  resumes a crashed / failed run) rather than an "Approve and Run" button per
  cycle.
- **Live state = `GET /api/sessions/{id}`** (`current_node`, `status`,
  `run_phase`, `run_error`, counts); there is no `/state` endpoint.
- **No per-epoch loss data** in `ExperimentResult.metrics`, so
  `ExperimentVisualizer` plots the final metric against the varied
  hyperparameter instead of loss-vs-epoch curves.
- **No progress percentage / ETA** — the backend provides none, so the UI
  shows the active stage and the real counts only.
- **Upload is client-side text, not multipart.** `POST /api/datasets` takes the
  CSV as a JSON string, so the browser reads the chosen file (FileReader) and
  posts its contents. The preview table + task-type hint are parsed in the
  browser purely for UX; the backend (pandas) stays the source of truth.
