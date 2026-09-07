# backend/api/__init__.py
"""
backend/api
============
The FastAPI backend (Phase 6).

app.py          - create_app() factory + module-level ``app`` for uvicorn
main.py         - ``python -m backend.api.main`` entrypoint
dependencies.py - get_context(): the lazily-built StateMachineContext singleton
schemas.py      - request/response models + the shared ErrorResponse
errors.py       - exception -> ErrorResponse handlers
routes/         - sessions, experiments, datasets routers

Endpoints (11 = the design's 7 + 3 dataset endpoints the dataset-first
architecture needs + 1 cycle-history endpoint the autonomous loop needs):

    POST   /api/datasets                          ingest a CSV
    GET    /api/datasets
    GET    /api/datasets/{dataset_id}
    POST   /api/sessions                          create a session (needs dataset_id)
    GET    /api/sessions
    GET    /api/sessions/{session_id}
    POST   /api/sessions/{session_id}/run-cycle   run the whole adaptive investigation
    GET    /api/sessions/{session_id}/cycles      per-cycle investigation history
    GET    /api/sessions/{session_id}/experiments
    GET    /api/experiments/{experiment_id}
    GET    /api/sessions/{session_id}/recommendation
"""
