# backend/api/routes/__init__.py
"""
backend/api/routes
====================
The endpoint routers (Phase 6). All are mounted under ``/api`` by
``backend/api/app.py``.

sessions.py     - POST /api/sessions, GET /api/sessions,
                  GET /api/sessions/{id}, POST /api/sessions/{id}/run-cycle,
                  GET /api/sessions/{id}/recommendation
experiments.py  - GET /api/sessions/{id}/experiments, GET /api/experiments/{id}
datasets.py     - POST /api/datasets, GET /api/datasets, GET /api/datasets/{id}
"""
