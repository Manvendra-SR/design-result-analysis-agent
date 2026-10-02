"""
backend/api
============
The FastAPI backend.

app.py           create_app() + module-level ``app`` for uvicorn
main.py          ``python -m backend.api.main``
dependencies.py  get_repository, get_investigator
schemas.py       request/response shapes + ErrorResponse
errors.py        exception -> ErrorResponse handlers
routes/          datasets.py, sessions.py
"""
