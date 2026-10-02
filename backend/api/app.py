"""
backend/api/app.py
====================
The FastAPI application. ``uvicorn backend.api.app:app`` serves it; tests call
``create_app()`` and override the dependencies in ``dependencies.py``.
Importing this module touches neither the database nor the LLM.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backend.config as config
from backend.api.dependencies import get_repository
from backend.api.errors import install_error_handlers
from backend.api.routes import datasets, sessions
from backend.database.connection import database_reachable

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if not config.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not set - investigations will fail until it is added to .env.")
    try:
        repo = app.dependency_overrides.get(get_repository, get_repository)()
        interrupted = repo.fail_interrupted_runs()
        if interrupted:
            logger.warning("Marked %d investigation(s) interrupted by the last shutdown as failed.", interrupted)
    except Exception as exc:  # noqa: BLE001 - the API can still start without a database
        logger.warning("Database not reachable at startup: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Adaptive ML Experiment Agent",
        description="Plans, runs and statistically evaluates experiments that answer a question about a dataset.",
        lifespan=_lifespan,
    )
    origins = [o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    install_error_handlers(app)
    app.include_router(datasets.router)
    app.include_router(sessions.router)

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok", "database": database_reachable()}

    return app


app = create_app()
