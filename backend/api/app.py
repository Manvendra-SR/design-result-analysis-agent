"""
backend/api/app.py
====================
The FastAPI application factory.

``create_app()`` builds a fresh app (used by tests, which then override
``get_context``); ``app = create_app()`` at module scope is what
``uvicorn backend.api.app:app`` serves. The real ``StateMachineContext`` is
still built lazily on the first request that needs it, so importing this
module never touches the database or Ollama.

Middleware / handlers
---------------------
- CORS (origins from ``config.CORS_ORIGINS``; ``*`` in dev)
- a request-logging middleware (method, path, status, duration)
- the exception -> ``ErrorResponse`` handlers from ``errors.py``
- a lifespan hook that checks database connectivity on startup

Requirements
------------
12.11  CORS headers so the frontend can call the API
Non-Functional Reliability 1/2  startup connectivity check
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import backend.config as config
from backend.api.errors import install_error_handlers
from backend.api.routes import datasets, experiments, sessions
from backend.database.connection import test_connection

logger = logging.getLogger(__name__)


def _cors_origins() -> list[str]:
    raw = config.CORS_ORIGINS.strip()
    if raw == "*" or not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def _lifespan(_: FastAPI):
    reachable = test_connection()
    logger.info(
        "API startup: database reachable=%s, ollama=%s/%s",
        reachable, config.OLLAMA_BASE_URL, config.OLLAMA_MODEL,
    )
    if not reachable:
        logger.warning(
            "Database is not reachable at startup; requests that touch it will fail "
            "until it comes back (DATABASE_URL=%s).",
            "<set>" if config.DATABASE_URL else "<unset>",
        )
    yield
    logger.info("API shutdown")


def create_app() -> FastAPI:
    """Build the FastAPI app (no side effects beyond object construction)."""
    app = FastAPI(
        title="Adaptive ML Experiment Agent",
        version="0.1.0",
        description="Closed-loop adaptive experimentation over a user-uploaded dataset.",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response

    install_error_handlers(app)

    app.include_router(sessions.router)
    app.include_router(experiments.router)
    app.include_router(datasets.router)

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok", "database": test_connection()}

    return app


app = create_app()
