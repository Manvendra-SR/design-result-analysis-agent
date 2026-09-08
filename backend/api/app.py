"""
backend/api/app.py
====================
The FastAPI application factory.

``create_app()`` builds a fresh app (used by tests, which then override
``get_context``); ``app = create_app()`` at module scope is what
``uvicorn backend.api.app:app`` serves. The real ``StateMachineContext`` is
still built lazily on the first request that needs it, so importing this
module never touches the database or Groq.

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
    provider = config.LLM_PROVIDER
    if provider == "gemini":
        model, key_set, key_var, key_hint = (
            config.GEMINI_MODEL, bool(config.GEMINI_API_KEY),
            "GEMINI_API_KEY", "get a key at https://aistudio.google.com/apikey",
        )
    else:
        model, key_set, key_var, key_hint = (
            config.GROQ_MODEL, bool(config.GROQ_API_KEY),
            "GROQ_API_KEY", "free key at https://console.groq.com",
        )
    logger.info(
        "API startup: database reachable=%s, llm provider=%s model=%s",
        reachable, provider, model,
    )
    if not reachable:
        logger.warning(
            "Database is not reachable at startup; requests that touch it will fail "
            "until it comes back (DATABASE_URL=%s).",
            "<set>" if config.DATABASE_URL else "<unset>",
        )
    if not key_set:
        logger.warning(
            "%s is not set - run-cycle requests will fail until you add it to .env (%s).",
            key_var, key_hint,
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
