"""
backend/api/errors.py
=======================
Maps the exceptions the lower layers already raise onto HTTP responses in
the single ``ErrorResponse`` shape - so routes stay thin and never build
error bodies by hand.

    KeyError                         -> 404  (StateManager "... not found")
    CycleError                       -> 409  (session already concluded)
    PlanningError / PlanValidationError
    RecommendationError
    DatasetValidationError
    RequestValidationError           -> 400  (bad input)
    LLMError                         -> 502  (Ollama unreachable / bad output)
    HTTPException                     -> its own status, reshaped
    Exception                        -> 500  (logged with traceback)

Requirements
------------
12.9   Appropriate status codes + detailed messages for invalid requests
12.10  Appropriate status codes + logged details for internal errors
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.agents.llm_client import LLMError
from backend.agents.planner import PlanningError
from backend.agents.recommender import RecommendationError
from backend.api.schemas import ErrorResponse
from backend.models.dataset import DatasetValidationError
from backend.state_machine.executor import CycleError

logger = logging.getLogger(__name__)


def _body(error: str, message: str, details: dict | None = None) -> dict:
    return ErrorResponse(error=error, message=message, details=details).model_dump()


def _key_error_message(exc: KeyError) -> str:
    # KeyError stringifies with wrapping quotes; use the raw arg when present.
    return str(exc.args[0]) if exc.args else "Resource not found."


_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
}


def install_error_handlers(app: FastAPI) -> None:
    """Register every exception -> ErrorResponse handler on ``app``."""

    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content=_body("not_found", _key_error_message(exc)))

    @app.exception_handler(CycleError)
    async def _conflict(_: Request, exc: CycleError) -> JSONResponse:
        return JSONResponse(status_code=409, content=_body("conflict", str(exc)))

    @app.exception_handler(PlanningError)
    async def _planning_failed(_: Request, exc: PlanningError) -> JSONResponse:
        return JSONResponse(status_code=400, content=_body("planning_failed", str(exc)))

    @app.exception_handler(RecommendationError)
    async def _recommendation_failed(_: Request, exc: RecommendationError) -> JSONResponse:
        return JSONResponse(status_code=400, content=_body("recommendation_failed", str(exc)))

    @app.exception_handler(DatasetValidationError)
    async def _invalid_dataset(_: Request, exc: DatasetValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content=_body("invalid_dataset", str(exc)))

    @app.exception_handler(LLMError)
    async def _llm_unavailable(_: Request, exc: LLMError) -> JSONResponse:
        logger.warning("LLM error surfaced to API: %s", exc)
        return JSONResponse(status_code=502, content=_body("llm_unavailable", str(exc)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_body(
                "validation_error",
                "Request body or parameters failed validation.",
                {"errors": exc.errors()},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(
                _STATUS_CODES.get(exc.status_code, "http_error"), str(exc.detail)
            ),
        )

    @app.exception_handler(Exception)
    async def _internal_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error in API request: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_body("internal_error", "An unexpected error occurred."),
        )
