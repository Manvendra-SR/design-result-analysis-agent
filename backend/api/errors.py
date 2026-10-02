"""
backend/api/errors.py
=======================
Maps exceptions onto HTTP responses in the single ``ErrorResponse`` shape:

    KeyError                                  -> 404  (repository "... not found")
    DatasetValidationError, request validation -> 400
    DatasetInUseError                         -> 409
    HTTPException                             -> its own status
    anything else                             -> 500  (logged with traceback)

Planning / LLM failures never reach here: investigations run in the
background and record their failure on the session instead.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.schemas import ErrorResponse
from backend.models.dataset import DatasetInUseError, DatasetValidationError

logger = logging.getLogger(__name__)

_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 409: "conflict"}


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=code, message=message, details=details).model_dump()
    return JSONResponse(status_code=status, content=body)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return _error(404, "not_found", str(exc.args[0]) if exc.args else "Resource not found.")

    @app.exception_handler(DatasetValidationError)
    async def _invalid_dataset(_: Request, exc: DatasetValidationError) -> JSONResponse:
        return _error(400, "invalid_dataset", str(exc))

    @app.exception_handler(DatasetInUseError)
    async def _dataset_in_use(_: Request, exc: DatasetInUseError) -> JSONResponse:
        return _error(409, "dataset_in_use", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(400, "validation_error", "Request failed validation.", {"errors": exc.errors()})

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, _CODES.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return _error(500, "internal_error", "An unexpected error occurred.")
