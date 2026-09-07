"""
backend/api/main.py
=====================
uvicorn entrypoint.

    python -m backend.api.main          # uses API_HOST / API_PORT from .env
    uvicorn backend.api.app:app --reload

Kept separate from ``app.py`` so importing the app (in tests) never starts a
server.
"""

from __future__ import annotations

import uvicorn

import backend.config as config


def main() -> None:
    uvicorn.run(
        "backend.api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
