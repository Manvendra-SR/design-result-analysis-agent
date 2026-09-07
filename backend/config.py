"""
backend/config.py
=================
Loads environment variables from .env and exposes them as typed settings.
Configures the project-wide Python logging setup.

All components import from this module - never read os.environ directly.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (one level above this file's directory)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_FILE)

# ---------------------------------------------------------------------------
# Database settings
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
"""
SQLAlchemy-compatible database URL.
Example: postgresql://mluser:password@localhost:5432/mlexperiments
"""

# ---------------------------------------------------------------------------
# LLM settings (Ollama + Qwen - used from Phase 4 onward)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
"""Base URL of the Ollama HTTP API."""

OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
"""
Ollama model tag to use for LLM inference.
Change in .env to switch model without code changes.
"""

# ---------------------------------------------------------------------------
# Adaptive loop settings (Phase 5)
# ---------------------------------------------------------------------------
MAX_ADAPTIVE_CYCLES: int = int(os.environ.get("MAX_ADAPTIVE_CYCLES", "6"))
"""
Hard cap on the number of adaptive cycles a single ``execute_cycle`` /
``POST /run-cycle`` invocation will run. The Recommender_Agent normally
decides when to stop; this is the safety limit that prevents an infinite
in-graph loop if it never concludes. Requirement 11 expects 2-3 cycles, so
6 is generous headroom.
"""

# ---------------------------------------------------------------------------
# API settings (FastAPI backend - Phase 6)
# ---------------------------------------------------------------------------
CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "*")
"""
Allowed CORS origins for the API. ``*`` (the default) allows any origin,
which is fine for local development; set a comma-separated list of origins
(e.g. ``http://localhost:5173``) for anything else.
"""

API_HOST: str = os.environ.get("API_HOST", "127.0.0.1")
"""Host interface the uvicorn entrypoint binds to."""

API_PORT: int = int(os.environ.get("API_PORT", "8000"))
"""Port the uvicorn entrypoint binds to."""

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
_LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Optional file handler - writes to logs/app.log if the directory exists
_LOG_DIR = _PROJECT_ROOT / "logs"
if _LOG_DIR.exists():
    _file_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    _file_handler.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    _file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger(__name__)
logger.debug("Config loaded from %s", _ENV_FILE)
if DATABASE_URL:
    logger.debug("DATABASE_URL host portion: %s", DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else "<configured>")
else:
    logger.debug("DATABASE_URL is not currently set in environment")
logger.debug("OLLAMA_BASE_URL: %s", OLLAMA_BASE_URL)
logger.debug("OLLAMA_MODEL: %s", OLLAMA_MODEL)