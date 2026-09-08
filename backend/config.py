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
# LLM settings (used from Phase 4 onward)
#
# The Planner and Recommender agents talk to an LLM through the ``LLMClient``
# protocol. Two providers are supported; ``LLM_PROVIDER`` picks one:
#
#   groq   (default) - Groq Cloud, openai/gpt-oss-120b, OpenAI-compatible API
#   gemini           - Google Gemini API (GEMINI_API_KEY + GEMINI_MODEL)
#
# Both use strict / structured JSON output. The Anomaly_Detector stays
# template-based - it never calls an LLM.
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
"""Which LLM backend the agents use: ``groq`` (default) or ``gemini``."""

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
"""Groq Cloud API key. Required when ``LLM_PROVIDER=groq`` - the agents fail fast without it."""

GROQ_BASE_URL: str = os.environ.get(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
)
"""Base URL of Groq's OpenAI-compatible API."""

GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
"""
Groq model id. Default ``openai/gpt-oss-120b`` - supports strict
``response_format`` JSON-schema structured output and has strong reasoning
for the Recommender. Change in .env to switch model without code changes.
"""

# --- Gemini (only used when LLM_PROVIDER=gemini) ---------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
"""Google Gemini API key. Required when ``LLM_PROVIDER=gemini``."""

GEMINI_BASE_URL: str = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
"""Base URL of the Gemini (Generative Language) REST API."""

GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
"""
Gemini model id. Default ``gemini-3.8-flash`` - a current, fast model that
supports ``responseSchema`` structured output. Change in .env to use any
model your key has access to, without code changes.
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
if LLM_PROVIDER == "gemini":
    logger.debug("LLM_PROVIDER=gemini  GEMINI_MODEL: %s (api key %s)", GEMINI_MODEL, "set" if GEMINI_API_KEY else "MISSING")
else:
    logger.debug("LLM_PROVIDER=groq  GROQ_MODEL: %s (api key %s)", GROQ_MODEL, "set" if GROQ_API_KEY else "MISSING")