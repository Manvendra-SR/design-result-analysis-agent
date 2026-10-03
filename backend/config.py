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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
"""SQLAlchemy URL, e.g. postgresql://user:password@localhost:5432/mlexperiments"""

# ---------------------------------------------------------------------------
# LLM (Groq, OpenAI-compatible API, strict JSON-schema output)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Investigation loop
# ---------------------------------------------------------------------------
MAX_ROUNDS: int = int(os.environ.get("MAX_ROUNDS", "3"))
"""Rounds of experiments per investigation: the initial plan plus up to
MAX_ROUNDS - 1 rounds the Recommender chooses to refine. Together with the
3-candidate cap per round it bounds runs and LLM calls (MAX_ROUNDS + 1)."""

N_SEEDS: int = int(os.environ.get("N_SEEDS", "3"))
"""Training runs per candidate of a seeded family (random forest, MLP). Seeds
average out initialisation noise; they are not the statistical sample
(evaluation rows are - see tools/stats.py). The linear model and the decision
tree are seed-independent and always run once."""

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "*")
API_HOST: str = os.environ.get("API_HOST", "127.0.0.1")
API_PORT: int = int(os.environ.get("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_LOG_DIR = _PROJECT_ROOT / "logs"
if _LOG_DIR.exists():
    _file_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    _file_handler.setLevel(_LOG_LEVEL)
    _file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(_file_handler)
