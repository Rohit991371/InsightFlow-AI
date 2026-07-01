"""
Central configuration for InsightFlow AI.
All paths and constants are resolved relative to this file so the project
works regardless of the current working directory it's launched from.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = BASE_DIR / "uploads"
CHARTS_DIR = BASE_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"

for _dir in (UPLOADS_DIR, CHARTS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM settings (Groq - free tier, Llama 3.3 70B)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
OUTLIER_Z_THRESHOLD = 3.0
TOP_N_CATEGORIES = 5
CORRELATION_THRESHOLD = 0.5  # |r| above this is reported as "notable"

# Max rows to sample for expensive operations (keeps things fast on large files)
MAX_ROWS_FOR_PROFILING = 200_000

# ---------------------------------------------------------------------------
# Chart settings
# ---------------------------------------------------------------------------
CHART_DPI = 150
CHART_STYLE = "seaborn-v0_8-whitegrid"
CHART_FIGSIZE = (8, 5)

# ---------------------------------------------------------------------------
# Report settings
# ---------------------------------------------------------------------------
REPORT_TITLE = "InsightFlow AI — Executive Data Report"
COMPANY_NAME = os.environ.get("COMPANY_NAME", "Your Company")
