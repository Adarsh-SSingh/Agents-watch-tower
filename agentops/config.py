"""Central configuration. Nothing else in the project hardcodes these values."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- Paths -----------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
# Tests can point to a temporary database via AGENTOPS_DB_PATH.
DB_PATH = Path(os.getenv("AGENTOPS_DB_PATH", DATA_DIR / "agentops.db"))
LEGACY_TRACE_FILE = DATA_DIR / "raw" / "traces.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- LLM -----------------------------------------------------------
MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 500
# No hidden SDK retries and a fixed timeout, so latency data stays honest.
LLM_MAX_RETRIES = 0
LLM_TIMEOUT_SECONDS = 60


def get_nvidia_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is missing. Add it to your .env file.")
    return key