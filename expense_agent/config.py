"""Central configuration. Every threshold that encodes policy lives here,
not in prompts — deterministic limits must be enforceable in code."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- Policy thresholds (deterministic layer) ---
AUTO_APPROVE_LIMIT = _f("AUTO_APPROVE_LIMIT", 500.0)      # LLM approvals above this are forced to escalate
HARD_CEILING = _f("HARD_CEILING", 10_000.0)               # at/above this the LLM is never consulted
RECEIPT_REQUIRED_ABOVE = _f("RECEIPT_REQUIRED_ABOVE", 25.0)
CONFIDENCE_FLOOR = _f("CONFIDENCE_FLOOR", 0.6)
DUPLICATE_WINDOW_HOURS = _f("DUPLICATE_WINDOW_HOURS", 24.0)

VENDOR_BLACKLIST = {
    v.strip().lower()
    for v in os.getenv(
        "VENDOR_BLACKLIST",
        "shadow consulting ltd,quickcash services,luxe gifts co",
    ).split(",")
    if v.strip()
}

# --- Retrieval / LLM ---
TOP_K = int(os.getenv("TOP_K", "4"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# USD per 1M tokens (input, output) — used for per-decision cost telemetry
_PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
}


def prices_for(model: str) -> tuple[float, float]:
    return _PRICES.get(model, (0.30, 2.50))

# --- Storage ---
DB_PATH = Path(os.getenv("EXPENSE_AGENT_DB", str(ROOT / "data" / "expense_agent.db")))
CHECKPOINT_DB = Path(os.getenv("EXPENSE_AGENT_CHECKPOINT_DB", str(ROOT / "data" / "checkpoints.db")))
POLICY_PATH = Path(os.getenv("EXPENSE_AGENT_POLICY", str(ROOT / "policy" / "expense_policy.md")))
CACHE_DIR = ROOT / ".cache"
