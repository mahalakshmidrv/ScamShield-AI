"""Central configuration. Everything comes from environment variables with safe defaults."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts"
DATA_PROCESSED = ROOT / "data" / "processed"
CATEGORIES_FILE = Path(__file__).with_name("categories.json")

MAX_CHARS = int(os.environ.get("SCAMSHIELD_MAX_CHARS", "5000"))
LOW_CONF = float(os.environ.get("SCAMSHIELD_LOW_CONF_THRESHOLD", "0.40"))
STORE_INPUTS = os.environ.get("SCAMSHIELD_STORE_INPUTS", "0") == "1"

# Risk-engine weights: transparent and hand-set (NOT tuned on data).
W_WITH_URL = {"ml": 0.45, "rules": 0.35, "url": 0.20}
W_NO_URL = {"ml": 0.55, "rules": 0.45}
HIGH_T, MEDIUM_T = 0.65, 0.40


def load_categories() -> dict:
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return json.load(f)
