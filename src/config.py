"""Central configuration: paths and modelling constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("SCOUTIQ_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"

SNAPSHOTS_PATH = PROCESSED_DIR / "snapshots.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"
MODEL_PATH = MODELS_DIR / "scoutiq_model.joblib"

# --- Dataset definition -------------------------------------------------
WINDOW_DAYS = 365          # performance window ending strictly before valuation date
MIN_WINDOW_MINUTES = 450   # ~5 full matches; below this per-90 stats are too noisy
MIN_AGE, MAX_AGE = 15.0, 45.0
FIRST_SEASON = 2013        # first season with a full 365-day appearance window (2013 rows are
                           # later dropped: no prior-season market index)
LAG1_DAYS = 180            # prior valuation must be >= 180 days older (Model B)
LAG2_DAYS = 545

# Chronological split by season (season s runs Jul 1 of s to Jun 30 of s+1)
TRAIN_SEASONS = tuple(range(2014, 2022))   # 2014-2021
VAL_SEASONS = (2022, 2023)
TEST_SEASONS = (2024, 2025)

RANDOM_STATE = 42
POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]
