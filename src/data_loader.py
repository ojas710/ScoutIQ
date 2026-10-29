"""Load and validate the raw Transfermarkt CSV files.

Expected files (dcaribou/transfermarkt-datasets, Kaggle: davidcariboo/player-scores).
Files may carry an arbitrary prefix (e.g. '1793..._appearances.csv'); lookup is by suffix.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_DIR

REQUIRED_COLUMNS = {
    "players": ["player_id", "name", "date_of_birth", "position", "sub_position", "foot", "height_in_cm"],
    "player_valuations": ["player_id", "date", "market_value_in_eur", "current_club_id",
                          "player_club_domestic_competition_id"],
    "appearances": ["player_id", "player_club_id", "date", "competition_id", "yellow_cards",
                    "red_cards", "goals", "assists", "minutes_played"],
    "games": ["game_id", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"],
    "clubs": ["club_id", "name", "domestic_competition_id"],
    "competitions": ["competition_id", "name", "type", "sub_type", "country_name"],
}
DATE_COLUMNS = {"players": ["date_of_birth"], "player_valuations": ["date"],
                "appearances": ["date"], "games": ["date"]}


class DataNotFoundError(FileNotFoundError):
    pass


def find_file(table: str, raw_dir: Path | str = RAW_DIR) -> Path:
    raw_dir = Path(raw_dir)
    exact = raw_dir / f"{table}.csv"
    if exact.exists():
        return exact
    matches = sorted(raw_dir.glob(f"*{table}.csv"))
    if not matches:
        raise DataNotFoundError(
            f"Could not find '{table}.csv' in {raw_dir}. Download the Transfermarkt dataset "
            "(see data/README.md) and place the CSV files there, or set SCOUTIQ_RAW_DIR."
        )
    return matches[0]


def load_table(table: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    path = find_file(table, raw_dir)
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS[table] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns {missing}; schema may have changed.")
    for col in DATE_COLUMNS.get(table, []):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_all(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {t: load_table(t, raw_dir) for t in REQUIRED_COLUMNS}
