import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataNotFoundError, find_file, load_table
from src.feature_engineering import _cum_by_key, _window_sum, season_of


def test_season_of():
    d = pd.Series(pd.to_datetime(["2020-06-30", "2020-07-01", "2021-01-15"]))
    assert list(season_of(d)) == [2019, 2020, 2020]


def test_window_excludes_valuation_day_and_older_than_365_days():
    t = pd.Timestamp("2022-06-01")
    apps = pd.DataFrame({
        "player_id": 1,
        "date": [t - pd.Timedelta(days=366), t - pd.Timedelta(days=365), t - pd.Timedelta(days=10),
                 t, t + pd.Timedelta(days=5)],        # last two are on / after valuation date
        "minutes": [90.0] * 5, "goals": [1.0] * 5})
    snap = pd.DataFrame({"snap_id": [0], "player_id": [1], "date": [t],
                         "date_start": [t - pd.Timedelta(days=365)]})
    cum = _cum_by_key(apps, "player_id", ["minutes", "goals"])
    out = _window_sum(snap, cum, "player_id", "player_id", ["minutes", "goals"], "w_")
    # window is [t-365d, t): includes day t-365 and t-10 only
    assert out["w_minutes"].iloc[0] == 180 and out["w_goals"].iloc[0] == 2


def test_window_handles_players_without_history():
    t = pd.Timestamp("2022-06-01")
    snap = pd.DataFrame({"snap_id": [0], "player_id": [9], "date": [t],
                         "date_start": [t - pd.Timedelta(days=365)]})
    apps = pd.DataFrame({"player_id": [1], "date": [t - pd.Timedelta(days=5)], "minutes": [90.0]})
    out = _window_sum(snap, _cum_by_key(apps, "player_id", ["minutes"]), "player_id", "player_id",
                      ["minutes"], "w_")
    assert out["w_minutes"].iloc[0] == 0


def test_missing_raw_data_gives_clear_error(tmp_path):
    with pytest.raises(DataNotFoundError, match="data/README.md"):
        find_file("players", tmp_path)


def test_schema_change_detected(tmp_path):
    pd.DataFrame({"player_id": [1]}).to_csv(tmp_path / "players.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_table("players", tmp_path)
