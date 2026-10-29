"""Build the leakage-safe player-season snapshot dataset.

Unit of observation
-------------------
One row per (player, season): the player's LAST Transfermarkt valuation inside the season
(season s = 1 Jul s .. 30 Jun s+1). The valuation date is ``t``.

Target
------
``market_value_in_eur`` on date ``t`` (modelled on a log scale).
This is a Transfermarkt crowd/editor *market value estimate*, NOT a transfer fee.

Leakage rules
-------------
* Performance features use appearances in [t-365d, t) only (strictly before t).
* Club-context features use club games in [t-365d, t) only.
* Valuation-history features (Model B only) use valuations at least 180 days before t.
* Time-varying "current" columns of the raw files (players.market_value_in_eur,
  highest_market_value_in_eur, international_caps/goals, contract_expiration_date,
  clubs.total_market_value, squad stats) are deliberately NOT used: they are snapshots taken
  at scrape time and would leak future information.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import load_all

EURO_CLUB_COMPETITIONS = {"CL", "CLQ", "EL", "ELQ", "UCOL", "ECLQ", "USC"}

CUM_COLS = ["minutes", "goals", "assists", "apps", "apps60", "yellow", "red", "euro_minutes"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def season_of(dates: pd.Series) -> pd.Series:
    """Season label: year of the August-start; Jul-Dec -> year, Jan-Jun -> year-1."""
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


def _cum_by_key(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Per-key cumulative sums collapsed to one row per (key, date), sorted by date."""
    daily = df.groupby([key, "date"], as_index=False)[cols].sum()
    daily = daily.sort_values([key, "date"])
    daily[cols] = daily.groupby(key)[cols].cumsum()
    return daily.sort_values("date").reset_index(drop=True)


def _window_sum(snap: pd.DataFrame, cum: pd.DataFrame, key_left: str, key_right: str,
                cols: list[str], prefix: str) -> pd.DataFrame:
    """Sum of ``cols`` over [t-WINDOW, t) using two as-of joins on cumulative sums."""
    cum = cum.rename(columns={key_right: key_left})
    out = snap[["snap_id"]].copy()
    ends = {}
    for tag, col in (("end", "date"), ("start", "date_start")):
        left = snap[["snap_id", key_left, col]].rename(columns={col: "date"}).sort_values("date")
        m = pd.merge_asof(left, cum, on="date", by=key_left, allow_exact_matches=False)
        ends[tag] = m.set_index("snap_id")[cols].fillna(0.0)
    diff = (ends["end"] - ends["start"].reindex(ends["end"].index)).clip(lower=0)
    diff.columns = [f"{prefix}{c}" for c in cols]
    return out.merge(diff, left_on="snap_id", right_index=True, how="left")


# ----------------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------------
def build_snapshots(raw_dir=None, save_report: bool = True) -> pd.DataFrame:
    tables = load_all(raw_dir) if raw_dir else load_all()
    players, vals, apps = tables["players"], tables["player_valuations"], tables["appearances"]
    games, clubs, comps = tables["games"], tables["clubs"], tables["competitions"]
    report: dict = {}

    # ---- players -----------------------------------------------------------
    players = players.drop_duplicates("player_id")
    players = players[players["position"].isin(C.POSITIONS)].copy()
    players.loc[~players["height_in_cm"].between(150, 210), "height_in_cm"] = np.nan  # implausible
    players["foot"] = players["foot"].where(players["foot"].isin(["left", "right", "both"]))
    report["players_valid_position"] = int(len(players))

    # ---- leagues with performance coverage ---------------------------------
    dom = comps[comps["type"] == "domestic_league"]
    covered = sorted(set(apps["competition_id"]) & set(dom["competition_id"]))
    report["covered_leagues"] = covered
    comps = comps.assign(label=comps["name"].str.replace("-", " ").str.title() + " (" +
                         comps["country_name"].fillna("") + ")")
    league_label = dict(zip(comps["competition_id"], comps["label"]))

    # ---- valuations ----------------------------------------------------------
    vals = vals.dropna(subset=["date", "market_value_in_eur"])
    vals = vals[vals["market_value_in_eur"] > 0].drop_duplicates(["player_id", "date"])
    report["valuation_rows_raw"] = int(len(vals))
    club_comp = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))
    vals["league"] = vals["player_club_domestic_competition_id"].fillna(
        vals["current_club_id"].map(club_comp))
    vals = vals.sort_values(["player_id", "date"]).reset_index(drop=True)

    # history lags for Model B (computed on the FULL valuation history, before sampling)
    lag_src = vals[["player_id", "date", "market_value_in_eur"]].sort_values("date")

    # ---- one snapshot per player-season -----------------------------------
    vals["season"] = season_of(vals["date"])
    snap = (vals.sort_values("date").groupby(["player_id", "season"], as_index=False).tail(1)
            .reset_index(drop=True))
    snap = snap[snap["season"].between(C.FIRST_SEASON, max(C.TEST_SEASONS))]
    report["player_season_snapshots"] = int(len(snap))
    snap = snap[snap["league"].isin(covered)]
    report["after_league_coverage_filter"] = int(len(snap))
    snap = snap.merge(players[["player_id", "name", "date_of_birth", "position", "sub_position",
                               "foot", "height_in_cm"]], on="player_id", how="inner")
    snap["age"] = (snap["date"] - snap["date_of_birth"]).dt.days / 365.25
    snap = snap[snap["age"].between(C.MIN_AGE, C.MAX_AGE)]
    report["after_age_filter"] = int(len(snap))
    snap = snap.reset_index(drop=True)
    snap["snap_id"] = np.arange(len(snap))
    snap["date_start"] = snap["date"] - pd.Timedelta(days=C.WINDOW_DAYS)

    # ---- performance window (appearances) ---------------------------------
    apps = apps.drop_duplicates("appearance_id")
    apps = apps[apps["minutes_played"].between(0, 130)]
    apps = apps.assign(
        minutes=apps["minutes_played"].astype(float), apps=1.0,
        apps60=(apps["minutes_played"] >= 60).astype(float),
        yellow=apps["yellow_cards"].astype(float), red=apps["red_cards"].astype(float),
        goals=apps["goals"].astype(float), assists=apps["assists"].astype(float),
        euro_minutes=np.where(apps["competition_id"].isin(EURO_CLUB_COMPETITIONS),
                              apps["minutes_played"], 0).astype(float))
    cum_p = _cum_by_key(apps, "player_id", CUM_COLS)
    perf = _window_sum(snap, cum_p, "player_id", "player_id", CUM_COLS, "w_")
    snap = snap.merge(perf, on="snap_id", how="left")
    snap = snap[snap["w_minutes"] >= C.MIN_WINDOW_MINUTES].copy()
    report["after_min_minutes_filter"] = int(len(snap))

    # ---- club context (games) ---------------------------------------------
    g = games.dropna(subset=["date", "home_club_goals", "away_club_goals"])
    hg, ag = g["home_club_goals"], g["away_club_goals"]
    home = pd.DataFrame({"club_id": g["home_club_id"], "date": g["date"], "games": 1.0,
                         "points": np.where(hg > ag, 3, np.where(hg == ag, 1, 0)).astype(float),
                         "gd": (hg - ag).astype(float)})
    away = pd.DataFrame({"club_id": g["away_club_id"], "date": g["date"], "games": 1.0,
                         "points": np.where(ag > hg, 3, np.where(hg == ag, 1, 0)).astype(float),
                         "gd": (ag - hg).astype(float)})
    cum_c = _cum_by_key(pd.concat([home, away]), "club_id", ["games", "points", "gd"])
    snap["club_id"] = snap["current_club_id"]
    club = _window_sum(snap, cum_c, "club_id", "club_id", ["games", "points", "gd"], "c_")
    snap = snap.merge(club, on="snap_id", how="left")

    # ---- valuation history (Model B) --------------------------------------
    for tag, days in (("lag1", C.LAG1_DAYS), ("lag2", C.LAG2_DAYS)):
        left = snap[["snap_id", "player_id", "date"]].copy()
        left["date"] = left["date"] - pd.Timedelta(days=days)
        m = pd.merge_asof(left.sort_values("date"), lag_src.rename(
            columns={"date": f"{tag}_date", "market_value_in_eur": f"value_{tag}"}).assign(
            date=lambda d: d[f"{tag}_date"]).sort_values("date"),
            on="date", by="player_id", direction="backward")
        snap = snap.merge(m[["snap_id", f"value_{tag}"] + ([f"{tag}_date"] if tag == "lag1" else [])],
                          on="snap_id", how="left")

    # ---- derived features -----------------------------------------------------
    mins = snap["w_minutes"]
    snap["minutes_90s"] = mins / 90.0
    snap["goals_p90"] = snap["w_goals"] * 90 / mins
    snap["assists_p90"] = snap["w_assists"] * 90 / mins
    snap["ga_p90"] = snap["goals_p90"] + snap["assists_p90"]
    snap["yellow_p90"] = snap["w_yellow"] * 90 / mins
    snap["red_p90"] = snap["w_red"] * 90 / mins
    snap["appearances"] = snap["w_apps"]
    snap["apps_60plus"] = snap["w_apps60"]
    snap["minutes_per_app"] = mins / snap["w_apps"].clip(lower=1)
    snap["euro_minutes_share"] = snap["w_euro_minutes"] / mins
    cg = snap["c_games"].where(snap["c_games"] >= 5)
    snap["club_games"] = cg
    snap["club_ppg"] = snap["c_points"] / cg
    snap["club_gd_pg"] = snap["c_gd"] / cg
    snap["availability"] = (mins / (cg * 90)).clip(0, 1)
    snap["age_sq"] = snap["age"] ** 2
    for pos in ("Defender", "Midfield", "Attack"):          # position-aware output metrics
        snap[f"ga_p90_{pos.lower()}"] = np.where(snap["position"] == pos, snap["ga_p90"], 0.0)
    snap["log_value_lag1"] = np.log(snap["value_lag1"])
    snap["log_value_trend"] = np.log(snap["value_lag1"] / snap["value_lag2"])
    snap["days_since_lag1"] = (snap["date"] - snap["lag1_date"]).dt.days

    snap["log_value"] = np.log(snap["market_value_in_eur"])
    # Market index = median log-value of the ELIGIBLE population (same filters as the model
    # rows) in the PREVIOUS season. Season s-1 ended before any season-s valuation date, so it
    # is known at prediction time. It captures secular valuation inflation without leakage.
    season_median = snap.groupby("season")["log_value"].median()
    snap["market_index"] = snap["season"].map(lambda s_: season_median.get(s_ - 1, np.nan)).astype("float32")
    snap = snap.dropna(subset=["market_index"])        # first season has no prior index
    snap["log_lag1_rel"] = snap["log_value_lag1"] - snap["market_index"]   # Model B, deflated scale
    snap["split"] = np.select(
        [snap["season"].isin(C.TRAIN_SEASONS), snap["season"].isin(C.VAL_SEASONS),
         snap["season"].isin(C.TEST_SEASONS)], ["train", "val", "test"], default="drop")
    snap = snap[snap["split"] != "drop"]

    # ---- display columns --------------------------------------------------------
    club_name = dict(zip(clubs["club_id"], clubs["name"]))
    snap["club_name"] = snap["club_id"].map(club_name).fillna(snap["current_club_name"])
    snap["league_label"] = snap["league"].map(league_label)
    snap = snap.rename(columns={"name": "player_name", "date": "valuation_date",
                                "height_in_cm": "height_cm"})

    keep = ["player_id", "player_name", "club_id", "club_name", "league", "league_label", "season",
            "valuation_date", "split", "market_value_in_eur", "log_value", "position",
            "sub_position", "foot", "height_cm", "age", "age_sq", "minutes_90s", "appearances",
            "apps_60plus", "minutes_per_app", "goals_p90", "assists_p90", "ga_p90", "yellow_p90",
            "red_p90", "euro_minutes_share", "club_games", "club_ppg", "club_gd_pg", "availability",
            "ga_p90_defender", "ga_p90_midfield", "ga_p90_attack", "w_minutes", "w_goals",
            "w_assists", "market_index", "value_lag1", "log_value_lag1", "log_lag1_rel", "log_value_trend", "days_since_lag1"]
    snap = snap[keep].sort_values(["season", "player_id"]).reset_index(drop=True)
    for c in snap.select_dtypes("float64").columns:
        snap[c] = snap[c].astype("float32") if c != "market_value_in_eur" else snap[c]

    report["final_rows"] = int(len(snap))
    report["final_players"] = int(snap["player_id"].nunique())
    report["rows_by_split"] = snap["split"].value_counts().to_dict()
    report["missing_rate"] = snap.isna().mean().round(4).loc[lambda s: s > 0].to_dict()
    if save_report:
        C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        (C.METRICS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
    return snap


def run(out_path=C.SNAPSHOTS_PATH) -> pd.DataFrame:
    df = build_snapshots()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows -> {out_path}")
    return df


if __name__ == "__main__":
    run()
