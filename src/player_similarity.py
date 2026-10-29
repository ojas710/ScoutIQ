"""Statistical player similarity (standardised Euclidean nearest neighbours).

The similarity index is 1 / (1 + distance) in standardised feature space. It is a descriptive
closeness measure -- NOT a probability that a player would be a successful replacement.
Market value is deliberately NOT a similarity feature (it is only used as a filter).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

OUTFIELD_FEATURES = ["age", "height_cm", "minutes_90s", "availability", "goals_p90",
                     "assists_p90", "yellow_p90", "euro_minutes_share", "club_ppg"]
GK_FEATURES = ["age", "height_cm", "minutes_90s", "availability", "yellow_p90",
               "euro_minutes_share", "club_ppg"]       # goals/assists carry no GK signal


def features_for(position: str) -> list[str]:
    return GK_FEATURES if position == "Goalkeeper" else OUTFIELD_FEATURES


def find_similar(pool: pd.DataFrame, player_row: pd.Series, k: int = 10,
                 filters: dict | None = None) -> pd.DataFrame:
    """Return the k most similar players to ``player_row`` from ``pool`` (same position group).

    ``filters`` may contain: age (lo, hi), leagues (list), clubs (list), value (lo, hi) in EUR.
    Standardisation is fitted on the full same-position pool (before filtering), so
    filters narrow candidates without changing the feature scale."""
    if k < 1:
        raise ValueError("k must be >= 1")
    feats = features_for(player_row["position"])
    same_pos = pool[(pool["position"] == player_row["position"])].copy()
    same_pos = same_pos.dropna(subset=feats, how="any")
    if len(same_pos) == 0:
        return same_pos.assign(similarity=[])
    mu, sd = same_pos[feats].mean(), same_pos[feats].std().replace(0, 1)
    cand = same_pos[same_pos["player_id"] != player_row["player_id"]]
    f = filters or {}
    if f.get("age"):
        cand = cand[cand["age"].between(*f["age"])]
    if f.get("leagues"):
        cand = cand[cand["league_label"].isin(f["leagues"])]
    if f.get("clubs"):
        cand = cand[cand["club_name"].isin(f["clubs"])]
    if f.get("value"):
        cand = cand[cand["market_value_in_eur"].between(*f["value"])]
    if cand.empty:
        return cand.assign(similarity=[])
    target = ((player_row[feats].astype(float) - mu) / sd).to_numpy(dtype=float)[None, :]
    if np.isnan(target).any():
        raise ValueError("Selected player has missing similarity features.")
    X = ((cand[feats] - mu) / sd).to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(k, len(cand))).fit(X)
    dist, idx = nn.kneighbors(target)
    out = cand.iloc[idx[0]].copy()
    out["distance"] = dist[0]
    out["similarity"] = 1.0 / (1.0 + dist[0])
    return out
