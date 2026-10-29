import numpy as np
import pandas as pd
import pytest

from src.player_similarity import OUTFIELD_FEATURES, find_similar


def make_pool(n=60, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(0, 1, n) for f in OUTFIELD_FEATURES})
    d["age"] = rng.uniform(18, 35, n)
    d["player_id"] = np.arange(n); d["position"] = "Attack"
    d["league_label"] = "L"; d["club_name"] = "C"; d["market_value_in_eur"] = rng.uniform(1e6, 5e7, n)
    return d


def test_excludes_self_and_orders_by_similarity():
    pool = make_pool()
    res = find_similar(pool, pool.iloc[0], k=5)
    assert 0 not in res.player_id.values and len(res) == 5
    assert res.similarity.is_monotonic_decreasing and res.similarity.between(0, 1).all()


def test_near_duplicate_is_top_hit():
    pool = make_pool(); twin = pool.iloc[3].copy(); twin["player_id"] = 999
    pool = pd.concat([pool, twin.to_frame().T.astype(pool.dtypes)], ignore_index=True)
    assert find_similar(pool, pool.iloc[3], k=1).player_id.iloc[0] == 999


def test_filters_and_position_restriction():
    pool = make_pool(); pool.loc[:9, "position"] = "Defender"
    res = find_similar(pool, pool.iloc[20], k=50, filters={"age": (18, 25)})
    assert (res.position == "Attack").all() and res.age.between(18, 25).all()
    assert find_similar(pool, pool.iloc[20], k=5, filters={"value": (1e9, 2e9)}).empty


def test_invalid_k_and_missing_features():
    pool = make_pool()
    with pytest.raises(ValueError):
        find_similar(pool, pool.iloc[0], k=0)
    row = pool.iloc[0].copy(); row["goals_p90"] = np.nan
    with pytest.raises(ValueError):
        find_similar(pool, row, k=3)
