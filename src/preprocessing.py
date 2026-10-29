"""Feature lists and the scikit-learn preprocessing / model pipelines."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Model A ("fundamentals"): player profile + last-365-day performance + club context.
NUMERIC_A = ["age", "age_sq", "height_cm", "minutes_90s", "appearances", "apps_60plus",
             "minutes_per_app", "goals_p90", "assists_p90", "ga_p90", "yellow_p90", "red_p90",
             "euro_minutes_share", "club_ppg", "club_gd_pg", "availability",
             "ga_p90_defender", "ga_p90_midfield", "ga_p90_attack"]
CATEGORICAL_A = ["position", "sub_position", "league", "foot"]
# Model B adds the player's own earlier valuations (>=180 days old at prediction time).
NUMERIC_B = NUMERIC_A + ["log_lag1_rel", "log_value_trend", "days_since_lag1"]

TARGET = "market_value_in_eur"
OFFSET = "market_index"


class DeflatedLogRegressor(BaseEstimator, RegressorMixin):
    """Predicts value in EUR as exp(f(x) + market_index).

    The inner regressor is trained on log(value) - market_index, i.e. on valuations deflated
    by a market-wide level observed BEFORE the valuation date. This removes the secular
    valuation inflation that a purely chronological split otherwise turns into bias.
    X must contain the offset column; the preprocessing step ignores it as a feature."""

    def __init__(self, regressor, offset_col: str = OFFSET):
        self.regressor = regressor
        self.offset_col = offset_col

    def fit(self, X, y):
        off = np.asarray(X[self.offset_col], dtype=float)
        self.regressor_ = clone(self.regressor).fit(X, np.log(np.asarray(y, float)) - off)
        return self

    def predict(self, X):
        off = np.asarray(X[self.offset_col], dtype=float)
        return np.exp(self.regressor_.predict(X) + off)


def feature_sets(kind: str = "A") -> tuple[list[str], list[str]]:
    if kind == "A":
        return list(NUMERIC_A), list(CATEGORICAL_A)
    if kind == "B":
        return list(NUMERIC_B), list(CATEGORICAL_A)
    raise ValueError("kind must be 'A' or 'B'")


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Imputers/encoders/scalers are fitted inside the pipeline -> training data only."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                          min_frequency=20))]
    return ColumnTransformer([("num", Pipeline(num_steps), numeric),
                              ("cat", Pipeline(cat_steps), categorical)],
                             verbose_feature_names_out=False)


def build_model(estimator, numeric: list[str], categorical: list[str], scale: bool):
    """Pipeline(preprocess -> estimator) wrapped so it is trained on deflated log(value) and
    predicts in EUR."""
    pipe = Pipeline([("prep", build_preprocessor(numeric, categorical, scale)),
                     ("model", estimator)])
    return DeflatedLogRegressor(pipe)
