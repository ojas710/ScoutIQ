"""Global importance (permutation) and SHAP explanations.

Notes on interpretation: both methods describe how the MODEL's output changes with its
inputs (predictive associations). They do not show that a statistic causes a market value to
change, and correlated features share credit in unstable ways. SHAP values here are on the
deflated log-value scale: exp(contribution) - 1 is the approximate % effect on the estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from . import config as C
from .preprocessing import OFFSET, TARGET

BACKGROUND_PATH = C.MODELS_DIR / "shap_background.parquet"


def _inner(bundle):
    return bundle["model"].regressor_            # Pipeline(prep -> estimator)


def original_feature(name: str, bundle: dict) -> str:
    """Map a transformed (one-hot) column back to its source feature."""
    for c in bundle["categorical"]:
        if name.startswith(c + "_"):
            return c
    return name


def permutation_table(bundle, df: pd.DataFrame, n: int = 4000, repeats: int = 3) -> pd.DataFrame:
    """Increase in log-MAE when a feature is shuffled (on held-out validation rows)."""
    cols = bundle["numeric"] + bundle["categorical"]
    sample = df.sample(min(n, len(df)), random_state=C.RANDOM_STATE)
    inner = _inner(bundle)
    y = sample["log_value"].to_numpy() - sample[OFFSET].to_numpy()

    def scorer(est, X, y_):
        return -np.mean(np.abs(est.predict(X) - y_))
    res = permutation_importance(inner, sample[cols], y, scoring=scorer, n_repeats=repeats,
                                 random_state=C.RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": cols, "importance": res.importances_mean,
                          "std": res.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def make_background(train: pd.DataFrame, bundle: dict, n: int = 100) -> pd.DataFrame:
    cols = bundle["numeric"] + bundle["categorical"]
    bg = train[cols].sample(n, random_state=C.RANDOM_STATE)
    bg.to_parquet(BACKGROUND_PATH, index=False)
    return bg


def build_explainer(bundle: dict, background: pd.DataFrame):
    inner = _inner(bundle)
    prep, est = inner.named_steps["prep"], inner.named_steps["model"]
    bg_t = prep.transform(background[bundle["numeric"] + bundle["categorical"]])
    names = list(prep.get_feature_names_out())
    explainer = shap.Explainer(est.predict, shap.maskers.Independent(bg_t, max_samples=100),
                               algorithm="permutation", feature_names=names)
    return explainer, prep, names


def shap_for(bundle, explainer_tuple, rows: pd.DataFrame, max_evals: int | None = None):
    explainer, prep, names = explainer_tuple
    X = prep.transform(rows[bundle["numeric"] + bundle["categorical"]])
    return explainer(X, max_evals=max_evals or 2 * X.shape[1] + 1), names


def aggregate_contributions(shap_values: np.ndarray, names: list[str], bundle: dict) -> pd.DataFrame:
    """Sum one-hot contributions per original feature for ONE row."""
    df = pd.DataFrame({"feature": [original_feature(n, bundle) for n in names],
                       "contribution": shap_values})
    out = df.groupby("feature", as_index=False)["contribution"].sum()
    out["approx_pct_effect"] = (np.exp(out["contribution"]) - 1) * 100
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def global_shap(bundle, train: pd.DataFrame, test: pd.DataFrame, n: int = 400):
    bg = make_background(train, bundle)
    ex = build_explainer(bundle, bg)
    sample = test.sample(min(n, len(test)), random_state=C.RANDOM_STATE)
    sv, names = shap_for(bundle, ex, sample)
    return sample, sv, names
