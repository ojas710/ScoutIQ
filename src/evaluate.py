"""Evaluation metrics, error analysis, residuals and split-conformal intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

VALUE_BANDS = [0, 500_000, 2_000_000, 10_000_000, 30_000_000, np.inf]
BAND_LABELS = ["<€0.5m", "€0.5-2m", "€2-10m", "€10-30m", "≥€30m"]


def regression_metrics(y_true, y_pred) -> dict:
    """Metrics in ORIGINAL (EUR) scale, plus scale-free log metrics.

    MAPE is only defined for strictly positive y_true; Transfermarkt values are >= €10k, and
    rows with y_true <= 0 are rejected. Median APE is added because MAPE is dominated by
    low-value players."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        raise ValueError("Values must be strictly positive (log metrics / MAPE undefined otherwise)")
    err = y_pred - y_true
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y_true - y_true.mean()) ** 2))
    ape = np.abs(err) / y_true
    lerr = np.log(y_pred) - np.log(y_true)
    return {"MAE_eur": float(np.mean(np.abs(err))), "RMSE_eur": float(np.sqrt(np.mean(err ** 2))),
            "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "MAPE_pct": float(100 * ape.mean()), "MedianAPE_pct": float(100 * np.median(ape)),
            "MAE_log": float(np.mean(np.abs(lerr))), "RMSLE": float(np.sqrt(np.mean(lerr ** 2))),
            "R2_log": 1 - float(np.sum(lerr ** 2)) / float(np.sum((np.log(y_true) - np.log(y_true).mean()) ** 2)),
            "n": int(y_true.size)}


def valuation_residual(observed, predicted):
    """Valuation residual = observed market value - predicted market value.
    Negative -> observed value BELOW the model estimate ('potentially undervalued relative to
    model estimate'); positive -> observed above the model estimate."""
    return np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)


def add_value_band(values) -> pd.Series:
    return pd.cut(pd.Series(np.asarray(values)), VALUE_BANDS, labels=BAND_LABELS, right=False)


def grouped_metrics(df: pd.DataFrame, group_col: str, true_col="market_value_in_eur",
                    pred_col="predicted_value") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, observed=True):
        if len(g) >= 20:
            m = regression_metrics(g[true_col], g[pred_col])
            m[group_col] = key
            m["median_signed_log_error"] = float(np.median(np.log(g[pred_col]) - np.log(g[true_col])))
            rows.append(m)
    cols = [group_col, "n", "MAE_eur", "RMSE_eur", "R2", "MedianAPE_pct", "MAE_log",
            "median_signed_log_error"]
    return pd.DataFrame(rows)[cols]


def conformal_quantile(y_true, y_pred, alpha: float) -> float:
    """Split-conformal: finite-sample-corrected (1-alpha) quantile of |log(y)-log(yhat)|
    on a held-out CALIBRATION set (here: the validation seasons)."""
    scores = np.abs(np.log(np.asarray(y_true, float)) - np.log(np.asarray(y_pred, float)))
    n = scores.size
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def interval_from_quantile(y_pred, q: float):
    y_pred = np.asarray(y_pred, float)
    return y_pred * np.exp(-q), y_pred * np.exp(q)


def coverage(y_true, lo, hi) -> float:
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
