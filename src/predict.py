"""Inference: loads the saved pipeline, validates input and returns predictions + intervals."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .evaluate import interval_from_quantile
from .preprocessing import OFFSET


class ModelNotFoundError(FileNotFoundError):
    pass


def load_bundle(path=C.MODEL_PATH) -> dict:
    try:
        return joblib.load(path)
    except FileNotFoundError as e:
        raise ModelNotFoundError(
            f"Model file {path} not found. Run `python -m src.train` first.") from e


def model_columns(bundle) -> list[str]:
    return bundle["numeric"] + bundle["categorical"] + [OFFSET]


def predict_frame(bundle, df: pd.DataFrame) -> pd.DataFrame:
    """Predict for rows that already contain all feature columns (e.g. app dataset)."""
    cols = model_columns(bundle)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    pred = bundle["model"].predict(df[cols])
    q80, q90 = bundle["conformal"]["q80_log"], bundle["conformal"]["q90_log"]
    out = pd.DataFrame(index=df.index)
    out["predicted_value"] = pred
    out["pred_lo80"], out["pred_hi80"] = interval_from_quantile(pred, q80)
    out["pred_lo90"], out["pred_hi90"] = interval_from_quantile(pred, q90)
    return out


# ----------------------------------------------------------------------------------------
# Custom player input
# ----------------------------------------------------------------------------------------
HARD_LIMITS = {   # impossible values -> rejected
    "age": (15, 45), "height_cm": (150, 210), "minutes": (C.MIN_WINDOW_MINUTES, 5900),
    "appearances": (1, 70), "apps_60plus": (0, 70), "goals": (0, 70), "assists": (0, 50),
    "yellow": (0, 30), "red": (0, 5), "euro_minutes_share": (0, 1), "club_ppg": (0, 3),
    "club_gd_pg": (-4, 4), "availability": (0, 1),
}


def validate_custom_input(inp: dict, bundle: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block prediction; warnings flag extrapolation."""
    errors, warnings = [], []
    for k, (lo, hi) in HARD_LIMITS.items():
        v = inp.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if k != "height_cm":
                errors.append(f"'{k}' is required.")
            continue
        if not lo <= v <= hi:
            errors.append(f"'{k}' must be between {lo} and {hi} (got {v}).")
    if inp.get("apps_60plus", 0) > inp.get("appearances", 0):
        errors.append("Matches with 60+ minutes cannot exceed appearances.")
    if inp.get("minutes", 0) > 0 and inp.get("appearances", 0) > 0 and \
            inp["minutes"] > inp["appearances"] * 130:
        errors.append("Minutes are impossible for the number of appearances.")
    for c in ("position", "league", "foot"):
        if str(inp.get(c)) not in bundle["categories"][c]:
            errors.append(f"Unknown {c} '{inp.get(c)}'.")
    if errors:
        return errors, warnings
    row = build_custom_row(inp, bundle)
    for c, (lo, hi) in bundle["training_ranges"].items():
        v = row[c].iloc[0]
        if pd.notna(v) and not lo <= v <= hi:
            warnings.append(f"{c}={v:.2f} lies outside the typical training range "
                            f"({lo:.2f}–{hi:.2f}); the prediction is an extrapolation.")
    return errors, warnings


def build_custom_row(inp: dict, bundle: dict) -> pd.DataFrame:
    """Derive the exact model features from raw user inputs (same formulas as training)."""
    mins = float(inp["minutes"])
    r = {"age": float(inp["age"]), "height_cm": inp.get("height_cm", np.nan),
         "minutes_90s": mins / 90.0, "appearances": inp["appearances"],
         "apps_60plus": inp["apps_60plus"], "minutes_per_app": mins / max(inp["appearances"], 1),
         "goals_p90": inp["goals"] * 90 / mins, "assists_p90": inp["assists"] * 90 / mins,
         "yellow_p90": inp["yellow"] * 90 / mins, "red_p90": inp["red"] * 90 / mins,
         "euro_minutes_share": inp["euro_minutes_share"], "club_ppg": inp["club_ppg"],
         "club_gd_pg": inp["club_gd_pg"], "availability": inp["availability"],
         "position": inp["position"], "sub_position": inp.get("sub_position") or np.nan,
         "league": inp["league"], "foot": inp.get("foot") or np.nan,
         OFFSET: bundle["latest_market_index"]}
    r["age_sq"] = r["age"] ** 2
    r["ga_p90"] = r["goals_p90"] + r["assists_p90"]
    for pos in ("Defender", "Midfield", "Attack"):
        r[f"ga_p90_{pos.lower()}"] = r["ga_p90"] if inp["position"] == pos else 0.0
    return pd.DataFrame([r])[model_columns(bundle)]


def predict_custom(inp: dict, bundle: dict) -> dict:
    errors, warnings = validate_custom_input(inp, bundle)
    if errors:
        raise ValueError(" ".join(errors))
    row = build_custom_row(inp, bundle)
    res = predict_frame(bundle, row).iloc[0].to_dict()
    res["warnings"], res["row"] = warnings, row
    return res
