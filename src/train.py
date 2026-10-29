"""Train, compare, select and persist the valuation model.

Protocol (chronological, leakage-resistant)
  train: seasons 2013-2021 | validation: 2022-2023 | test: 2024-2025
  * all preprocessing is fitted inside the pipeline on training rows only
  * hyper-parameters and the final model are chosen on VALIDATION only
  * the test set is scored once, after selection
  * validation residuals calibrate split-conformal prediction intervals
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from . import config as C
from .evaluate import (add_value_band, conformal_quantile, coverage, grouped_metrics,
                       interval_from_quantile, regression_metrics, valuation_residual)
from .preprocessing import OFFSET, TARGET, build_model, feature_sets

COMPLEXITY = {"Dummy (median)": 0, "Linear Regression": 1, "Ridge": 2, "Random Forest": 4,
              "HistGradientBoosting": 3}


def candidates() -> dict[str, list[tuple[str, object, bool]]]:
    """name -> list of (config label, estimator, needs_scaling)."""
    return {
        "Dummy (median)": [("median", DummyRegressor(strategy="median"), False)],
        "Linear Regression": [("ols", LinearRegression(), True)],
        "Ridge": [(f"alpha={a}", Ridge(alpha=a), True) for a in (1, 10, 100)],
        "Random Forest": [("n=200,leaf=5,mf=0.4", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.4, n_jobs=-1,
            random_state=C.RANDOM_STATE), False)],
        "HistGradientBoosting": [
            (f"lr=0.05,iter={it},leaf={lf}", HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=it, max_leaf_nodes=lf, l2_regularization=1.0,
                min_samples_leaf=40, random_state=C.RANDOM_STATE), False)
            for it, lf in ((300, 15), (600, 15), (400, 31))],
    }


def split_data(df: pd.DataFrame):
    return df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]


def fit_eval(estimator, scale, kind, tr, va):
    num, cat = feature_sets(kind)
    model = build_model(estimator, num, cat, scale)
    t0 = time.time()
    model.fit(tr[num + cat + [OFFSET]], tr[TARGET])
    fit_s = time.time() - t0
    pred = model.predict(va[num + cat + [OFFSET]])
    return model, pred, fit_s


def choose(val_table: pd.DataFrame) -> str:
    """Pick by validation log-MAE (scale-free, robust to the skewed target); among models
    within 1% of the best, prefer the simpler one (interpretability/practicality)."""
    best = val_table["MAE_log"].min()
    near = val_table[val_table["MAE_log"] <= best * 1.01]
    return near.sort_values(["complexity", "MAE_log"]).iloc[0]["model"]


def run() -> dict:
    df = pd.read_parquet(C.SNAPSHOTS_PATH)
    tr, va, te = split_data(df)
    print(f"train={len(tr):,} val={len(va):,} test={len(te):,}")
    fitted, rows = {}, []
    for kind in ("A", "B"):
        for name, configs in candidates().items():
            if kind == "B" and name not in ("Ridge", "HistGradientBoosting"):
                continue
            best = None
            for label, est, scale in configs:
                model, pred, fit_s = fit_eval(est, scale, kind, tr, va)
                m = regression_metrics(va[TARGET], pred)
                print(f"[{kind}] {name:22s} {label:28s} val MAE=€{m['MAE_eur']/1e6:.2f}m "
                      f"R2={m['R2']:.3f} MAElog={m['MAE_log']:.3f} ({fit_s:.0f}s)")
                if best is None or m["MAE_log"] < best[2]["MAE_log"]:
                    best = (label, model, m)
            fitted[(kind, name)] = best[1]
            rows.append({"feature_set": kind, "model": name, "config": best[0],
                         "complexity": COMPLEXITY[name], **{f"val_{k}": v for k, v in best[2].items()},
                         "MAE_log": best[2]["MAE_log"]})
    table = pd.DataFrame(rows)

    # ---- model selection on VALIDATION, Model A only -------------------------------
    tab_a = table[table.feature_set == "A"]
    chosen = choose(tab_a)
    print("Selected (validation):", chosen)

    # ---- test scoring AFTER selection ----------------------------------------------
    for i, r in table.iterrows():
        model = fitted[(r.feature_set, r.model)]
        num, cat = feature_sets(r.feature_set)
        m = regression_metrics(te[TARGET], model.predict(te[num + cat + [OFFSET]]))
        for k, v in m.items():
            table.loc[i, f"test_{k}"] = v
    table["selected"] = (table.feature_set == "A") & (table.model == chosen)
    table.drop(columns=["MAE_log"]).to_csv(C.METRICS_DIR / "model_comparison.csv", index=False)

    # ---- final model: predictions, conformal intervals, error analysis -------------
    model = fitted[("A", chosen)]
    num, cat = feature_sets("A")
    pred_va = model.predict(va[num + cat + [OFFSET]])
    pred_te = model.predict(te[num + cat + [OFFSET]])
    q80 = conformal_quantile(va[TARGET], pred_va, 0.20)
    q90 = conformal_quantile(va[TARGET], pred_va, 0.10)
    lo80, hi80 = interval_from_quantile(pred_te, q80)
    lo90, hi90 = interval_from_quantile(pred_te, q90)
    cov = {"nominal_80": 0.80, "empirical_test_80": coverage(te[TARGET], lo80, hi80),
           "nominal_90": 0.90, "empirical_test_90": coverage(te[TARGET], lo90, hi90),
           "q80_log": q80, "q90_log": q90,
           "median_width_ratio_90": float(np.median(hi90 / lo90))}
    print("Conformal:", cov)

    out = pd.concat([va.assign(predicted_value=pred_va), te.assign(predicted_value=pred_te)])
    out["residual"] = valuation_residual(out[TARGET], out["predicted_value"])
    out["log_residual"] = np.log(out[TARGET]) - np.log(out["predicted_value"])
    out["pred_lo90"], out["pred_hi90"] = interval_from_quantile(out["predicted_value"], q90)
    out["value_band"] = add_value_band(out[TARGET]).astype(str).values
    train_players = set(tr.player_id)
    out["player_seen_in_train"] = out.player_id.isin(train_players)
    out.to_parquet(C.PREDICTIONS_PATH, index=False)

    test_out = out[out.split == "test"]
    analysis = {
        "by_position": grouped_metrics(test_out, "position"),
        "by_value_band": grouped_metrics(test_out, "value_band"),
        "by_season": grouped_metrics(test_out, "season"),
        "by_league": grouped_metrics(test_out, "league"),
        "by_seen_player": grouped_metrics(test_out, "player_seen_in_train"),
    }
    for k, v in analysis.items():
        v.to_csv(C.METRICS_DIR / f"error_{k}.csv", index=False)

    # Model B (valuation-history) ablation + "naive persistence" reference
    naive = te.dropna(subset=["value_lag1"])
    persist = regression_metrics(naive[TARGET], naive["value_lag1"])

    # ---- bundle -----------------------------------------------------------------------
    ranges = {c: [float(tr[c].quantile(0.005)), float(tr[c].quantile(0.995))] for c in num}
    bundle = {"model": model, "numeric": num, "categorical": cat, "name": chosen,
              "config": table[table.selected].iloc[0]["config"],
              "conformal": cov, "training_ranges": ranges, "latest_market_index": float(df.sort_values("valuation_date")["market_index"].iloc[-1]),
              "categories": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cat},
              "split": {"train": list(C.TRAIN_SEASONS), "val": list(C.VAL_SEASONS),
                        "test": list(C.TEST_SEASONS)},
              "metrics": {"val": regression_metrics(va[TARGET], pred_va),
                          "test": regression_metrics(te[TARGET], pred_te)}}
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, C.MODEL_PATH, compress=3)
    summary = {"selected_model": chosen, "config": bundle["config"], "metrics": bundle["metrics"],
               "conformal": cov, "naive_persistence_test": persist,
               "rows": {"train": len(tr), "val": len(va), "test": len(te)}}
    (C.METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(table.drop(columns=["MAE_log"]).round(3).to_string())
    for k, v in analysis.items():
        print("\n", k); print(v.round(3).to_string())
    print("\npersistence baseline (test):", {k: round(v, 3) for k, v in persist.items()})
    return summary


if __name__ == "__main__":
    run()
