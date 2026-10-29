"""Generate static report figures into outputs/figures (matplotlib)."""
import os
os.environ["TQDM_DISABLE"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config as C

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10})
ACC = "#1f9d6b"
df = pd.read_parquet(C.SNAPSHOTS_PATH)
pred = pd.read_parquet(C.PREDICTIONS_PATH)
test = pred[pred.split == "test"]
F = C.FIG_DIR; F.mkdir(parents=True, exist_ok=True)

def save(name): plt.tight_layout(); plt.savefig(F / name); plt.close()

# 1 value distribution (raw vs log)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(df.market_value_in_eur / 1e6, bins=80, color=ACC); ax[0].set_xlim(0, 60)
ax[0].set(title="Market value (EUR m, clipped at 60)", xlabel="€m", ylabel="player-seasons")
ax[1].hist(df.log_value, bins=60, color=ACC); ax[1].set(title="log(market value)", xlabel="ln(EUR)")
save("01_value_distribution.png")

# 2 age curve
g = df.assign(a=df.age.round()).groupby(["a", "position"]).market_value_in_eur.median().unstack() / 1e6
g = g.loc[18:36]
g.plot(figsize=(7, 4), marker="o", ms=3); plt.title("Median market value by age and position (observed)")
plt.ylabel("€m"); plt.xlabel("age")
save("02_age_value_curve.png")

# 3 season medians (inflation)
s = df.groupby("season").market_value_in_eur.median() / 1e6
plt.figure(figsize=(6.5, 3.6)); plt.bar(s.index, s.values, color=ACC)
plt.title("Median observed value of eligible players by season"); plt.ylabel("€m")
save("03_season_median_value.png")

# 4 playing time / goals vs value
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
d = df[df.position.isin(["Attack"])]
ax[0].scatter(d.ga_p90, d.log_value, s=3, alpha=.2, color=ACC); ax[0].set(title="Attackers: G+A per 90 vs ln(value)", xlabel="G+A /90", ylabel="ln(EUR)")
ax[1].scatter(df.minutes_90s, df.log_value, s=3, alpha=.15, color=ACC); ax[1].set(title="Minutes (90s) vs ln(value), all positions", xlabel="90s in last 365d")
save("04_performance_vs_value.png")

# 5 actual vs predicted
plt.figure(figsize=(5.2, 5))
plt.scatter(test.predicted_value / 1e6, test.market_value_in_eur / 1e6, s=4, alpha=.25, color=ACC)
lim = [0.02, 250]; plt.plot(lim, lim, "k--", lw=1); plt.xscale("log"); plt.yscale("log")
plt.xlabel("predicted €m"); plt.ylabel("observed €m"); plt.title("Test seasons: observed vs predicted (log axes)")
save("05_actual_vs_predicted.png")

# 6 residual distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(test.log_residual, bins=70, color=ACC); ax[0].axvline(0, c="k", lw=1)
ax[0].set(title="log residual = ln(observed) − ln(predicted)", xlabel="log residual")
r = test.residual / 1e6; ax[1].hist(r.clip(-20, 20), bins=80, color=ACC); ax[1].axvline(0, c="k", lw=1)
ax[1].set(title="EUR residual (observed − predicted), clipped ±20", xlabel="€m")
save("06_residuals.png")

# 7 error by position and value band
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ep = pd.read_csv(C.METRICS_DIR / "error_by_position.csv"); eb = pd.read_csv(C.METRICS_DIR / "error_by_value_band.csv")
order = ["<€0.5m", "€0.5-2m", "€2-10m", "€10-30m", "≥€30m"]; eb = eb.set_index("value_band").loc[order].reset_index()
ax[0].bar(ep.position, ep.MAE_log, color=ACC); ax[0].set(title="Test MAE (log scale) by position")
ax[1].bar(eb.value_band, eb.median_signed_log_error, color=ACC); ax[1].axhline(0, c="k", lw=1)
ax[1].set(title="Median signed log error by value band\n(+ = over-prediction)")
save("07_error_by_group.png")

# 8 permutation importance
pi = pd.read_csv(C.METRICS_DIR / "permutation_importance.csv").head(15)[::-1]
plt.figure(figsize=(6.5, 4.5)); plt.barh(pi.feature, pi.importance, xerr=pi["std"], color=ACC)
plt.title("Permutation importance (↑ log-MAE when shuffled, validation)")
save("08_permutation_importance.png")

# 9 SHAP summary
z = np.load(C.METRICS_DIR / "shap_global.npz", allow_pickle=True)
shap.summary_plot(z["values"], z["data"], feature_names=list(z["names"]), show=False, max_display=15)
plt.title("SHAP summary (test sample, deflated log-value scale)"); save("09_shap_summary.png")

# 10 model comparison
mc = pd.read_csv(C.METRICS_DIR / "model_comparison.csv"); mc = mc[mc.feature_set == "A"]
plt.figure(figsize=(7, 3.6)); x = np.arange(len(mc))
plt.bar(x - .2, mc.val_MAE_eur / 1e6, .4, label="validation", color="#6b7280")
plt.bar(x + .2, mc.test_MAE_eur / 1e6, .4, label="test", color=ACC)
plt.xticks(x, mc.model, rotation=20, ha="right"); plt.ylabel("MAE €m"); plt.legend(); plt.title("Model comparison (Model A)")
save("10_model_comparison.png")
print("figures:", sorted(p.name for p in F.glob("*.png")))
