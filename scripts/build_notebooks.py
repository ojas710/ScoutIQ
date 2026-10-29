"""Generate and execute the two analysis notebooks (outputs are embedded)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import sys, warnings; sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src import config as C
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40)
plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': .25})"""

eda = nbf.v4.new_notebook(); eda.cells = [
 md("# ScoutIQ – Exploratory Data Analysis\nRaw Transfermarkt tables → player-season snapshots. Values are Transfermarkt **market values**, not transfer fees."),
 code(SETUP + "\nfrom src.data_loader import load_all\nraw = load_all()\nfor k, v in raw.items(): print(f'{k:18s} shape={v.shape}')"),
 md("## 1. Raw data quality"),
 code("v, a = raw['player_valuations'], raw['appearances']\nprint('valuation date range:', v.date.min().date(), '→', v.date.max().date())\nprint('duplicate (player, date) valuations:', v.duplicated(['player_id','date']).sum())\nprint('duplicate appearance ids:', a.appearance_id.duplicated().sum())\nprint('valuation min/median/max (EUR):', v.market_value_in_eur.min(), v.market_value_in_eur.median(), v.market_value_in_eur.max())\nprint('appearance minutes > 120:', (a.minutes_played > 120).sum())\nprint('players with missing height/foot:', raw['players'][['height_in_cm','foot']].isna().mean().round(3).to_dict())\nprint('valuations without league id:', v.player_club_domestic_competition_id.isna().mean().round(3))\nprint(open(C.METRICS_DIR/'data_quality.json').read())"),
 md("## 2. Modelling dataset (one row per player-season)\n`src/feature_engineering.py` filters to 14 leagues with appearance data, ≥450 minutes in the 365 days before the valuation date, age 15–45."),
 code("df = pd.read_parquet(C.SNAPSHOTS_PATH)\nprint(df.shape); print(df.split.value_counts().to_dict())\ndisplay(df.isna().mean().round(3)[lambda s: s>0].to_frame('missing_rate'))\ndisplay(df[['market_value_in_eur','age','minutes_90s','goals_p90','assists_p90','club_ppg']].describe().round(2))"),
 md("## 3. Target distribution\nValues are strongly right-skewed (mean ≫ median) → model `log(value)`."),
 code("fig, ax = plt.subplots(1,2,figsize=(11,3.5))\nax[0].hist(df.market_value_in_eur/1e6, bins=80, range=(0,60)); ax[0].set_title('€m (clipped at 60)')\nax[1].hist(df.log_value, bins=60); ax[1].set_title('ln(value)')\nplt.show()\nprint('skew raw / log:', round(df.market_value_in_eur.skew(),2), round(df.log_value.skew(),2))"),
 md("## 4. Value by age, position, league and season"),
 code("g = df.assign(a=df.age.round()).groupby(['a','position']).market_value_in_eur.median().unstack().loc[18:36]/1e6\ng.plot(marker='o', ms=3, figsize=(8,3.8), title='Median value by age (€m)'); plt.show()\ndisplay((df.groupby('position').market_value_in_eur.agg(['count','median','mean'])/[1,1e6,1e6]).round(2))\ndisplay((df.groupby('league_label').market_value_in_eur.median()/1e6).sort_values(ascending=False).round(2).to_frame('median €m'))\ndisplay((df.groupby('season').market_value_in_eur.median()/1e6).round(2).to_frame('median €m').T)"),
 md("Among regular players (≥450 minutes) the median value is highest for the youngest outfield players and declines with age, falling steeply after ~30; goalkeepers peak later (~29). The early peak is partly a selection effect: teenagers who already play regularly are exceptional. The median observed value of eligible players drifts upward across seasons — the reason the model uses a prior-season market index instead of treating all seasons as exchangeable."),
 md("## 5. Playing time, output and value (Spearman, with caveats)\nCorrelation is not causation: playing time, club strength and league are entangled."),
 code("cols = ['log_value','age','minutes_90s','appearances','goals_p90','assists_p90','ga_p90','club_ppg','euro_minutes_share','availability']\ncorr = df[cols].corr(method='spearman')\ndisplay(corr['log_value'].round(2).to_frame('spearman vs ln(value)'))\natt = df[df.position=='Attack']\nprint('Attackers only: spearman(ga_p90, ln value) =', round(att.ga_p90.corr(att.log_value, method='spearman'),2))\ngk = df[df.position=='Goalkeeper']\nprint('Goalkeepers: spearman(ga_p90, ln value) =', round(gk.ga_p90.corr(gk.log_value, method='spearman'),2), '(goals/assists are not meaningful for keepers)')"),
 md("## 6. Outliers\nExtreme valuations are real (not errors), so they are kept; log-scaling limits their leverage. Implausible heights are nulled at ingestion."),
 code("q1, q3 = df.log_value.quantile([.25,.75]); iqr = q3-q1\nprint('rows beyond 1.5×IQR on log scale:', ((df.log_value > q3+1.5*iqr)|(df.log_value < q1-1.5*iqr)).sum())\ndisplay(df.nlargest(8,'market_value_in_eur')[['player_name','club_name','season','age','market_value_in_eur']])\nprint('goals/90 > 1.5:', (df.goals_p90>1.5).sum(), ' (small minute samples are excluded by the 450-minute rule)')"),
 md("## 7. Model errors (held-out test seasons)"),
 code("p = pd.read_parquet(C.PREDICTIONS_PATH); t = p[p.split=='test']\nprint('test rows', len(t))\nfig, ax = plt.subplots(1,2,figsize=(11,3.8))\nax[0].scatter(t.predicted_value/1e6, t.market_value_in_eur/1e6, s=3, alpha=.2); ax[0].set(xscale='log', yscale='log', title='observed vs predicted (€m)')\nax[0].plot([.05,250],[.05,250],'k--')\nax[1].hist(t.log_residual, bins=60); ax[1].set_title('ln(observed) − ln(predicted)'); plt.show()\nfor f in ['error_by_position','error_by_value_band','error_by_seen_player']:\n    display(pd.read_csv(C.METRICS_DIR/f'{f}.csv').round(3))"),
]
exp = nbf.v4.new_notebook(); exp.cells = [
 md("# ScoutIQ – Model experiments\nTwo experiments that motivated design decisions. Every number is computed when this notebook runs."),
 code(SETUP + "\nfrom sklearn.ensemble import HistGradientBoostingRegressor\nfrom sklearn.compose import TransformedTargetRegressor\nfrom src.preprocessing import build_preprocessor, build_model, feature_sets, TARGET, OFFSET\nfrom src.evaluate import regression_metrics\nfrom sklearn.pipeline import Pipeline\ndf = pd.read_parquet(C.SNAPSHOTS_PATH)\ntr, va, te = (df[df.split==s] for s in ('train','val','test'))\nnum, cat = feature_sets('A')\nhgb = lambda: HistGradientBoostingRegressor(learning_rate=.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1., min_samples_leaf=40, random_state=42)"),
 md("## Experiment 1 – plain log target vs. market-index-deflated target\nValuations inflate over time. A chronological split turns that drift into systematic bias unless it is modelled with information available before the prediction date."),
 code("plain = TransformedTargetRegressor(Pipeline([('prep', build_preprocessor(num, cat, False)), ('m', hgb())]), func=np.log, inverse_func=np.exp)\nplain.fit(tr[num+cat], tr[TARGET])\ndefl = build_model(hgb(), num, cat, False).fit(tr[num+cat+[OFFSET]], tr[TARGET])\nrows = []\nfor name, m, cols in (('plain log target', plain, num+cat), ('deflated log target (used)', defl, num+cat+[OFFSET])):\n    for sname, d in (('val', va), ('test', te)):\n        pr = m.predict(d[cols]); r = regression_metrics(d[TARGET], pr)\n        rows.append(dict(model=name, split=sname, MAE_eur_m=r['MAE_eur']/1e6, R2=r['R2'], MAE_log=r['MAE_log'], median_signed_log_err=float(np.median(np.log(pr)-np.log(d[TARGET])))))\ndisplay(pd.DataFrame(rows).round(3))"),
 md("Positive signed error = over-prediction. The bias column shows whether deflation removes the drift."),
 md("## Experiment 2 – fundamentals (Model A) vs. adding own valuation history (Model B)\nModel B includes the valuation ≥180 days earlier. It is far more accurate but largely learns persistence, so it is *not* appropriate for finding players the market might misprice."),
 code("rows = []\nfor kind in ('A','B'):\n    n, c = feature_sets(kind); m = build_model(hgb(), n, c, False).fit(tr[n+c+[OFFSET]], tr[TARGET])\n    for sname, d in (('val', va), ('test', te)):\n        d2 = d.dropna(subset=['log_lag1_rel']) if kind=='B' else d\n        r = regression_metrics(d2[TARGET], m.predict(d2[n+c+[OFFSET]]))\n        rows.append(dict(feature_set=kind, split=sname, n=r['n'], MAE_eur_m=r['MAE_eur']/1e6, R2=r['R2'], MedianAPE=r['MedianAPE_pct']))\ndisplay(pd.DataFrame(rows).round(3))\nprint('NB: Model B rows require a prior valuation, so its evaluation set is smaller.')"),
 md("## Experiment 3 – final comparison table (from `outputs/metrics`)"),
 code("display(pd.read_csv(C.METRICS_DIR/'model_comparison.csv').round(3).iloc[:, [0,1,2,4,6,15,17,19,20]])"),
]
for name, nb in (("exploratory_data_analysis", eda), ("model_experiments", exp)):
    ExecutePreprocessor(timeout=1500, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb"); print("executed", name)
