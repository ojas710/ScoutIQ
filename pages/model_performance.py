import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C, ui
from src.formatting import format_eur_m

ui.setup("Model Performance & Methodology")
df = ui.get_data()
summary = json.loads((C.METRICS_DIR / "summary.json").read_text())
mc = pd.read_csv(C.METRICS_DIR / "model_comparison.csv")
bundle = ui.get_bundle()

tabs = st.tabs(["Model comparison", "Out-of-sample fit", "Error analysis", "Explainability", "Methodology & limits"])

with tabs[0]:
    a = mc[mc.feature_set == "A"]
    show = a[["model", "config", "val_MAE_eur", "val_RMSE_eur", "val_R2", "test_MAE_eur", "test_RMSE_eur",
              "test_R2", "test_MedianAPE_pct", "selected"]].copy()
    for c in ("val_MAE_eur", "val_RMSE_eur", "test_MAE_eur", "test_RMSE_eur"):
        show[c] = show[c].map(lambda v: format_eur_m(v, 2))
    for c in ("val_R2", "test_R2"):
        show[c] = show[c].round(3)
    show["test_MedianAPE_pct"] = show.test_MedianAPE_pct.round(1)
    st.markdown("**Model A — fundamentals only (used by the app)**")
    st.dataframe(show.rename(columns={"selected": "chosen"}), hide_index=True, width='stretch')
    st.caption("Selection used validation seasons only (log-MAE, simpler model preferred within 1%). Test seasons "
               "were scored after selection.")
    b = mc[mc.feature_set == "B"][["model", "val_MAE_eur", "val_R2", "test_MAE_eur", "test_R2", "test_MedianAPE_pct"]].copy()
    b["val_MAE_eur"] = b.val_MAE_eur.map(lambda v: format_eur_m(v, 2)); b["test_MAE_eur"] = b.test_MAE_eur.map(lambda v: format_eur_m(v, 2))
    st.markdown("**Model B — adds the player's own earlier valuations (ablation, not used for scouting)**")
    st.dataframe(b.round(3), hide_index=True, width='stretch')
    p = summary["naive_persistence_test"]
    ui.note(f"Reference: simply carrying forward the player's valuation from ≥180 days earlier gives test "
            f"MAE {format_eur_m(p['MAE_eur'], 2)}, R² {p['R2']:.2f}. Model B is mostly learning persistence; Model A "
            "answers a different question — what do performance, age, league and club context alone imply?")

with tabs[1]:
    ui.badge("pred"); ui.badge("obs")
    oos = df[df.split.isin(["val", "test"])].dropna(subset=["predicted_value"])
    split = st.radio("Split", ["test", "val"], horizontal=True)
    o = oos[oos.split == split]
    fig = px.scatter(o.sample(min(4000, len(o)), random_state=1), x="predicted_value", y="market_value_in_eur",
                     log_x=True, log_y=True, opacity=.3, color="position",
                     labels={"predicted_value": "Predicted (€)", "market_value_in_eur": "Observed (€)"})
    fig.add_shape(type="line", x0=5e4, y0=5e4, x1=2e8, y1=2e8, line=dict(color="white", dash="dash"))
    c1, c2 = st.columns(2)
    c1.plotly_chart(ui.fig_style(fig, 420), width='stretch')
    h = px.histogram(o, x="log_residual", nbins=70, title="ln(observed) − ln(predicted)")
    h.update_traces(marker_color=ui.ACCENT)
    c2.plotly_chart(ui.fig_style(h, 420), width='stretch')
    cv = summary["conformal"]
    st.markdown(f"**Interval calibration (test):** nominal 80% → **{cv['empirical_test_80']:.1%}**, "
                f"nominal 90% → **{cv['empirical_test_90']:.1%}**. Under-coverage reflects temporal drift between "
                "validation and test seasons.")

with tabs[2]:
    for name, title in (("position", "By position"), ("value_band", "By observed value band"),
                        ("season", "By season"), ("league", "By league"), ("seen_player", "Seen vs unseen players")):
        e = pd.read_csv(C.METRICS_DIR / f"error_by_{name}.csv")
        st.markdown(f"**{title}** (test seasons)")
        st.dataframe(e.round(3), hide_index=True, width='stretch')
    ui.note("<b>Signed log error &gt; 0</b> = over-prediction. Cheap players (&lt;€0.5m) are over-predicted and "
            "expensive ones (≥€10m) under-predicted — the usual shrinkage toward the mean. Players seen in training "
            "are predicted only slightly better than unseen players (see the table above), so player overlap is not the main source of accuracy.")

with tabs[3]:
    pi = pd.read_csv(C.METRICS_DIR / "permutation_importance.csv").head(15)
    fig = px.bar(pi.iloc[::-1], x="importance", y="feature", orientation="h", error_x="std",
                 title="Permutation importance on validation (increase in log-MAE)")
    fig.update_traces(marker_color=ui.ACCENT)
    st.plotly_chart(ui.fig_style(fig, 460), width='stretch')
    img = C.FIG_DIR / "09_shap_summary.png"
    if img.exists():
        st.image(str(img), caption="SHAP summary on a random sample of 400 test rows")
    ui.note("Importance and SHAP describe <b>what the fitted model relies on</b> — predictive associations, not "
            "causes. A high ‘league’ importance mostly reflects club wealth and visibility, which correlate with "
            "valuations; moving a player to another league would not mechanically change his value. Correlated "
            "features (appearances, minutes, club strength) share credit unstably.")

with tabs[4]:
    dq = json.loads((C.METRICS_DIR / "data_quality.json").read_text())
    st.markdown(f"""
**Data provenance.** Transfermarkt data as published by *dcaribou/transfermarkt-datasets* (Kaggle:
`davidcariboo/player-scores`). Files used: players, player_valuations, appearances, games, clubs, competitions.
Verify the dataset licence and Transfermarkt's terms of use before redistributing.

**Target.** `market_value_in_eur` — the player's Transfermarkt market value on the **last valuation date of each
season** (season *s* = Jul *s* – Jun *s+1*). It is a community/editor valuation, **not a transfer fee**.

**Features (all observable before the valuation date).** Age, position, league, foot, height; goals, assists,
cards, minutes, appearances and European-cup minutes in the 365 days before the valuation; club points and goal
difference per game and the share of club minutes the player was available in that window; a market-wide
valuation index (median log-value of the previous season). Raw columns that are *current* snapshots (current
value, peak value, caps, contract expiry) were excluded as leakage.

**Split.** Train {C.TRAIN_SEASONS[0]}–{C.TRAIN_SEASONS[-1]} · validation {C.VAL_SEASONS[0]}–{C.VAL_SEASONS[-1]} ·
test {C.TEST_SEASONS[0]}–{C.TEST_SEASONS[-1]} (rows: {summary['rows']['train']:,} / {summary['rows']['val']:,} /
{summary['rows']['test']:,}). Preprocessing is fitted on training rows only. The model is trained on log-value minus
the market index and predictions are inverted to EUR.

**Known limitations.**
- No xG/xA, shots, passing or progression data in this dataset; attacking/creative quality is only proxied by goals and assists.
- Minutes/appearances include all recorded competitions; 'starts' are unavailable (60+ minute matches used as proxy).
- Only players with ≥{C.MIN_WINDOW_MINUTES} minutes in the last year at clubs in 14 leagues are modelled — a selected sample of regular players.
- Valuations are subjective, rounded and update irregularly; the model reproduces Transfermarkt's valuation behaviour, not intrinsic worth.
- Cheap players are over-predicted, expensive ones under-predicted; intervals under-cover on later seasons.
- Loan status is unknown: a loanee inherits his parent/host club's context features, which can produce large apparent gaps.
- Injuries, contract length, nationality/marketing, transfer context and video-based quality are not modelled.
- Rows that dropped out at filters: {dq['player_season_snapshots']:,} player-seasons → {dq['after_league_coverage_filter']:,} (league coverage) → {dq['after_min_minutes_filter']:,} (minutes) .
""")
