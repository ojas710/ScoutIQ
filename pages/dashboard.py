import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C, ui
from src.formatting import format_eur_m

ui.setup("Executive Dashboard", "Market overview of 14 European top-flight leagues, 2014–2025 seasons")
df = ui.get_data()
summary = json.loads((C.METRICS_DIR / "summary.json").read_text())

season = st.sidebar.selectbox("Season", sorted(df.season.unique(), reverse=True),
                              help="Season s runs July s to June s+1. Each player appears once per season "
                                   "(last valuation of the season).")
cur = df[df.season == season]

ui.badge("agg")
ui.metrics_row([
    ("Player-seasons analysed", f"{len(df):,}", f"{df.player_id.nunique():,} distinct players, seasons "
                                                f"{df.season.min()}–{df.season.max()}"),
    (f"Median observed value ({season})", format_eur_m(cur.market_value_in_eur.median()), None),
    (f"Mean observed value ({season})", format_eur_m(cur.market_value_in_eur.mean()), "Mean ≫ median: the distribution is heavily right-skewed"),
    ("Best model", summary["selected_model"].replace("HistGradientBoosting", "HistGB"), f"Chosen on validation seasons 2022–23: {summary['config']}"),
])
t = summary["metrics"]["test"]
ui.badge("pred")
ui.metrics_row([("Test MAE", format_eur_m(t["MAE_eur"], 2), "Held-out seasons 2024–25"),
                ("Test RMSE", format_eur_m(t["RMSE_eur"], 2), None),
                ("Test R²", f"{t['R2']:.2f}", "In EUR scale; dominated by high-value players"),
                ("Test median abs. % error", f"{t['MedianAPE_pct']:.0f}%", "Typical relative miss per player")])

c1, c2 = st.columns(2)
with c1:
    ui.badge("obs")
    fig = px.histogram(cur, x=cur.market_value_in_eur / 1e6, nbins=60, log_y=True,
                       labels={"x": "Observed market value (€m)"}, title=f"Valuation distribution, {season}")
    fig.update_traces(marker_color=ui.ACCENT)
    st.plotly_chart(ui.fig_style(fig), width='stretch')
with c2:
    ui.badge("obs")
    top = cur.nlargest(10, "market_value_in_eur")[["player_name", "club_name", "position", "age",
                                                    "market_value_in_eur"]]
    top = top.assign(value=top.market_value_in_eur.map(format_eur_m), age=top.age.round(1))
    st.markdown(f"**Most valuable players, {season}**")
    st.dataframe(top.drop(columns="market_value_in_eur"), hide_index=True, width='stretch')

c1, c2 = st.columns(2)
with c1:
    ui.badge("agg")
    trend = df.groupby("season").market_value_in_eur.agg(median="median", mean="mean").reset_index()
    fig = px.line(trend.melt("season"), x="season", y="value", color="variable", markers=True,
                  title="Observed value by season (eligible players)", labels={"value": "EUR"})
    st.plotly_chart(ui.fig_style(fig), width='stretch')
with c2:
    ui.badge("agg")
    lg = cur.groupby("league_label").market_value_in_eur.median().sort_values().reset_index()
    fig = px.bar(lg, x="market_value_in_eur", y="league_label", orientation="h",
                 title=f"Median observed value by league, {season}", labels={"market_value_in_eur": "EUR", "league_label": ""})
    fig.update_traces(marker_color=ui.ACCENT)
    st.plotly_chart(ui.fig_style(fig, 420), width='stretch')

st.subheader("Predicted vs observed (out-of-sample)")
ui.badge("pred"); ui.badge("obs")
oos = df.dropna(subset=["predicted_value"])
sea = st.selectbox("Evaluation season", sorted(oos.season.unique(), reverse=True),
                   help="Only validation (2022–23) and test (2024–25) seasons have honest out-of-sample predictions.")
o = oos[oos.season == sea].sample(frac=1, random_state=0).head(3000)
fig = px.scatter(o, x="predicted_value", y="market_value_in_eur", log_x=True, log_y=True,
                 hover_data=["player_name", "club_name"], opacity=.35, color="position",
                 labels={"predicted_value": "Predicted (€)", "market_value_in_eur": "Observed (€)"})
fig.add_shape(type="line", x0=5e4, y0=5e4, x1=2e8, y1=2e8, line=dict(color="white", dash="dash"))
st.plotly_chart(ui.fig_style(fig, 460), width='stretch')
ui.note("Market values are Transfermarkt estimates at a valuation date, <b>not transfer fees</b>. "
        "Predictions come from a model trained only on earlier seasons (2014–2021).")
