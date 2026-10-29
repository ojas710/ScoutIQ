"""Shared Streamlit helpers: cached loaders, styling, badges and reusable widgets."""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config as C
from .explainability import (BACKGROUND_PATH, aggregate_contributions, build_explainer,
                             shap_for)
from .formatting import format_eur_m
from .player_similarity import find_similar
from .predict import (ModelNotFoundError, build_custom_row, load_bundle, predict_custom,
                      validate_custom_input)

ACCENT = "#1f9d6b"
PLOTLY_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=45, b=10))

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.74rem;
        font-weight:600; margin-right:6px; border:1px solid #30363d;}
.b-obs {background:#102a43; color:#7cc4ff;}
.b-pred {background:#14342a; color:#5fd9a5;}
.b-agg {background:#2b2a14; color:#e6d36a;}
.note {border-left:3px solid #1f9d6b; padding:6px 12px; background:#161b22; border-radius:4px;
       font-size:0.88rem; color:#b6c2cf;}
[data-testid="stMetric"] {background:#161b22; border:1px solid #21262d; padding:12px 14px;
       border-radius:10px;}
</style>
"""

BADGES = {"obs": ('<span class="badge b-obs">OBSERVED · Transfermarkt market value</span>'),
          "pred": ('<span class="badge b-pred">MODEL PREDICTION</span>'),
          "agg": ('<span class="badge b-agg">AGGREGATE STATISTIC</span>')}


def setup(title: str, subtitle: str | None = None):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def badge(kind: str) -> None:
    st.markdown(BADGES[kind], unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def fig_style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


@st.cache_resource(show_spinner="Loading model…")
def get_bundle():
    try:
        return load_bundle()
    except ModelNotFoundError as e:
        st.error(str(e)); st.stop()


@st.cache_data(show_spinner="Loading data…")
def get_data() -> pd.DataFrame:
    """All player-season snapshots; predictions only exist for out-of-sample (val/test) rows."""
    if not C.SNAPSHOTS_PATH.exists() or not C.PREDICTIONS_PATH.exists():
        st.error("Processed data not found. Run `python -m src.feature_engineering` and "
                 "`python -m src.train` (see README)."); st.stop()
    snaps = pd.read_parquet(C.SNAPSHOTS_PATH)
    preds = pd.read_parquet(C.PREDICTIONS_PATH)[
        ["player_id", "season", "predicted_value", "pred_lo90", "pred_hi90", "residual",
         "log_residual", "player_seen_in_train"]]
    return snaps.merge(preds, on=["player_id", "season"], how="left")


@st.cache_resource(show_spinner="Preparing explainer…")
def get_explainer():
    if not BACKGROUND_PATH.exists():
        return None
    return build_explainer(get_bundle(), pd.read_parquet(BACKGROUND_PATH))


def local_explanation(rows: pd.DataFrame) -> pd.DataFrame | None:
    """SHAP contributions (grouped by original feature) for ONE row; None if unavailable."""
    ex = get_explainer()
    if ex is None:
        return None
    bundle = get_bundle()
    sv, names = shap_for(bundle, ex, rows)
    return aggregate_contributions(sv.values[0], names, bundle)


def contribution_chart(contrib: pd.DataFrame, top: int = 10) -> go.Figure:
    c = contrib.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=c["contribution"], y=c["feature"], orientation="h",
                           marker_color=np.where(c["contribution"] >= 0, ACCENT, "#d1495b"),
                           customdata=c["approx_pct_effect"],
                           hovertemplate="%{y}: %{customdata:.0f}% vs. reference<extra></extra>"))
    fig.update_layout(title="Factors associated with higher (green) / lower (red) estimate",
                      xaxis_title="SHAP contribution (log-value)")
    return fig_style(fig, 380)


def league_map(df: pd.DataFrame) -> dict[str, str]:
    return dict(df[["league", "league_label"]].drop_duplicates().values)


def player_options(df: pd.DataFrame) -> pd.DataFrame:
    return (df.sort_values("market_value_in_eur", ascending=False)
            .drop_duplicates("player_id")[["player_id", "player_name", "club_name"]])


def metrics_row(items: list[tuple[str, str, str | None]]):
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        col.metric(label, value, help=help_)


# ---------------------------------------------------------------------------------------
# custom-player form (used on the Valuation page and the Custom Predictor page)
# ---------------------------------------------------------------------------------------
def custom_form(key: str, df: pd.DataFrame, show_similar: bool = True):
    bundle = get_bundle()
    lm = league_map(df)
    rng = bundle["training_ranges"]
    with st.form(f"custom_{key}"):
        c1, c2, c3 = st.columns(3)
        position = c1.selectbox("Position", bundle["categories"]["position"])
        subs = sorted(df.loc[df.position == position, "sub_position"].dropna().unique())
        sub_position = c2.selectbox("Sub-position", subs)
        league = c3.selectbox("League", list(lm), format_func=lambda k: lm[k])
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 15.0, 45.0, 24.0, 0.5, help="Years at valuation date")
        height = c2.number_input("Height (cm)", 150, 210, 182)
        foot = c3.selectbox("Preferred foot", bundle["categories"]["foot"])
        st.markdown("**Last 365 days (all competitions)**")
        c1, c2, c3, c4 = st.columns(4)
        minutes = c1.number_input("Minutes played", int(C.MIN_WINDOW_MINUTES), 5900, 2000,
                                  help=f"Model requires at least {C.MIN_WINDOW_MINUTES} minutes")
        apps = c2.number_input("Appearances", 1, 70, 28)
        apps60 = c3.number_input("Matches with 60+ min", 0, 70, 22)
        euro = c4.slider("Share of minutes in European club cups", 0.0, 1.0, 0.0, 0.05)
        c1, c2, c3, c4 = st.columns(4)
        goals = c1.number_input("Goals", 0, 70, 5)
        assists = c2.number_input("Assists", 0, 50, 3)
        yellow = c3.number_input("Yellow cards", 0, 30, 3)
        red = c4.number_input("Red cards", 0, 5, 0)
        st.markdown("**Club context (last 365 days)**")
        c1, c2, c3 = st.columns(3)
        ppg = c1.slider("Club points per game", 0.0, 3.0, 1.4, 0.05)
        gd = c2.slider("Club goal difference per game", -4.0, 4.0, 0.0, 0.1)
        avail = c3.slider("Share of club minutes available", 0.0, 1.0,
                          float(min(1.0, 2000 / (38 * 90))), 0.01)
        go_btn = st.form_submit_button("Predict market value", type="primary")
    if not go_btn:
        return
    inp = dict(position=position, sub_position=sub_position, league=league, age=age,
               height_cm=height, foot=foot, minutes=minutes, appearances=apps,
               apps_60plus=apps60, euro_minutes_share=euro, goals=goals, assists=assists,
               yellow=yellow, red=red, club_ppg=ppg, club_gd_pg=gd, availability=avail)
    errors, warns = validate_custom_input(inp, bundle)
    if errors:
        for e in errors:
            st.error(e)
        return
    res = predict_custom(inp, bundle)
    badge("pred")
    metrics_row([("Predicted value", format_eur_m(res["predicted_value"]), "Model estimate, not a transfer fee"),
                 ("80% interval", f"{format_eur_m(res['pred_lo80'])} – {format_eur_m(res['pred_hi80'])}", None),
                 ("90% interval", f"{format_eur_m(res['pred_lo90'])} – {format_eur_m(res['pred_hi90'])}", None)])
    cov = bundle["conformal"]
    note(f"Intervals are split-conformal ranges calibrated on 2022–23 validation errors. On the held-out "
         f"2024–25 test set they actually covered {cov['empirical_test_80']:.0%} (nominal 80%) and "
         f"{cov['empirical_test_90']:.0%} (nominal 90%) of observed values, and they are wide "
         f"(median upper/lower ratio ≈ {cov['median_width_ratio_90']:.0f}× at 90%). They are not guarantees. "
         f"Hypothetical inputs are valued as if observed in the latest season "
         f"(market index {np.exp(bundle['latest_market_index']) / 1e6:.2f}m).")
    for w in warns:
        st.warning(w)
    contrib = local_explanation(res["row"])
    if contrib is not None:
        st.plotly_chart(contribution_chart(contrib), width='stretch')
        st.caption("Contributions describe the model's behaviour (associations), not causes.")
    if show_similar:
        pool = df[(df.season == df.season.max())]
        row = res["row"].iloc[0].copy()
        row["player_id"] = -1
        row["market_value_in_eur"] = np.nan
        sim = find_similar(pool, row, k=8)
        if not sim.empty:
            st.markdown("**Most statistically similar players (latest season)**")
            st.dataframe(sim[["player_name", "club_name", "age", "market_value_in_eur", "similarity"]]
                         .assign(market_value_in_eur=lambda d: d.market_value_in_eur.map(format_eur_m),
                                 age=lambda d: d.age.round(1), similarity=lambda d: d.similarity.round(3))
                         .rename(columns={"market_value_in_eur": "observed value"}),
                         hide_index=True, width='stretch')
