import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.formatting import format_eur_m

ui.setup("Player Valuation", "Look up a historical player-season record and see what the model estimates")
df = ui.get_data()
opts = ui.player_options(df)
label = opts.player_name + " — " + opts.club_name.fillna("?")
choice = st.selectbox("Search player", label.tolist(), index=None, placeholder="Type a name…")
tab1, tab2 = st.tabs(["Historical record", "Custom hypothetical player"])

with tab1:
    if choice is None:
        st.info("Select a player to begin. Sorted by peak observed value; type to search.")
    else:
        pid = int(opts.loc[label == choice, "player_id"].iloc[0])
        recs = df[df.player_id == pid].sort_values("season", ascending=False)
        season = st.selectbox("Season record", recs.season.tolist(),
                              format_func=lambda s: f"{s}/{str(s + 1)[-2:]}")
        r = recs[recs.season == season].iloc[0]
        st.subheader(r.player_name)
        st.caption(f"{r.club_name} · {r.league_label} · {r.position} ({r.sub_position}) · "
                   f"age {r.age:.1f} · valuation date {r.valuation_date:%d %b %Y}")
        ui.badge("obs")
        has_pred = not (r.predicted_value != r.predicted_value)
        items = [("Observed market value", format_eur_m(r.market_value_in_eur), "Transfermarkt estimate, not a fee")]
        if has_pred:
            items += [("Model prediction", format_eur_m(r.predicted_value), "Model trained on 2014–21 only"),
                      ("Observed − predicted", format_eur_m(r.residual), "Negative: observed below model estimate")]
        ui.metrics_row(items)
        if has_pred:
            st.caption(f"Approximate 90% interval for the prediction: {format_eur_m(r.pred_lo90)} – {format_eur_m(r.pred_hi90)}")
        if has_pred:
            ui.badge("pred")
            ui.note("Individual predictions are uncertain: the typical absolute error on held-out seasons is "
                    "~40–50% of the value. The 90% interval is wide and only approximately calibrated.")
            contrib = ui.local_explanation(recs[recs.season == season])
            if contrib is not None:
                st.plotly_chart(ui.contribution_chart(contrib), width='stretch')
        else:
            ui.note(f"Season {season} belongs to the model's <b>training</b> period, so a prediction would be "
                    "in-sample and is intentionally hidden. Pick a 2022+ record for an honest comparison.")
        st.markdown("**Performance in the 365 days before the valuation date** (all competitions, per Transfermarkt)")
        stats = {"Minutes": f"{r.w_minutes:,.0f}", "Appearances": f"{r.appearances:.0f}",
                 "Goals": f"{r.w_goals:.0f}", "Assists": f"{r.w_assists:.0f}",
                 "Goals/90": f"{r.goals_p90:.2f}", "Assists/90": f"{r.assists_p90:.2f}",
                 "Club pts/game": f"{r.club_ppg:.2f}" if r.club_ppg == r.club_ppg else "n/a"}
        st.table(stats)
        hist = recs.sort_values("season")
        fig = go.Figure(go.Scatter(x=hist.season, y=hist.market_value_in_eur / 1e6, mode="lines+markers",
                                   name="Observed", line_color=ui.ACCENT))
        p = hist.dropna(subset=["predicted_value"])
        if len(p):
            fig.add_trace(go.Scatter(x=p.season, y=p.predicted_value / 1e6, mode="markers", name="Predicted (out-of-sample)",
                                     marker=dict(symbol="diamond", size=10, color="#f2c14e")))
        fig.update_layout(title="Observed valuation history (season-end) vs predictions", yaxis_title="€m")
        st.plotly_chart(ui.fig_style(fig, 340), width='stretch')

with tab2:
    st.caption("Enter a hypothetical profile. Inputs are validated against realistic limits.")
    ui.custom_form("val", df, show_similar=False)
