import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.formatting import format_eur_m

ui.setup("Player Comparison", "Compare 2–4 players at the same valuation season")
df = ui.get_data()
season = st.sidebar.selectbox("Season", sorted(df.season.unique(), reverse=True))
cur = df[df.season == season].copy()
cur["label"] = cur.player_name + " (" + cur.club_name.fillna("?") + ")"
cur = cur.sort_values("market_value_in_eur", ascending=False)
picked = st.multiselect("Players (2–4)", cur.label.tolist(), max_selections=4,
                        placeholder="Search and add players")
if len(picked) < 2:
    st.info("Pick at least two players."); st.stop()
sel = cur[cur.label.isin(picked)].set_index("label").loc[picked].reset_index()

ui.badge("obs")
show = sel[["label", "position", "age", "minutes_90s", "market_value_in_eur", "predicted_value"]].copy()
show["observed"] = show.market_value_in_eur.map(format_eur_m)
show["predicted"] = show.predicted_value.map(format_eur_m)
show["age"] = show.age.round(1); show["90s played"] = show.minutes_90s.round(1)
st.dataframe(show[["label", "position", "age", "90s played", "observed", "predicted"]],
             hide_index=True, width='stretch')
if show.predicted_value.isna().any():
    st.caption("Predictions are shown only for out-of-sample seasons (2022+); 'n/a' otherwise.")

c1, c2 = st.columns(2)
fig = go.Figure()
fig.add_bar(x=sel.label, y=sel.market_value_in_eur / 1e6, name="Observed", marker_color=ui.ACCENT)
if sel.predicted_value.notna().any():
    fig.add_bar(x=sel.label, y=sel.predicted_value / 1e6, name="Predicted", marker_color="#f2c14e")
fig.update_layout(title="Market value (€m)", barmode="group")
c1.plotly_chart(ui.fig_style(fig, 340), width='stretch')
fig = px.scatter(sel, x="age", y="minutes_90s", text="label", size=np.sqrt(sel.market_value_in_eur / 1e6),
                 labels={"minutes_90s": "90s played (365d)"}, title="Age vs playing time (bubble = value)")
fig.update_traces(marker_color=ui.ACCENT, textposition="top center")
c2.plotly_chart(ui.fig_style(fig, 340), width='stretch')

st.subheader("Normalised performance radar")
positions = sel.position.unique()
if len(positions) > 1:
    ui.note("The selected players play in <b>different position groups</b>, so a performance radar would not be "
            "a like-for-like comparison. Select players of one position group (e.g. all midfielders) to enable it.")
else:
    pos = positions[0]
    metrics = {"goals_p90": "Goals/90", "assists_p90": "Assists/90", "minutes_90s": "Minutes",
               "availability": "Club-minutes share", "euro_minutes_share": "European minutes share",
               "club_ppg": "Club points/game"}
    if pos == "Goalkeeper":
        metrics = {k: v for k, v in metrics.items() if k not in ("goals_p90", "assists_p90")}
    pool = cur[cur.position == pos]
    ok = [m for m in metrics if sel[m].notna().all()]
    dropped = [metrics[m] for m in metrics if m not in ok]
    if dropped:
        st.caption("Omitted (missing for at least one player): " + ", ".join(dropped))
    if len(ok) >= 3:
        fig = go.Figure()
        for _, r in sel.iterrows():
            vals = [float((pool[m] <= r[m]).mean() * 100) for m in ok]
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[metrics[m] for m in ok] + [metrics[ok[0]]],
                                          fill="toself", name=r.label, opacity=.55))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                          title=f"Percentile among {pos} players, season {season} (n={len(pool)})")
        st.plotly_chart(ui.fig_style(fig, 460), width='stretch')
        st.caption("Percentiles are relative to all players of the same position in the dataset's eligible "
                   "population (≥450 minutes). Context metrics reflect the club, not only the player.")
    else:
        st.warning("Too many metrics are missing to draw a reliable radar.")
