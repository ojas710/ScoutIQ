import numpy as np
import streamlit as st

from src import ui
from src.formatting import format_eur_m

ui.setup("Potentially Undervalued / Overvalued Players",
         "Gap between observed market value and the model's out-of-sample estimate")
df = ui.get_data()
oos = df.dropna(subset=["predicted_value"]).copy()
ui.note("Residual = observed − predicted. <b>Negative</b>: observed value is below the model estimate "
        "(“potentially undervalued relative to model estimate”). <b>Positive</b>: above it. Only validation/test "
        "seasons (predictions from a model that never saw them) are used. This is a screening aid — not a "
        "guaranteed bargain list and not a transfer-fee estimate; the model omits injuries, contracts, "
        "technical quality and many other drivers of value.")
c1, c2, c3 = st.columns(3)
season = c1.selectbox("Season", sorted(oos.season.unique(), reverse=True))
pos = c2.multiselect("Position", sorted(oos.position.unique()), default=sorted(oos.position.unique()))
age = c3.slider("Age", 15, 45, (16, 40))
c1, c2, c3 = st.columns(3)
leagues = c1.multiselect("League", sorted(oos.league_label.dropna().unique()))
min90 = c2.slider("Minimum playing time (90s in 365d)", 5, 38, 15)
minval = c3.slider("Minimum observed value (€m)", 0.0, 20.0, 1.0, 0.5,
                   help="Very cheap players are systematically over-predicted (regression to the mean), "
                        "so they would otherwise dominate the 'undervalued' list.")
adjust = st.checkbox("Centre on the season's median residual", value=True,
                     help="Removes any season-wide bias so ranks reflect relative gaps among peers.")

d = oos[(oos.season == season) & oos.position.isin(pos) & oos.age.between(*age)
        & (oos.minutes_90s >= min90) & (oos.market_value_in_eur >= minval * 1e6)]
if leagues:
    d = d[d.league_label.isin(leagues)]
if len(d) < 10:
    st.warning("Too few players match these filters."); st.stop()
shift = d.log_residual.median() if adjust else 0.0
d = d.assign(gap=np.exp(d.log_residual - shift) - 1)
ui.badge("pred"); ui.badge("obs")
st.caption(f"{len(d):,} players · season-median log residual = {d.log_residual.median():+.2f}"
           f"{' (removed)' if adjust else ''}")

def table(x):
    t = x[["player_name", "club_name", "position", "age", "market_value_in_eur", "predicted_value", "gap"]].copy()
    t["age"] = t.age.round(1)
    t["observed"] = t.market_value_in_eur.map(format_eur_m); t["predicted"] = t.predicted_value.map(format_eur_m)
    t["observed vs model"] = (t.gap * 100).round(0).astype(int).astype(str) + "%"
    return t[["player_name", "club_name", "position", "age", "observed", "predicted", "observed vs model"]]

c1, c2 = st.columns(2)
c1.subheader("Observed below model estimate"); c1.dataframe(table(d.nsmallest(15, "gap")), hide_index=True)
c2.subheader("Observed above model estimate"); c2.dataframe(table(d.nlargest(15, "gap")), hide_index=True)
st.caption("The 90% typical error is wide; a −50% gap can lie within normal model error. Treat as a prompt for "
           "video/scouting review, not as evidence of mispricing.")
