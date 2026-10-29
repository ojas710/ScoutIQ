import streamlit as st

from src import ui
from src.formatting import format_eur_m
from src.player_similarity import features_for, find_similar

ui.setup("Similar Player Finder", "Nearest neighbours in standardised performance & context space")
df = ui.get_data()
season = st.sidebar.selectbox("Season (candidate pool)", sorted(df.season.unique(), reverse=True))
pool = df[df.season == season].copy()
pool["label"] = pool.player_name + " (" + pool.club_name.fillna("?") + ")"
choice = st.selectbox("Reference player", pool.sort_values("market_value_in_eur", ascending=False).label.tolist(),
                      index=None, placeholder="Search…")
with st.expander("Filters", expanded=True):
    c1, c2 = st.columns(2)
    age = c1.slider("Age range", 15, 45, (16, 40))
    vmax = float(pool.market_value_in_eur.max() / 1e6)
    val = c2.slider("Observed value range (€m)", 0.0, vmax, (0.0, vmax))
    leagues = st.multiselect("League", sorted(pool.league_label.dropna().unique()))
    clubs = st.multiselect("Club", sorted(pool.club_name.dropna().unique()))
    k = st.slider("Number of results", 3, 25, 10)
if choice is None:
    st.info("Choose a reference player."); st.stop()
ref = pool[pool.label == choice].iloc[0]
try:
    res = find_similar(pool, ref, k=k, filters={"age": age, "leagues": leagues, "clubs": clubs,
                                                "value": (val[0] * 1e6, val[1] * 1e6)})
except ValueError as e:
    st.error(str(e)); st.stop()
if res.empty:
    st.warning("No players match these filters. Widen the filters."); st.stop()
ui.badge("obs")
out = res[["player_name", "club_name", "league_label", "age", "market_value_in_eur", "similarity"]].copy()
out["age"] = out.age.round(1); out["market_value_in_eur"] = out.market_value_in_eur.map(format_eur_m)
out["similarity"] = out.similarity.round(3)
st.dataframe(out.rename(columns={"market_value_in_eur": "observed value", "similarity": "similarity index"}),
             hide_index=True, width='stretch')
ui.note(f"Position group: <b>{ref.position}</b> only (stats are not comparable across roles). Features used: "
        f"{', '.join(features_for(ref.position))}. The similarity index = 1/(1+distance) in standardised space; "
        "it is a descriptive closeness measure, <b>not</b> a probability that a player would be a successful "
        "replacement. Market value is only a filter, never a feature.")
