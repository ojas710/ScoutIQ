"""ScoutIQ – Football Transfer Value Predictor & Scouting Intelligence Platform."""
import streamlit as st

st.set_page_config(page_title="ScoutIQ", page_icon="⚽", layout="wide")

pages = [
    st.Page("pages/dashboard.py", title="Executive Dashboard", icon="📊", default=True),
    st.Page("pages/player_valuation.py", title="Player Valuation", icon="🔎"),
    st.Page("pages/player_comparison.py", title="Player Comparison", icon="⚖️"),
    st.Page("pages/similar_players.py", title="Similar Player Finder", icon="🧭"),
    st.Page("pages/undervalued_players.py", title="Under/Over-valued Players", icon="📉"),
    st.Page("pages/model_performance.py", title="Model & Methodology", icon="🧪"),
    st.Page("pages/custom_predictor.py", title="Custom Player Predictor", icon="🛠️"),
]
st.navigation(pages).run()

with st.sidebar:
    st.markdown("### ⚽ ScoutIQ")
    st.caption("Market values are Transfermarkt estimates, not transfer fees. "
               "Predictions are model estimates with substantial uncertainty.")
