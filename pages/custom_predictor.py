import streamlit as st

from src import ui

ui.setup("Custom Player Predictor", "Estimate a market value from a hypothetical performance profile")
df = ui.get_data()
ui.note("Inputs are limited to realistic ranges. Values outside the range seen in training are flagged because "
        "the model can only extrapolate unreliably. The output is a <b>market-value estimate</b>, not a transfer fee.")
ui.custom_form("custom", df, show_similar=True)
