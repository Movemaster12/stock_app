import streamlit as st

# Multi-page Navigation
dashboard = st.Page("stock_dashboard.py", title="Stocks Dashboard")
predictor = st.Page("stock_predictor.py", title="Stocks Predictor")

pg = st.navigation([dashboard, predictor])
pg.run()
