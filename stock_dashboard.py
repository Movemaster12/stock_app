import streamlit as st
import yfinance as yf
import altair as alt
import plotly.graph_objects as go
from stock_utils import fetch_stock_info, stock_name, fetch_annual_finac, fetch_quarterly_finac, fetch_weekly_price_history

st.title('Stock Dashboard')
my_symbol = st.text_input('Enter a stock ticker', '^GSPC', key='dash') # S&P 500 is default


# Put some of this into a function so that it can be reused in stock_predictor - done
if my_symbol and my_symbol.strip():
    try:
        information = fetch_stock_info(my_symbol)
        stock_name(information,my_symbol)
        # Put a collapsible element here ?
        st.subheader(f'Market Cap: ${information["marketCap"]:,.0f}' if information.get('marketCap') else 'Market Cap: N/A')
        st.subheader(f'Sector: {information.get("sector", 'N/A')}')
        price_history = fetch_weekly_price_history(my_symbol)

    except Exception as e:
        st.error(f'Ticker not found!')
else:
    st.info('Please enter a stock ticker symbol to get started')