import streamlit as st
import yfinance as yf

# Short desc for given company
@st.cache_data
def fetch_stock_info(symbol):
    stock = yf.Ticker(symbol)
    return stock.info

# Get the past quarter's worth of financial performance
@st.cache_data
def fetch_quarterly_finac(symbol):
    stock = yf.Ticker(symbol)
    return stock.quarterly_financials.T

# Get the annual financials
@st.cache_data
def fetch_annual_finac(symbol):
    stock = yf.Ticker(symbol)
    return stock.financials.T

# Get the weekly prices over an interval for a certain period
@st.cache_data
def fetch_weekly_price_history(symbol):
    stock = yf.Ticker(symbol)
    return stock.history(period='1y', interval='1wk')

def stock_name(info, symbol):
    st.header(f'{info["longName"]} ({symbol})')