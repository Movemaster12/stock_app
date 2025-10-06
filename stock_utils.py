import streamlit as st
import yfinance as yf

# Short desc for given company
@st.cache_data(ttl=60)
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

@st.cache_data(ttl=60)
def fetch_daily_price(symbol):
    stock = yf.Ticker(symbol)
    return stock.history(period='1d', interval='1m')

@st.cache_data(ttl=600)
def fetch_weekly_price(symbol):
    stock = yf.Ticker(symbol)
    return stock.history(period='7d', interval='30m')

@st.cache_data(ttl=3600)
def fetch_annual_price(symbol):
    stock = yf.Ticker(symbol)
    return stock.history(period='1y', interval='1d')

@st.cache_data
def stock_name(info, symbol):
    st.header(f'{info["longName"]} ({symbol})')

@st.cache_data(ttl=60)
def calculate_price_change(info):
    current_price = info.get('regularMarketPrice')
    previous_close = info.get('regularMarketPreviousClose')

    change_num = (current_price - previous_close)
    change_percent = f'{(change_num / previous_close )*100:.2f}%'
    if change_num > 0:
        change_colour = 'g'
    elif change_num < 0:
        change_colour = 'r'
    else:
        change_colour = 'w'

    change_dict = {
        'change_num': change_num,
        'change_percent': change_percent,
        'change_colour' : change_colour
    }

    return change_dict