import streamlit as st
from stock_utils import fetch_stock_info, stock_name

st.title('Stocks Predictor')
my_symbol = st.text_input('Enter a stock ticker', '^GSPC', key='predictor') # S&P 500 is default

if my_symbol and my_symbol.strip():
    try:
        information = fetch_stock_info(my_symbol)
        stock_name(information,my_symbol)
    except Exception as e:
        st.error(f'Ticker not found')
else:
    st.info('Please enter a stock ticker symbol to get started')