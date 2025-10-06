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
        st.subheader(f'{information['regularMarketPrice']} {information['currency']}')

        if information.get("marketCap") or information.get("sector")!= None:
            st.subheader(f'Market Cap: ${information["marketCap"]:,.0f}')
            st.subheader(f'Sector: {information.get("sector")}')

        price_history = fetch_weekly_price_history(my_symbol)

        st.header('Chart')
        price_history = price_history.rename_axis('Data').reset_index()
        candle_stick_chart = go.Figure(data=[go.Candlestick(x=price_history['Data'],
                                       open=price_history['Open'],
                                       low=price_history['Low'],
                                       high=price_history['High'],
                                       close=price_history['Close'])])

        st.plotly_chart(candle_stick_chart, use_container_width=True)

        if information.get("marketCap") or information.get("sector")!= None:
            quarterly_financials = fetch_quarterly_finac(my_symbol)
            annual_financials = fetch_annual_finac(my_symbol)

            st.header('Financials')
            selection =st.segmented_control(label='Period', options=['Quarterly', 'Annual'], default='Quarterly')
            
            if selection == 'Quarterly':
                quarterly_financials = quarterly_financials.rename_axis('Quarter').reset_index()
                quarterly_financials['Quarter'] = quarterly_financials['Quarter'].astype(str)
                revenue_chart = alt.Chart(quarterly_financials).mark_bar(color='red').encode(
                x='Quarter:O', y='Total Revenue')
                st.altair_chart(revenue_chart, use_container_width=True)

                net_income_chart = alt.Chart(quarterly_financials).mark_bar(color='orange').encode(
                x='Quarter:O', y='Net Income')
                st.altair_chart(net_income_chart, use_container_width=True)
            elif selection == 'Annual':
                annual_financials = annual_financials.rename_axis('Year').reset_index()
                annual_financials['Year'] = annual_financials['Year'].astype(str).transform(lambda year: year.split('-')[0])
                revenue_chart = alt.Chart(annual_financials).mark_bar(color='red').encode(
                x='Year:O', y='Total Revenue')
                st.altair_chart(revenue_chart, use_container_width=True)

                net_income_chart = alt.Chart(annual_financials).mark_bar(color='orange').encode(
                x='Year:O', y='Net Income')
                st.altair_chart(net_income_chart, use_container_width=True)

    except Exception as e:
        st.error(f' Error occured: {e}')
else:
    st.info('Please enter a stocks ticker to get started')

