import streamlit as st
# import yfinance as yf
import altair as alt
import plotly.graph_objects as go
from stock_utils import fetch_stock_info, stock_name, fetch_annual_finac, fetch_quarterly_finac, fetch_weekly_price_history, calculate_price_change, fetch_daily_price, fetch_weekly_price, fetch_annual_price

st.title('Stock Dashboard')
my_symbol = st.text_input('Enter a stock ticker', '^GSPC', key='dash') # S&P 500 is default


# Put some of this into a function so that it can be reused in stock_predictor - done
if my_symbol and my_symbol.strip():
    try:
        information = fetch_stock_info(my_symbol)
        stock_name(information,my_symbol)
        changes = calculate_price_change(information)
        # st.write(information)
        if changes['change_colour'] == 'r':
            colour = 'red'
            st.subheader(f'{information['regularMarketPrice']} {information['currency']} :red[{changes['change_num']:.2f} ({changes['change_percent']})]')
        elif changes['change_colour'] == 'g':
            colour = 'green'
            st.subheader(f'{information['regularMarketPrice']} {information['currency']} :green[+{changes['change_num']:.2f} (+{changes['change_percent']})]')
        else:
            colour = 'white'
            st.subheader(f'{information['regularMarketPrice']} {information['currency']} :white[{changes['change_num']:.2f} ({changes['change_percent']})]')

        if information.get("marketCap") or information.get("sector")!= None:
            with st.expander('Show more'):        
                st.subheader(f'Market Cap: ${information["marketCap"]:,.0f}')
                st.subheader(f'Sector: {information.get("sector")}')
                st.write(f'{information.get("longBusinessSummary")}')

        # Daily
        daily_history = fetch_daily_price(my_symbol)
        daily_history = daily_history.rename_axis('Data').reset_index()
        daily_line_chart = go.Figure(data=[go.Scatter(x=daily_history['Data'],
                                                y=daily_history['Open'],
                                                mode='lines',
                                                line=dict(
                                                    color= colour
                                                ))])

        # Weekly
        weekly_history = fetch_weekly_price(my_symbol)
        #st.write(weekly_history)
        weekly_history = weekly_history.rename_axis('Data').reset_index()
        weekly_line_chart = go.Figure(data=[go.Scatter(x=weekly_history['Data'],
                                                       y=weekly_history['Open'],
                                                       mode='lines',
                                                       line=dict(
                                                           color=colour
                                                       ))])
        weekly_line_chart.update_xaxes(
            rangebreaks=[
                dict(bounds=[17, 9.5], pattern="hour"),
                dict(bounds=["sat", "mon"])

            ]
        )

        # Annual
        annual_history = fetch_annual_price(my_symbol)
        annual_history = annual_history.rename_axis('Data').reset_index()
        annual_line_chart=go.Figure(data=[go.Scatter(x=annual_history ['Data'],
                                                       y=annual_history['Open'],
                                                       mode='lines',
                                                       line=dict(
                                                           color=colour
                                                       ))])
        annual_line_chart.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat","mon"])
            ]
        )

        # Selection over time periods
        # uodate line colour to represent trend of growth for each period
        history_selection = st.segmented_control(label='History',options=['1D', '1W', '1Y'], default='1D')
        if history_selection == '1D':
            st.plotly_chart(daily_line_chart, use_container_width=True)
        elif history_selection == '1W':
            st.plotly_chart(weekly_line_chart, use_container_width=True)
        elif history_selection == '1Y':
            st.plotly_chart(annual_line_chart, use_container_width=True)

        price_history = fetch_weekly_price_history(my_symbol)
        price_history = price_history.rename_axis('Data').reset_index()
        st.header('Weekly Changes')
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
                quart_chart_data = quarterly_financials.melt(
                    id_vars=['Quarter'], value_vars=['Total Revenue', 'Net Income'],
                    var_name='Metric', value_name='Amount'
                )
                combined_quart_chart = alt.Chart(quart_chart_data).mark_bar().encode(
                    x=alt.X('Quarter:O', title='Quarter'), y=alt.Y('Amount', title=f'Amount {information['currency']}'),
                    color=alt.Color('Metric:N', scale=alt.Scale(domain=['Total Revenue', 'Net Income'], range=['red','orange'])),
                    xOffset='Metric:N',
                    tooltip=[
                        alt.Tooltip('Quarter:O', title='Quarter'),
                        alt.Tooltip('Metric:N', title='Metric'),
                        alt.Tooltip('Amount:Q', title='Amount', format=',')
                    ]
                    ).properties(title='Revenue vs Net Income')
                st.altair_chart(combined_quart_chart, use_container_width=True)

                
            elif selection == 'Annual':
                annual_financials = annual_financials.rename_axis('Year').reset_index()
                annual_financials['Year'] = annual_financials['Year'].astype(str).transform(lambda year: year.split('-')[0])
                annual_chart_data = annual_financials.melt(
                    id_vars=['Year'], value_vars=['Total Revenue', 'Net Income'],
                    var_name='Metric', value_name='Amount'
                )
                combined_annual_chart = alt.Chart(annual_chart_data).mark_bar().encode(
                    x=alt.X('Year:O', title='Year'), y=alt.Y('Amount', title=f'Amount {information['currency']}'),
                    color=alt.Color('Metric:N', scale=alt.Scale(domain=['Total Revenue', 'Net Income'], range=['red','orange'])),
                    xOffset='Metric:N',
                    tooltip=[
                        alt.Tooltip('Year:O', title='Year'),
                        alt.Tooltip('Metric:N', title='Metric'),
                        alt.Tooltip('Amount:Q', title='Amount', format=',')]
                    ).properties(title='Revenue vs Net Income')
                st.altair_chart(combined_annual_chart, use_container_width=True)

    except Exception as e:
        st.error(f' Error occured: {e}')
else:
    st.info('Please enter a stocks ticker to get started')

