import streamlit as st
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def display_stock_table(data):
    st.write("Stock Data")
    st.dataframe(data)

def plot_stock_prices(data):
    st.line_chart(data['Close'])

def download_stock_data(data, ticker):
    csv = data.to_csv().encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'{ticker}_stock_data.csv',
        mime='text/csv',
    )

def render_stock_history():
    st.title("Stock History")

    ticker = st.text_input("Enter Stock Ticker (e.g., TSLA):", "TSLA")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    if st.button("Fetch Data"):
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]

            display_stock_table(filtered_data)

            plot_stock_prices(filtered_data)
            download_stock_data(filtered_data, ticker)
        else:
            st.error("No data found for the given ticker and date range.")