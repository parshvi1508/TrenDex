import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_sma(data, window=14):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window=14):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def render_technical_indicators():
    st.title("Technical Indicators")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2021-01-01"))
    
    if st.button("Fetch Data"):
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            sma = calculate_sma(data)
            ema = calculate_ema(data)
            rsi = calculate_rsi(data)
            
            st.subheader("Stock Price")
            st.line_chart(data['Close'], use_container_width=True)
            
            st.subheader("Simple Moving Average (SMA)")
            st.line_chart(sma, use_container_width=True)
            
            st.subheader("Exponential Moving Average (EMA)")
            st.line_chart(ema, use_container_width=True)
            
            st.subheader("Relative Strength Index (RSI)")
            st.line_chart(rsi, use_container_width=True)
        else:
            st.error("No data found for the given ticker and date range.")
