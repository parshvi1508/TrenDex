import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error(f"No data found for ticker {ticker}")
        return None
    return data

def preprocess_data(data):
    if data.isnull().values.any():
        st.warning("There are missing values in the data. Filling with forward fill method.")
        data.fillna(method='ffill', inplace=True)
    return data

def predict_stock_prices(data, days_to_predict):
    data['Date'] = pd.to_datetime(data.index)
    data['Date_Num'] = (data['Date'] - data['Date'][0]).dt.days
    X = np.array(data['Date_Num']).reshape(-1, 1)
    y = np.array(data['Close'])
    model = LinearRegression()
    model.fit(X, y)
    last_date = data['Date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    future_days = [(date - data['Date'][0]).days for date in future_dates]
    future_days = np.array(future_days).reshape(-1, 1)

    predicted_prices = model.predict(future_days)
    
    predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predicted_prices.flatten() 
    })
    predictions.set_index('Date', inplace=True)
    return predictions

def plot_stock_prices(data, predictions):
    fig, ax = plt.subplots(figsize=(10,6))
    
    ax.plot(data.index, data['Close'], label='Actual Price', color='blue')
    ax.plot(predictions.index, predictions['Predicted_Price'], label='Predicted Price', color='red', linestyle='--')
    
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    
    st.pyplot(fig)

def render_predictions():
    st.title("Simple Stock Price Prediction")

    ticker = st.text_input("Enter Stock Ticker (e.g., TSLA)", "AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2021-01-01"))

    prediction_days = st.slider("Number of Days to Predict", min_value=1, max_value=30, value=5)
    if st.button("Fetch Data"):
        data = fetch_stock_data(ticker, start_date, end_date)
        
        if data is not None:
            data = preprocess_data(data)
            predictions = predict_stock_prices(data, prediction_days)
            st.write(f"Predicted prices for the next {prediction_days} days:")
            st.write(predictions)
            plot_stock_prices(data, predictions)
            csv = predictions.to_csv().encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f'{ticker}_predictions.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    render_predictions()
