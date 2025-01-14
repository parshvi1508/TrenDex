import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data available for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def preprocess_data(data):
    if len(data) < 1:
        return None, None
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    return scaled_prices, scaler

def create_dataset(data, look_back=1):
    if data is None or len(data) <= look_back:
        return np.array([]), np.array([])
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def train_model(X_train, y_train):
    if len(X_train) < 1 or len(y_train) < 1:
        return None
    model = LinearRegression()
    try:
        model.fit(X_train.reshape(-1, 1) if len(X_train.shape) == 1 else X_train, y_train)
        return model
    except ValueError:
        return None

def make_predictions(model, X_test):
    if model is None or len(X_test) < 1:
        return np.array([])
    try:
        predictions = model.predict(X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test)
        return predictions
    except ValueError:
        return np.array([])

def add_moving_averages(data, sma_period=20, ema_period=50):
    if len(data) < max(sma_period, ema_period):
        st.warning(f"Not enough data points for moving averages. Need at least {max(sma_period, ema_period)} points.")
        data['SMA'] = np.nan
        data['EMA'] = np.nan
        return data
    
    data['SMA'] = data['Close'].rolling(window=sma_period, min_periods=1).mean()
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False, min_periods=1).mean()
    return data

def plot_comparison_chart(data_list, tickers, sma_period, ema_period):
    if not data_list or not tickers:
        st.warning("No data available for comparison chart")
        return
        
    fig = go.Figure()
    for i, data in enumerate(data_list):
        if data is not None and not data.empty:
            fig.add_trace(go.Candlestick(x=data.index,
                                         open=data['Open'], high=data['High'], 
                                         low=data['Low'], close=data['Close'],
                                         name=f'{tickers[i]} Candlestick'))
            if 'SMA' in data.columns and 'EMA' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], 
                                       mode='lines', 
                                       name=f'{tickers[i]} SMA {sma_period}'))
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], 
                                       mode='lines', 
                                       name=f'{tickers[i]} EMA {ema_period}'))

    fig.update_layout(title=f"Stock Comparison: {', '.join(tickers)}",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template="plotly_dark")
    
    st.plotly_chart(fig)

def predict_future_prices(model, last_data_point, days_to_predict=5):
    if model is None or len(last_data_point) < 1:
        return []
    
    predictions = []
    current_data = last_data_point.copy()
    
    try:
        for _ in range(days_to_predict):
            next_prediction = model.predict(current_data.reshape(1, -1))
            predictions.append(next_prediction[0])
            current_data = np.roll(current_data, -1)
            current_data[-1] = next_prediction[0]
        return predictions
    except ValueError:
        return []

def render(start_date, end_date):
    tickers_input = st.text_input("Enter Stock Tickers (comma separated, e.g., TSLA, AAPL):", "TSLA,AAPL")
    if not tickers_input.strip():
        st.warning("Please enter at least one stock ticker.")
        return
        
    tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]
    
    all_data = {}
    data_list = []
    look_back = 15
    
    for ticker in tickers:
        st.write(f"Fetching data for {ticker}...")
        data = fetch_stock_data(ticker, start_date, end_date)
        
        if data is not None and not data.empty:
            sma_period = st.slider(f"Simple Moving Average Period for {ticker}", 
                                 5, 50, 20, key=f"sma_slider_{ticker}")
            ema_period = st.slider(f"Exponential Moving Average Period for {ticker}", 
                                 5, 50, 50, key=f"ema_slider_{ticker}")
            
            data_with_moving_averages = add_moving_averages(data.copy(), sma_period, ema_period)
            all_data[ticker] = data
            data_list.append(data_with_moving_averages)
        else:
            st.error(f"Failed to fetch data for {ticker}")
            continue

    if not all_data:
        st.error("No valid data available for any ticker.")
        return
    
    
    plot_comparison_chart(data_list, tickers, sma_period, ema_period)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker, data in all_data.items():
        ax.plot(data.index, data['Close'], label=ticker)
    ax.set_title("Stock Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for idx, (ticker, data) in enumerate(all_data.items()):
        scaled_data, scaler = preprocess_data(data)
        if scaled_data is None:
            continue
            
        X, y = create_dataset(scaled_data, look_back)
        if len(X) < 2 or len(y) < 2:
            st.warning(f"Not enough data points for predictions for {ticker}")
            continue
            
        X = X.reshape(X.shape[0], look_back)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)
        if model is None:
            continue
            
        predictions = make_predictions(model, X_test)
        if len(predictions) > 0:
            ax.plot(data.index[-len(predictions):], predictions, 
                   label=f'{ticker} Predictions', 
                   color=colors[idx % len(colors)])
    
    
    ax.set_title("Stock Price Predictions Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc='upper left')
    ax.grid(True)
    st.pyplot(fig)

    future_predictions = {}
    for ticker, data in all_data.items():
        scaled_data, scaler = preprocess_data(data)
        if scaled_data is None or len(scaled_data) < look_back:
            continue
            
        last_data_point = scaled_data[-look_back:]
        model = train_model(scaled_data[:-1].reshape(-1, 1), scaled_data[1:].reshape(-1))
        
        if model is not None:
            future_preds = predict_future_prices(model, last_data_point)
            if future_preds:
                future_dates = pd.date_range(start=data.index[-1], periods=len(future_preds) + 1)[1:]
                future_predictions[ticker] = (future_dates, future_preds)

    if future_predictions:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, (ticker, (future_dates, future_preds)) in enumerate(future_predictions.items()):
            ax.plot(future_dates, future_preds, 
                   label=f'{ticker} Future Predictions', 
                   color=colors[idx % len(colors)])

        ax.set_title("Future Stock Price Predictions Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend(loc='upper left')
        ax.grid(True)
        st.pyplot(fig)

    