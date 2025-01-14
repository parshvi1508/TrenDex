import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import comparison
import history
import predictions
import tech_ind as ti

st.title("TrenDex: Stock Price Trend Predictor")

selected = option_menu(
    menu_title=None,
    options=["Home", "Compare", "Stock History", "Predictions", "Technical Indicators"],
    icons=["house", "bar-chart-line", "clock-history", "graph-up-arrow", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    ticker = st.text_input("Enter Stock Ticker (e.g., TSLA):", "TSLA")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    def fetch_stock_data(ticker, start_date, end_date):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def preprocess_data(data):
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)
        return scaled_prices, scaler

    def create_dataset(data, look_back=1):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back, 0])
            y.append(data[i + look_back, 0])
        return np.array(X), np.array(y)

    def train_model(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def make_predictions(model, X_test):
        predictions = model.predict(X_test)
        return predictions

    def evaluate_model(y_test, predictions):
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"R-squared (R²): {r2}")

    def predict_future_prices(model, last_data_point, days_to_predict=5):
        predictions = []
        current_data = last_data_point
        for _ in range(days_to_predict):
            next_prediction = model.predict([current_data])
            predictions.append(next_prediction[0])
            current_data = np.append(current_data[1:], next_prediction)
        return predictions

    def add_moving_averages(data, sma_period=20, ema_period=50):
        data['SMA'] = data['Close'].rolling(window=sma_period).mean()
        data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
        return data

    def plot_candlestick(data, sma_period=20, ema_period=50):
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                             name='Candlesticks'),
                              go.Scatter(x=data.index, y=data['SMA'], mode='lines', name=f'SMA {sma_period}'),
                              go.Scatter(x=data.index, y=data['EMA'], mode='lines', name=f'EMA {ema_period}')
                              ])

        fig.update_layout(title="Candlestick Chart with Moving Averages",
                          xaxis_title="Date",
                          yaxis_title="Price (USD)",
                          template="plotly_dark")

        st.plotly_chart(fig)

    if "data" not in st.session_state:
        st.session_state.data = None
    if "predictions_graph" not in st.session_state:
        st.session_state.predictions_graph = None
    if "future_predictions_graph" not in st.session_state:
        st.session_state.future_predictions_graph = None

    if st.button("Fetch and Predict"):
        st.write("Fetching data and making predictions...")
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None:
            st.session_state.data = data.copy()
            st.write("Stock Data:", data.head())
            scaled_data, scaler = preprocess_data(data)
            look_back = 15
            X, y = create_dataset(scaled_data, look_back)
            X = X.reshape(X.shape[0], look_back)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = train_model(X_train, y_train)
            predictions = make_predictions(model, X_test)
            evaluate_model(y_test, predictions)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_test, label='Actual Prices', color='blue')
            ax.plot(predictions, label='Predicted Prices', color='red')
            ax.set_title(f'{ticker} Stock Price Prediction')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True)
            st.session_state.predictions_graph = fig

            future_predictions = predict_future_prices(model, X[-1], days_to_predict=5)
            future_dates = pd.date_range(start=data.index[-1], periods=6)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(future_dates[1:], future_predictions, label='Future Predictions', color='green')
            ax.set_title(f'{ticker} Future Stock Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True)
            st.session_state.future_predictions_graph = fig

    if st.session_state.predictions_graph is not None:
        st.pyplot(st.session_state.predictions_graph)

    if st.session_state.future_predictions_graph is not None:
        st.pyplot(st.session_state.future_predictions_graph)

    if st.session_state.data is not None:
        sma_period = st.slider("Simple Moving Average Period", 5, 50, st.session_state.get("sma_period", 20), key="sma_slider")
        ema_period = st.slider("Exponential Moving Average Period", 5, 50, st.session_state.get("ema_period", 50), key="ema_slider")
        st.session_state.sma_period = sma_period
        st.session_state.ema_period = ema_period

        data_with_moving_averages = add_moving_averages(st.session_state.data.copy(), sma_period, ema_period)
        plot_candlestick(data_with_moving_averages, sma_period, ema_period)

elif selected == "Compare":
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    comparison.render(start_date, end_date)

elif selected == "Stock History":
    
    history.render_stock_history()

elif selected == "Predictions":
    predictions.render_predictions()

elif selected == "Technical Indicators":
    ti.render_technical_indicators()