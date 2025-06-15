import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from alpha_vantage.timeseries import TimeSeries
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main():
    df = yf.download("MSFT", period="1mo", interval="1d")
    if df.empty:
        print("Failed to download data. Try again later or check your internet connection.")
        return
    df['RSI'] = calculate_rsi(df['Close'])
    # Compute next day's close minus today's close
    df['Change'] = df['Close'].shift(-1) - df['Close']
    # Label: Buy if change > 1, Sell if change < -1, else Hold
    conditions = [
        df['Change'] > 1,
        df['Change'] < -1
    ]
    choices = ['Buy', 'Sell']
    df['Label'] = np.select(conditions, choices, default='Hold')
    df = df.dropna()
    X = df[['RSI']]
    y = df['Label']
    if len(X) == 0:
        print("No data to train on after feature engineering.")
        return
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'stock_classifier.pkl')
    print('Model trained and saved as stock_classifier.pkl')

    ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
    data, meta_data = ts.get_daily(symbol='AAPL', outputsize='compact')
    print(data.head())

if __name__ == '__main__':
    main() 