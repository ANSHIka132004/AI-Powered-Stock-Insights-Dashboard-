from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import logging
import google.generativeai as genai
from textblob import TextBlob
import yfinance as yf  # For symbol resolution
import json
import streamlit as st
import threading
import time
import re
import joblib
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go

# Create a requests session with a custom User-Agent
session = requests.Session()
session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Database Initialization
DATABASE_FILE = 'user.db'

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            stock_symbol TEXT NOT NULL,
            rsi REAL,
            sentiment_score REAL,
            news_summary TEXT,
            llm_decision TEXT,
            llm_reasoning TEXT,
            ml_prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call init_db when the app starts
init_db()

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flask_app.log"),
        logging.StreamHandler()
    ]
)

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Load the pre-trained ML model
try:
    ml_model = joblib.load('stock_classifier.pkl')
    logging.info("Stock classifier model loaded successfully.")
except FileNotFoundError:
    ml_model = None
    logging.warning("stock_classifier.pkl not found. Please run train_model.py to train the model.")
except Exception as e:
    ml_model = None
    logging.error(f"Error loading stock classifier model: {e}")

# Add caching for stock data
stock_cache = {}
CACHE_DURATION = 300  # 5 minutes in seconds

def get_cached_stock_data(symbol):
    current_time = time.time()
    if symbol in stock_cache:
        cache_time, cache_data = stock_cache[symbol]
        if current_time - cache_time < CACHE_DURATION:
            return cache_data
    return None

def cache_stock_data(symbol, data):
    stock_cache[symbol] = (time.time(), data)

# Function to fetch stock symbol from Yahoo Finance
def fetch_stock_symbol(company_name):
    try:
        ticker = yf.Ticker(company_name, session=session)
        if ticker.info and "symbol" in ticker.info:
            return ticker.info["symbol"]
        return None
    except Exception as e:
        logging.error(f"Error resolving stock symbol with yfinance: {e}")
        return None

# Function to fetch stock symbol from Alpha Vantage
def fetch_stock_symbol_alpha_vantage(company_name):
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        print("Alpha Vantage raw response:", response.text)
        response.raise_for_status()
        data = response.json()
        best_match = data.get("bestMatches", [])
        if best_match:
            return best_match[0]["1. symbol"]  # Return the first matching symbol
        return None
    except requests.RequestException as e:
        logging.error(f"Error resolving stock symbol with Alpha Vantage: {e}")
        return None

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(company_name):
    try:
        # Check cache first
        cached_data = get_cached_stock_data(company_name)
        if cached_data:
            return cached_data

        ticker = yf.Ticker(company_name, session=session)
        if not ticker.info:
            return None, None
        
        # Get historical data with a delay to avoid rate limiting
        time.sleep(1)  # Add delay between requests
        hist = ticker.history(period="1mo", interval="5m")
        logging.info(f"Fetched historical data for {company_name}: {hist.shape[0]} rows")

        if hist.empty:
            logging.warning(f"Historical data is empty for {company_name}")
            return ticker.info.get("symbol"), None
            
        result = (ticker.info.get("symbol"), hist)
        cache_stock_data(company_name, result)
        return result
    except Exception as e:
        logging.error(f"Error fetching stock data with yfinance: {e}")
        return None, None

# Function to fetch current stock price
def fetch_current_price(stock_symbol):
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check cache first
            cached_data = get_cached_stock_data(stock_symbol)
            if cached_data:
                symbol, hist = cached_data
                

            ticker = yf.Ticker(stock_symbol, session=session)
            time.sleep(retry_delay)  # Add delay between requests
            
            # Try to get price from info first
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                return info['regularMarketPrice']
            
            # If info fails, try to get from history
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return hist['Close'].iloc[-1]
                
            return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            st.sidebar.error(f"Error fetching current price for {stock_symbol}: {str(e)}")
            return None

# Function to calculate RSI
def calculate_rsi(stock_data):
    if stock_data.empty:
        return 0
    closes = stock_data['Close'].tolist()
    gain, loss = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gain.append(delta)
            loss.append(0)
        else:
            gain.append(0)
            loss.append(abs(delta))
    avg_gain = sum(gain) / len(gain)
    avg_loss = sum(loss) / len(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to fetch news
def fetch_news(stock):
    API_KEY = os.getenv("GOOGLE_API_KEY")
    CX = os.getenv("SEARCH_ENGINE_ID")
    query = f"{stock} stock news"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Log the API response for debugging
        logging.info(f"Google Search API Response: {data}")
        
        items = data.get("items", [])
        if not items:
            logging.warning(f"No news items found for {stock}")
            return []
            
        news = [item["title"] + " - " + item["snippet"] for item in items[:5]]
        logging.info(f"Fetched {len(news)} news items for {stock}")
        return news
    except requests.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in fetch_news: {e}")
        return []

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        if not text:
            logging.warning("Empty text provided for sentiment analysis")
            return {"compound": 0}
            
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        logging.info(f"Sentiment analysis result: {sentiment_score}")
        return {"compound": sentiment_score}
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return {"compound": 0}

# Function to query LLM
def query_llm(stock, rsi, sentiment, news_summary):
    prompt = f"""
    Stock: {stock}
    RSI: {rsi}
    Sentiment Score: {sentiment['compound']}
    News Summary: {news_summary}

    Based on the provided information, provide a clear stock recommendation (Buy, Sell, or Hold) and then provide a detailed explanation for your reasoning.
    Format your response strictly as follows:

    Recommendation: [Buy/Sell/Hold]

    Reasoning:
    [Your detailed explanation here]
    """
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        
        # Parse the LLM's response
        response_text = response.text
        decision_match = re.search(r"Recommendation: (.*)", response_text)
        reasoning_match = re.search(r"Reasoning:\n(.*)", response_text, re.DOTALL)
        
        decision = decision_match.group(1).strip() if decision_match else "No recommendation found"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No explanation available"
        
        return {"decision": decision, "explanation": reasoning}
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        return {"error": "Failed to query LLM"}

# Analyze endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        user_input = data.get("user_input", "")
        company_name_or_symbol = user_input.split()[-1]

        # Resolve stock symbol
        stock_symbol = fetch_stock_symbol(company_name_or_symbol) or \
                       fetch_stock_symbol_alpha_vantage(company_name_or_symbol) or \
                       company_name_or_symbol

        if not stock_symbol:
            return jsonify({"error": "Unable to resolve stock symbol. Please check the company name or symbol."}), 400

        stock_data = get_cached_stock_data(company_name_or_symbol)
        if not stock_data:
            stock_symbol, stock_data = fetch_stock_data(company_name_or_symbol)
            if not stock_symbol:
                return jsonify({"error": f"No stock data found for {company_name_or_symbol}. Try again later."}), 404
        
        rsi = calculate_rsi(stock_data)
        news = fetch_news(company_name_or_symbol)
        sentiment = analyze_sentiment(" ".join(news))
        news_summary = " ".join(news)
        llm_response = query_llm(company_name_or_symbol, rsi, sentiment, news_summary)

        ml_prediction = "N/A"
        if ml_model and not pd.isna(rsi):
            try:
                # Reshape RSI for prediction if it's a single value
                ml_prediction = ml_model.predict([[rsi]])[0]
            except Exception as e:
                logging.error(f"Error predicting with ML model: {e}")

        # Prepare historical data for JSON response including OHLCV
        historical_prices = []
        if stock_data is not None and not stock_data.empty:
            # Ensure the index is a datetime index for proper conversion
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data.index = pd.to_datetime(stock_data.index)
            for index, row in stock_data.iterrows():
                historical_prices.append({
                    'Date': index.strftime('%Y-%m-%d'),
                    'Open': row.get('Open'),
                    'High': row.get('High'),
                    'Low': row.get('Low'),
                    'Close': row.get('Close'),
                    'Volume': row.get('Volume')
                })

        response = {
            "stock": company_name_or_symbol,
            "rsi": rsi,
            "sentiment": sentiment,
            "news_summary": news_summary,
            "llm_decision": llm_response.get("decision", "No recommendation"),
            "llm_reasoning": llm_response.get("explanation", "No explanation available"),
            "ml_prediction": ml_prediction,
            "historical_prices": historical_prices  # Add historical data to response
        }

        # Store the analysis result in the database
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analyses (stock_symbol, rsi, sentiment_score, news_summary, llm_decision, llm_reasoning, ml_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (response["stock"], response["rsi"], response["sentiment"]["compound"], 
             response["news_summary"], response["llm_decision"], 
             response["llm_reasoning"], response["ml_prediction"]))
            conn.commit()
            conn.close()
            logging.info(f"Analysis for {response['stock']} saved to database.")
        except Exception as db_e:
            logging.error(f"Error saving analysis to database: {db_e}")

        return jsonify(response)
    except Exception as e:
        import traceback
        print("Exception in /analyze:", e)
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route('/resolve-symbol', methods=['POST'])
def resolve_symbol():
    data = request.json
    company_name = data.get("company_name", "")
    symbol = fetch_stock_symbol_alpha_vantage(company_name)
    return jsonify({"symbol": symbol or ""})

def run_streamlit():
    # Streamlit app configuration
    st.set_page_config(page_title="AI Stock Market Agent", layout="wide")

    # Title and description
    st.title("AI-Powered Stock Market Analyst")
    st.markdown(
        """
        An agent for stock market analysis using AI. This application provides insights on stocks based on real-time market data, news sentiment, and advanced technical analysis.
        This app uses cutting-edge AI to provide insights on stocks.
        """
    )

    # Sidebar for current stock prices
    st.sidebar.title("Stock Prices")
    st.sidebar.markdown("### Current Prices")

    # List of stock symbols to display in the sidebar
    stock_symbols = ["TSLA", "AAPL", "GOOGL", "AMZN", "MSFT"]

    # Display current prices in the sidebar
    for symbol in stock_symbols:
        price = fetch_current_price(symbol)
        if price:
            st.sidebar.write(f"{symbol}: ${price:.2f}")

    # Input section for the user to specify the company name or stock symbol
    user_input = st.text_input("Enter Company Name or Stock Symbol (e.g., Tesla, TSLA, SUZLON):", "")

    # Analyze button
    if st.button("Analyze Stock"):
        if user_input.strip():
            st.info(f"Fetching analysis for: {user_input.title()}")

            # Call the Flask backend
            api_url = "http://localhost:5000/analyze"
            payload = {"user_input": user_input}

            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                data = response.json()

                # Display results
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.subheader(f"Results for {data['stock']}")
                    st.write(f"**RSI:** {data['rsi']:.2f}")
                    st.write(f"**Sentiment Score:** {data['sentiment']['compound']:.2f}")
                    st.write("**News Summary:**")
                    # Clean up news summary to remove extra newlines and format properly
                    cleaned_news_summary = data['news_summary'].replace('\n', ' ').replace('. ', '.\n\n').strip()
                    st.markdown(cleaned_news_summary)
                    st.write(f"**LLM Decision:** {data['llm_decision']}")
                    st.write(f"**LLM Reasoning:** {data['llm_reasoning']}")
                    st.write(f"**ML Prediction:** {data['ml_prediction']}")

                    # Display Historical Prices Graph
                    if data['historical_prices']:
                        st.subheader(f"Historical Prices for {data['stock']}")
                        historical_df = pd.DataFrame(data['historical_prices'])
                        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                        
                        # Create Candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=historical_df['Date'],
                            open=historical_df['Open'],
                            high=historical_df['High'],
                            low=historical_df['Low'],
                            close=historical_df['Close']
                        )])

                        fig.update_layout(
                            xaxis_rangeslider_visible=False,
                            title=f'{data["stock"]} Candlestick Chart',
                            yaxis_title='Price'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except requests.RequestException as e:
                st.error(f"Error communicating with the server: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False))
    flask_thread.daemon = True  # This ensures the thread will be killed when the main program exits
    flask_thread.start()
    
    # Run Streamlit
    run_streamlit()