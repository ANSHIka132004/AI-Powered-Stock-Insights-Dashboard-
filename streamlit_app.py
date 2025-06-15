import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os

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

# Flask backend URL (will be http://localhost:5000 for local development)
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://localhost:5000")

# Sidebar for current stock prices
st.sidebar.title("Stock Prices")
st.sidebar.markdown("### Current Prices")

# Function to fetch current stock price from backend
def fetch_current_price_from_backend(stock_symbol):
    try:
        response = requests.post(f"{FLASK_API_URL}/fetch-current-price", json={"stock_symbol": stock_symbol})
        response.raise_for_status()
        return response.json().get("price")
    except requests.RequestException as e:
        st.sidebar.error(f"Error fetching current price for {stock_symbol}: {e}")
        return None

# List of stock symbols to display in the sidebar
stock_symbols = ["TSLA", "AAPL", "GOOGL", "AMZN", "MSFT"]

# Display current prices in the sidebar
for symbol in stock_symbols:
    price = fetch_current_price_from_backend(symbol)
    if price:
        st.sidebar.write(f"{symbol}: ${price:.2f}")

# Input section for the user to specify the company name or stock symbol
user_input = st.text_input("Enter Company Name or Stock Symbol (e.g., Tesla, TSLA, SUZLON):", "")

# Analyze button
if st.button("Analyze Stock"):
    if user_input.strip():
        st.info(f"Fetching analysis for: {user_input.title()}")

        # Call the Flask backend's analyze endpoint
        api_url = f"{FLASK_API_URL}/analyze"
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