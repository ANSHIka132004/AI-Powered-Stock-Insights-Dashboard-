# AI-Powered-Stock-Insights-Dashboard
# Core Technologies

# Backend Framework:🖥 

Flask (Python)

Flask-CORS (for handling cross-origin requests)
# AI/ML Components:🤖

Gemini AI (gemini-2.0-flash-exp) for natural language understanding

TextBlob for sentiment analysis

Custom ML model (stock_classifier.pkl) trained for stock predictions

Joblib (for saving/loading ML models)

# Data Sources & APIs:

Yahoo Finance (yfinance) for stock price and indicator data

Alpha Vantage API (for resolving stock symbols)

Google Custom Search API (for fetching news articles)

# Data Processing & Visualization:🧹

Pandas (data manipulation)

Matplotlib & Plotly (data visualization)

SQLite (lightweight database for caching & persistence)

# Deployment & Configuration🚀 
.env setup using python-dotenv for managing API keys and secrets

Logging system (for debugging and monitoring)

Caching mechanism to avoid repeated API calls and improve performance
