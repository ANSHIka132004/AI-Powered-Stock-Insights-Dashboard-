![Screenshot 2025-06-13 172356](https://github.com/user-attachments/assets/feb85e59-9d65-4af3-b639-ff018d15c06e)
![Screenshot 2025-06-13 172418](https://github.com/user-attachments/assets/8702649c-7653-491e-a732-a596a5410b15)

![Screenshot 2025-06-13 173859](https://github.com/user-attachments/assets/de89279c-2ce2-452f-a329-cf97cd034ca4)
![Screenshot 2025-06-13 173943](https://github.com/user-attachments/assets/a29ba650-fa81-4487-853f-9d67079c4a2c)
![Screenshot 2025-06-13 203124](https://github.com/user-attachments/assets/49ffd883-6228-45d9-9639-7a3cba043327)
# AI-Powered-Stock-Insights-Dashboard
# Core Technologies

# Backend Framework:🖥 

Flask (Python)

Flask-CORS (for handling cross-origin requests)

# Data Sources & APIs:

Yahoo Finance (yfinance) for stock price and indicator data

Alpha Vantage API (for resolving stock symbols)

Google Custom Search API (for fetching news articles)

# Data Processing & Visualization:🧹

Pandas (data manipulation)

Matplotlib & Plotly (data visualization)

SQLite (lightweight database for caching & persistence)

# ML Integration:

Uses a custom trained ML classifier (stock_classifier.pkl) to predict stock movement based on RSI (Relative Strength Index).

Model loaded with joblib; integrated with error handling and logging.

Output is stored alongside other analysis results.

# NLP Components:

TextBlob: Performs sentiment analysis on news articles.

Gemini AI (LLM): Generates human-like recommendations and reasoning based on stock data and news sentiment.

News is fetched using the Google Custom Search API, summarized, analyzed, and passed to the LL

# Deployment & Configuration🚀 
.env setup using python-dotenv for managing API keys and secrets

Logging system (for debugging and monitoring)

Caching mechanism to avoid repeated API calls and improve performance
