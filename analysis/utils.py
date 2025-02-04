# analysis/utils.py
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon', quiet=True)
def get_additional_headlines(query, api_key, page_size=20):
    """
    Uses NewsAPI to fetch additional headlines related to the query.
    Returns a list of headlines.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
       "q": query,
       "pageSize": page_size,
       "sortBy": "publishedAt",
       "language": "en"
    }
    headers = {"Authorization": api_key}
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            headlines = [article.get("title", "") for article in articles if article.get("title", "")]
            return headlines
        else:
            return []
    except Exception as e:
        print("Error fetching additional headlines:", e)
        return []
def format_market_cap(cap):
    """
    Convert a market cap value (in rupees) to Indian format in crores.
    1 crore = 1e7.
    """
    if cap is None or np.isnan(cap):
        return "N/A"
    crores = cap / 1e7
    # Format with Indian style comma separation (you can further tweak this if desired)
    return f"₹{crores:,.2f} Cr"

def get_media_dir():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    media_dir = os.path.join(base_dir, "media", "analysis")
    os.makedirs(media_dir, exist_ok=True)
    return media_dir
# def get_media_dir():
#     """
#     Returns the directory (and creates it if necessary) to store output graphs.
#     We'll store graphs in <project_root>/media/analysis.
#     """
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     media_dir = os.path.join(base_dir, "media", "analysis")
#     os.makedirs(media_dir, exist_ok=True)
#     return media_dir

def get_historical_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="5y")
    return data

def compute_annualized_volatility(data):
    data['Daily Return'] = data['Close'].pct_change()
    daily_std = data['Daily Return'].std()
    volatility = daily_std * np.sqrt(252) * 100  # as a percentage
    return volatility

def compute_var(data, confidence=95):
    returns = data['Daily Return'].dropna() * 100
    var = np.percentile(returns, 100 - confidence)
    return var

def get_live_price(ticker_symbol):
    live_data = yf.download(ticker_symbol, period="1d", interval="1m", progress=False)
    if live_data.empty:
        return None
    return live_data['Close'].iloc[-1]

def get_fundamental_data(ticker_symbol):
    info = yf.Ticker(ticker_symbol).info
    pe_ratio = info.get('trailingPE', np.nan)
    eps = info.get('trailingEps', np.nan)
    market_cap = info.get('marketCap', np.nan)
    return pe_ratio, eps, market_cap


def get_news_sentiment(ticker_symbol):
    """
    Retrieve recent news headlines from yfinance and additional headlines from NewsAPI (if available),
    then compute and return the average compound sentiment score along with all headlines.
    """
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    analyzer = SentimentIntensityAnalyzer()
    compound_scores = []
    headlines = []
    # Headlines from yfinance
    for article in news:
        title = article.get('title', '')
        if title:
            headlines.append(title)
            score = analyzer.polarity_scores(title)['compound']
            compound_scores.append(score)
    # If a NewsAPI key is provided, fetch additional headlines.
    NEWS_API_KEY = "add your key"
    if NEWS_API_KEY:
        # Using the ticker symbol as query; alternatively, you could extract the company name from fundamentals.
        additional = get_additional_headlines(ticker_symbol, NEWS_API_KEY, page_size=20)
        headlines.extend(additional)
        for title in additional:
            score = analyzer.polarity_scores(title)['compound']
            compound_scores.append(score)
    avg_sentiment = np.mean(compound_scores) if compound_scores else 0
    return avg_sentiment, headlines


def compute_technical_indicators(data):
    data = data.copy()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    return data

def plot_price_with_bbands(data, ticker_symbol, media_dir):
    data_last_year = data.last("1y")
    plt.figure(figsize=(12,6))
    plt.plot(data_last_year.index, data_last_year['Close'], label='Close Price', color='blue')
    plt.plot(data_last_year.index, data_last_year['BB_High'], label='Bollinger High', linestyle='--', color='red')
    plt.plot(data_last_year.index, data_last_year['BB_Low'], label='Bollinger Low', linestyle='--', color='green')
    plt.title(f"{ticker_symbol} Price & Bollinger Bands (1y)")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.legend()
    filename = f"price_bb_{ticker_symbol}.png"
    filepath = os.path.join(media_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filename

def plot_rsi(data, ticker_symbol, media_dir):
    data_last_year = data.last("1y")
    plt.figure(figsize=(12,4))
    plt.plot(data_last_year.index, data_last_year['RSI'], label='RSI', color='purple')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.title(f"{ticker_symbol} RSI (1y)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    filename = f"rsi_{ticker_symbol}.png"
    filepath = os.path.join(media_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filename

def plot_macd(data, ticker_symbol, media_dir):
    data_last_year = data.last("1y")
    plt.figure(figsize=(12,4))
    plt.plot(data_last_year.index, data_last_year['MACD'], label='MACD', color='blue')
    plt.plot(data_last_year.index, data_last_year['MACD_Signal'], label='MACD Signal', color='orange')
    plt.title(f"{ticker_symbol} MACD (1y)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    filename = f"macd_{ticker_symbol}.png"
    filepath = os.path.join(media_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filename

# analysis/utils.py (only the updated perform_analysis function shown)
def perform_analysis(ticker_symbol):
    historical_data = get_historical_data(ticker_symbol)
    if historical_data.empty:
        return None, None
    data = compute_technical_indicators(historical_data)
    
    # Risk metrics
    volatility = compute_annualized_volatility(data)
    var_95 = compute_var(data, confidence=95)
    
    # Live price (or fallback to last historical close)
    live_price = get_live_price(ticker_symbol)
    if live_price is None:
        live_price = data['Close'].iloc[-1]
    
    # Fundamental data
    pe_ratio, eps, market_cap = get_fundamental_data(ticker_symbol)
    
    # Format market cap in Indian format
    market_cap_formatted = format_market_cap(market_cap)
    
    # News sentiment
    avg_sentiment, news_headlines = get_news_sentiment(ticker_symbol)
    
    # Technical indicators from latest data
    latest = data.iloc[-1]
    technical_score = 0
    if latest['RSI'] < 30:
        technical_score += 1
    elif latest['RSI'] > 70:
        technical_score -= 1
    if latest['MACD'] > latest['MACD_Signal']:
        technical_score += 1
    else:
        technical_score -= 1
    if live_price <= latest['BB_Low'] * 1.05:
        technical_score += 1
    elif live_price >= latest['BB_High'] * 0.95:
        technical_score -= 1

    # Fundamental score based on P/E ratio.
    fundamental_score = 0
    if not np.isnan(pe_ratio):
        if pe_ratio < 15:
            fundamental_score += 1
        elif pe_ratio > 25:
            fundamental_score -= 1

    overall_score = technical_score + fundamental_score + (avg_sentiment * 2)
    if overall_score >= 2:
        recommendation = "BUY"
    elif overall_score <= -2:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    # Compute Supported Breaking Point & Suggested Buy Price
    supported_breaking_point = latest['BB_Low']
    suggested_buy_price = supported_breaking_point * 1.02

    # Compute Resistance Level as the highest high over the last 20 days,
    # and suggest a sell price as 2% below that resistance.
    resistance_level = data.tail(20)['High'].max()
    suggested_sell_price = resistance_level * 0.98

    # Compute Highest Hit Price from the entire historical dataset.
    highest_hit_price = data['High'].max()
    
    results = {
        "ticker": ticker_symbol,
        "live_price": live_price,
        "volatility": volatility,
        "var_95": var_95,
        "pe_ratio": pe_ratio,
        "eps": eps,
        "market_cap": market_cap,  # raw value
        "market_cap_formatted": market_cap_formatted,
        "avg_sentiment": avg_sentiment,
        "technical_score": technical_score,
        "fundamental_score": fundamental_score,
        "overall_score": overall_score,
        "recommendation": recommendation,
        "latest_RSI": latest['RSI'],
        "latest_MACD": latest['MACD'],
        "latest_MACD_Signal": latest['MACD_Signal'],
        "BB_High": latest['BB_High'],
        "BB_Low": latest['BB_Low'],
        "news_headlines": news_headlines,
        "supported_breaking_point": supported_breaking_point,
        "suggested_buy_price": suggested_buy_price,
        "resistance_level": resistance_level,
        "suggested_sell_price": suggested_sell_price,
        "highest_hit_price": highest_hit_price,
    }
    
    media_dir = get_media_dir()
    graph_files = {
        "price_bb": plot_price_with_bbands(data, ticker_symbol, media_dir),
        "rsi": plot_rsi(data, ticker_symbol, media_dir),
        "macd": plot_macd(data, ticker_symbol, media_dir)
    }
    
    return results, graph_files
