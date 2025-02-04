# views.py
from django.shortcuts import render
from django.conf import settings
import yfinance as yf
import pandas as pd
from .utils import perform_analysis, perform_lstm_analysis, suggest_stock_to_buy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_nifty_components():
    """
    Fetches the current components of Nifty 50 index.
    Returns a list of stock symbols with .NS suffix.
    """
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        df = pd.read_csv(url)
        symbols = [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
        logger.info(f"Successfully fetched {len(symbols)} Nifty components")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching Nifty components: {str(e)}")
        return []

def get_stock_info(symbol):
    """
    Fetches detailed stock information using yfinance.
    Returns a dictionary with stock details.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'symbol': symbol,
            'volume': info.get('regularMarketVolume', 0),
            'market_cap': info.get('marketCap', 0),
            'price': info.get('regularMarketPrice', 0),
            'pe_ratio': info.get('forwardPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'name': info.get('longName', symbol.replace('.NS', ''))
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_trending_symbols(max_stocks=10):
    """
    Gets top Indian stocks using market data.
    Returns list of symbols with .NS suffix sorted by trading volume.
    """
    try:
        nifty_symbols = get_nifty_components()
        
        if not nifty_symbols:
            logger.error("Failed to fetch Nifty components")
            return [], []
        
        stock_data = []
        for symbol in nifty_symbols:
            stock_info = get_stock_info(symbol)
            if stock_info and stock_info['volume'] > 0:
                stock_data.append(stock_info)
        
        if not stock_data:
            logger.error("No valid stock data retrieved")
            return [], []
        
        stock_data.sort(key=lambda x: x['volume'], reverse=True)
        top_stocks = stock_data[:max_stocks]
        symbols = [stock['symbol'] for stock in top_stocks]
        
        logger.info(f"Successfully fetched {len(symbols)} trending stocks")
        return symbols, top_stocks

    except Exception as e:
        logger.error(f"Error in get_trending_symbols: {str(e)}")
        return [], []

def index(request):
    """
    Main view function for the stock analysis application.
    """
    trending_symbols, stock_details = get_trending_symbols()
    
    if not trending_symbols:
        context = {
            "error": "Unable to fetch stock data. Please try again later.",
            "trending_symbols": [],
            "stock_details": []
        }
        return render(request, "analysis/index.html", context)
    
    display_symbols = [symbol.replace('.NS', '') for symbol in trending_symbols]
    
    if request.method == "POST":
        ticker = request.POST.get("ticker")
        method = request.POST.get("method", "heuristic")
        
        if ticker:
            ticker_with_suffix = ticker if ticker.endswith('.NS') else f"{ticker}.NS"
            
            try:
                if method == "both":
                    heuristic_results, heuristic_graphs = perform_analysis(ticker_with_suffix)
                    lstm_results, lstm_graphs = perform_lstm_analysis(ticker_with_suffix)
                    
                    context = {
                        "trending_symbols": display_symbols,
                        "stock_details": stock_details,
                        "selected_method": "both",
                        "selected_stock": ticker,
                        "heuristic_results": heuristic_results,
                        "heuristic_graph_urls": {k: f"{settings.MEDIA_URL}analysis/{v}" for k, v in heuristic_graphs.items()},
                        "lstm_results": lstm_results,
                        "lstm_graph_urls": {k: f"{settings.MEDIA_URL}analysis/{v}" for k, v in lstm_graphs.items()},
                    }
                    return render(request, "analysis/report_both.html", context)
                
                elif method == "lstm":
                    results, graph_files = perform_lstm_analysis(ticker_with_suffix)
                    context = {
                        "results": results,
                        "graph_urls": {k: f"{settings.MEDIA_URL}analysis/{v}" for k, v in graph_files.items()},
                        "trending_symbols": display_symbols,
                        "stock_details": stock_details,
                        "selected_method": "lstm",
                        "selected_stock": ticker
                    }
                    return render(request, "analysis/report.html", context)
                
                else:
                    results, graph_files = perform_analysis(ticker_with_suffix)
                    context = {
                        "results": results,
                        "graph_urls": {k: f"{settings.MEDIA_URL}analysis/{v}" for k, v in graph_files.items()},
                        "trending_symbols": display_symbols,
                        "stock_details": stock_details,
                        "selected_method": "heuristic",
                        "selected_stock": ticker
                    }
                    return render(request, "analysis/report.html", context)
                    
            except Exception as e:
                logger.error(f"Error processing {ticker_with_suffix}: {str(e)}")
                context = {
                    "error": f"Error processing {ticker}: {str(e)}",
                    "trending_symbols": display_symbols,
                    "stock_details": stock_details
                }
                return render(request, "analysis/index.html", context)
    
    suggested_stock_data = suggest_stock_to_buy(trending_symbols) if trending_symbols else None
    
    if isinstance(suggested_stock_data, dict):
        suggested_symbol = suggested_stock_data.get('symbol', '').replace('.NS', '')
    else:
        suggested_symbol = ''
    
    context = {
        "trending_symbols": display_symbols,
        "stock_details": stock_details,
        "suggested_stock": suggested_symbol,
        "suggested_stock_data": suggested_stock_data,
    }
    return render(request, "analysis/index.html", context)