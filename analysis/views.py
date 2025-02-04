from django.shortcuts import render
from django.conf import settings
from .utils import perform_analysis, perform_lstm_analysis

def index(request):
    # Define a static list of trending symbols (update this list or later replace with dynamic data)
    trending_symbols = [
        "TATAPOWER.NS",
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "AXISBANK.NS",
        "ICICIBANK.NS",
        "HINDUNILVR.NS"
    ]
    
    if request.method == "POST":
        ticker = request.POST.get("ticker")
        # Retrieve the selected method from the form; valid values: "heuristic", "lstm", or "both"
        method = request.POST.get("method", "heuristic")
        
        if ticker:
            if method == "both":
                # Call both analysis methods
                heuristic_results, heuristic_graphs = perform_analysis(ticker)
                lstm_results, lstm_graphs = perform_lstm_analysis(ticker)
                
                # Build full URLs for the graph images (assuming MEDIA_URL is defined in settings)
                heuristic_graph_urls = {}
                for key, filename in heuristic_graphs.items():
                    heuristic_graph_urls[key] = f"{settings.MEDIA_URL}analysis/{filename}"
                    
                lstm_graph_urls = {}
                for key, filename in lstm_graphs.items():
                    lstm_graph_urls[key] = f"{settings.MEDIA_URL}analysis/{filename}"
                    
                context = {
                    "trending_symbols": trending_symbols,
                    "selected_method": "both",
                    "heuristic_results": heuristic_results,
                    "heuristic_graph_urls": heuristic_graph_urls,
                    "lstm_results": lstm_results,
                    "lstm_graph_urls": lstm_graph_urls,
                }
                # Render a combined report template that shows both methods
                return render(request, "analysis/report_both.html", context)
            
            elif method == "lstm":
                # Use LSTM analysis method
                results, graph_files = perform_lstm_analysis(ticker)
                graph_urls = {}
                for key, filename in graph_files.items():
                    graph_urls[key] = f"{settings.MEDIA_URL}analysis/{filename}"
                context = {
                    "results": results,
                    "graph_urls": graph_urls,
                    "trending_symbols": trending_symbols,
                    "selected_method": "lstm",
                }
                return render(request, "analysis/report.html", context)
            
            else:
                # Default: use heuristic analysis method
                results, graph_files = perform_analysis(ticker)
                graph_urls = {}
                for key, filename in graph_files.items():
                    graph_urls[key] = f"{settings.MEDIA_URL}analysis/{filename}"
                context = {
                    "results": results,
                    "graph_urls": graph_urls,
                    "trending_symbols": trending_symbols,
                    "selected_method": "heuristic",
                }
                return render(request, "analysis/report.html", context)
    
    # For GET requests, simply render the index page with trending symbols.
    return render(request, "analysis/index.html", {"trending_symbols": trending_symbols})
