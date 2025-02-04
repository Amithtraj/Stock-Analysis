# analysis/views.py
from django.conf import settings
import os

from django.shortcuts import render

from analysis.utils import perform_analysis

def index(request):
    trending_symbols = [
        "TATAPOWER.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS",
        "HDFCBANK.NS", "AXISBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS"
    ]
    
    if request.method == "POST":
        ticker = request.POST.get("ticker")
        if ticker:
            results, graph_files = perform_analysis(ticker)
            if results is None:
                context = {"error": "No data available for this ticker.", "trending_symbols": trending_symbols}
                return render(request, "analysis/index.html", context)
            # Build full URLs for the graph images
            graph_urls = {}
            for key, filename in graph_files.items():
                # MEDIA_URL is usually '/media/' as defined in settings.py
                graph_urls[key] = os.path.join(settings.MEDIA_URL, "analysis", filename)
            context = {
                "results": results,
                "graph_urls": graph_urls,
                "trending_symbols": trending_symbols
            }
            return render(request, "analysis/report.html", context)
    return render(request, "analysis/index.html", {"trending_symbols": trending_symbols})
