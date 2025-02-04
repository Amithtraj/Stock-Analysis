# Stock Analysis Django Project

This project is a Django-based web application for advanced stock analysis. It provides two different analysis methods:

1. **Heuristic Analysis**: Combines technical indicators (RSI, MACD, Bollinger Bands), fundamental data, and sentiment analysis (using nltk's VADER and additional headlines via NewsAPI) to compute support/resistance levels, suggested buy/sell prices, and an overall recommendation.

2. **LSTM Analysis**: Uses a deep learning LSTM model (built with TensorFlow/Keras) to forecast the next day’s closing price based on historical data and provides a BUY/HOLD recommendation based on a 1% threshold. This method uses 10 years of historical data and trains for 20 epochs.

Additionally, users can choose to run both methods at once and compare the results.

## Features

- **Technical Analysis**:  
  - Computes technical indicators (RSI, MACD, Bollinger Bands) from historical data.
  - Calculates support levels, resistance levels, highest hit price, and suggested buy/sell prices.

- **Fundamental Analysis**:  
  - Retrieves fundamental data such as P/E ratio, EPS, and Market Cap (formatted in Indian crores).

- **Sentiment Analysis**:  
  - Uses nltk's VADER to compute sentiment scores from news headlines fetched from both yfinance and NewsAPI.
  - Aggregates headlines for a robust sentiment analysis.

- **LSTM Forecasting**:  
  - Implements an LSTM model to forecast the next day’s closing price.
  - Uses 10 years of historical data for training and predicts a BUY/HOLD signal based on a 1% threshold.

- **Modern UI**:  
  - A Bootstrap-based frontend provides a user-friendly interface.
  - Users can select the analysis method (Heuristic, LSTM, or Both) and view trending symbols.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/stock-analysis-django.git
   cd stock-analysis-django
**Create and Activate a Virtual Environment:**


  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate




pip install django yfinance tensorflow scikit-learn pandas numpy matplotlib ta nltk requests
**Set Up Environment Variables (Optional):**

For sentiment analysis, the application uses NewsAPI. Obtain a free API key from NewsAPI.org and set it in your environment:

On Windows (Command Prompt):


set NEWS_API_KEY=your_news_api_key
On Windows (PowerShell):

powershell

$env:NEWS_API_KEY="your_news_api_key"
On macOS/Linux:


export NEWS_API_KEY=your_news_api_key
Apply Migrations:  
  python manage.py migrate
Usage
Run the Django Development Server:


python manage.py runserver
Access the Application:

Open your web browser and navigate to http://127.0.0.1:8000/. On the home page, you can:

Enter a ticker symbol (e.g., TATAPOWER.NS).
Select the analysis method via radio buttons:
Heuristic Analysis
LSTM Analysis
Both Methods
View a list of trending symbols.
View the Report:

After submission, the application processes the request using the chosen method(s) and renders a detailed report that includes:

