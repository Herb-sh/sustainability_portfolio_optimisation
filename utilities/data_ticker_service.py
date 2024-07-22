import importlib
import utilities.variables as variables
import utilities.parser as parser
import time
import yfinance as yf
import pandas as pd
importlib.reload(parser)


def fetch_market_cap(tickers, batch_size=100, delay=2):
    market_caps = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        tickers_str = ' '.join(batch)

        try:
            data = yf.Tickers(tickers_str)
            for ticker in batch:
                info = data.tickers[ticker].info
                market_cap = info.get('marketCap', 'N/A')
                market_caps.append({'stock_ticker_symbol': ticker, 'market_capital': market_cap})
        except Exception as e:
            print(f"Error fetching data for batch starting at index {i}: {e}")

        time.sleep(delay)  # delay to avoid getting blocked

    return pd.DataFrame(market_caps)

def get_monthly_returns(tickers, start_date, end_date):
    monthly_returns = {}
    for ticker in tickers:
        # Download the stock data
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            continue

        # Resample to monthly data and calculate the monthly returns
        data['Monthly Return'] = data['Adj Close'].resample('ME').ffill().pct_change()
        monthly_returns[ticker] = data['Monthly Return']

    # Convert the dictionary to a DataFrame
    monthly_returns_df = pd.DataFrame(monthly_returns)
    return monthly_returns_df