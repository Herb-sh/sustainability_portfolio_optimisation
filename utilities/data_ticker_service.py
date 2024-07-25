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

def get_monthly_returns(tickers, start_date, end_date, interval='1mo'):
    monthly_returns = {}
    for ticker in tickers:
        # Download the stock data
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            continue
        # Resample to monthly data and calculate the monthly returns
        data['Monthly Return'] = (data['Adj Close'].pct_change() * 100).round(2).dropna()
        monthly_returns[ticker] = data['Monthly Return']

    # Convert the dictionary to a DataFrame
    monthly_returns_df = pd.DataFrame(monthly_returns)
    return monthly_returns_df

def get_returns_in_chunks(tickers, start_date, end_date, interval='1mo', chunk_size=5, sleep_duration=5):
    # Initialize an empty DataFrame to store the results
    all_data = pd.DataFrame()

    # Download data in chunks
    for chunk in chunks(tickers, chunk_size):
        print(f"Downloading data for tickers: {chunk}")

        # Download data for the current chunk
        data = get_monthly_returns(chunk, start_date, end_date, interval)
        # Append the downloaded data to the all_data DataFrame
        all_data = pd.concat([all_data, data], axis=1)

        # Pause to avoid overloading the API
        print("Pausing to avoid overloading the API...")
        time.sleep(sleep_duration)  # Adjust the sleep time as needed

    # Return the resulting DataFrame
    return all_data

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]