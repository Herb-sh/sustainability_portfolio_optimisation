import numpy as np
import pandas as pd
import utilities.variables as variables
from pypfopt import expected_returns, EfficientFrontier

# Return & Volatility pro years
# 1, 5, 10, 25 year returns
# Set average yearly return of the last years (1, 5, 10, 25)
def set_yearly_return_rates_by_years(df_overview, df_monthly_return):
    # Loop through time spans
    for i, years in enumerate(variables.time_span_years):
        # Loop through tickers/stock name
        for j, ticker in enumerate(df_overview['stock_ticker_symbol']):
            # columns of df_monthly_return(returns dataframe) do not contain the suffix for stock_exchange,
            # for that reason we have to remove it from stock_ticker_symbol when comparing tickers with returns dataframe
            ticker_plain = ticker.split('.')[0]

            # if ticker is found in monthly adjacent columns, meaning there are available data to calculate
            if (ticker_plain in df_monthly_return.columns) or (ticker in df_monthly_return.columns):
                ticker_column = ticker if ticker in df_monthly_return.columns else ticker_plain
                # Get date "years" ago
                date = pd.Timestamp.today() - pd.DateOffset(years=years)
                # Pick only stocks that are after this date
                monthly_return_list = df_monthly_return.loc[ pd.to_datetime(df_monthly_return['Date']) >= date, ticker_column].dropna().tolist()
                if monthly_return_list and len(monthly_return_list) >= 2:
                    # Calculate the annualized average return
                    return_prod = np.prod(monthly_return_list) ** (1/years) if np.prod(monthly_return_list) > 0 else 0
                    annualized_return = return_prod # ** (1/years)
                    df_overview.loc[df_overview['stock_ticker_symbol'] == ticker, 'return_rate_' + str(years) + 'y_avg'] = annualized_return

def set_volatility_by_years(df_overview, df_monthly_adj_close):
    # 1, 5, 10, 25 year returns
    # Loop through time spans
    for i, years in enumerate(variables.time_span_years):
        for i, ticker in enumerate(df_overview['stock_ticker_symbol']):
            # columns of df_monthly_return(returns dataframe) do not contain the suffix for stock_exchange,
            # for that reason we have to remove it from stock_ticker_symbol when comparing tickers with returns dataframe
            ticker_plain = ticker.split('.')[0]
            if (ticker_plain in df_monthly_adj_close.columns) or (ticker in df_monthly_adj_close.columns):
                ticker_column = ticker if ticker in df_monthly_adj_close.columns else ticker_plain
                # Get date "years" ago
                date = pd.Timestamp.today() - pd.DateOffset(years=years)
                # Pick only stocks that are after this date
                adj_close_filtered = df_monthly_adj_close.loc[ pd.to_datetime(df_monthly_adj_close['Date']) >= date, ticker_column].dropna()

                if len(adj_close_filtered.tolist()) >= 2:
                    std_deviation = adj_close_filtered.pct_change().std()
                    df_overview.loc[df_overview['stock_ticker_symbol'] == ticker_column, 'volatility_' + str(years) + 'y'] = std_deviation
                else:
                    print(f"Ticket {ticker_column} is ignored.")

def evenly_spaced_sample(lst, n):
    """Returns `n` evenly spaced elements from a sorted list."""
    indices = np.linspace(0, len(lst) - 1, n, dtype=int)
    return [lst[i] for i in indices]

def evenly_spaced_dataframe(data, n):
    mu = expected_returns.mean_historical_return(data)

    # Sort tickers by expected return
    sorted_tickers = mu.sort_values().index.tolist()

    # Select evenly spread tickers

    # Choose x well-spread tickers
    sampled_tickers = evenly_spaced_sample(sorted_tickers, n)

    # Extract corresponding data from the original DataFrame
    return data[sampled_tickers]

def chunkify_df(df: pd.DataFrame, chunk_size: int):
    start = 0
    length = df.shape[0]

    # If DF is smaller than the chunk, return the DF
    if length <= chunk_size:
        yield df[:]
        return

    # Yield individual chunks
    while start + chunk_size <= length:
        yield df[start:chunk_size + start]
        start = start + chunk_size

    # Yield the remainder chunk, if needed
    if start < length:
        yield df[start:]

