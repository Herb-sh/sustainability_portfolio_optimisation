import numpy as np
import pandas as pd
from scipy.optimize import minimize
import utilities.variables as variables
from pypfopt import EfficientFrontier

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

# Efficient-Frontier
def portfolio_performance(weights, returns, volatilities):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(volatilities), weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, returns, volatilities, risk_free_rate=0):
    p_return, p_volatility = portfolio_performance(weights, returns, volatilities)
    return -(p_return - risk_free_rate) / p_volatility

def minimize_volatility(weights, returns, volatilities):
    return portfolio_performance(weights, returns, volatilities)[1]

def efficient_frontier(df, line_point_nr=20):
    returns = df['return_rate_5y_avg'].values
    volatilities = df['volatility_5y'].values
    num_assets = len(returns)
    results = []
    target_returns = np.linspace(min(returns), max(returns), line_point_nr)
    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * returns) - target_return}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]

        result = minimize(minimize_volatility, initial_guess, args=(returns, volatilities),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        results.append(result['fun'])

    return target_returns, results