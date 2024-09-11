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
            # if ticker is found in monthly adjacent columns, meaning there are available data to calculate
            if ticker in df_monthly_return.columns:
                # Get date "years" ago
                date = pd.Timestamp.today() - pd.DateOffset(years=years)
                # Pick only stocks that are after this date
                monthly_return_list = df_monthly_return.loc[ pd.to_datetime(df_monthly_return['Date']) >= date, ticker].dropna().tolist()
                if len(monthly_return_list) >= 2:
                    # Calculate the i-years total return
                    total_return = np.prod(monthly_return_list) - 1

                    # Calculate the annualized average return
                    annualized_return = np.prod(monthly_return_list) ** (1/years)
                    df_overview.loc[df_overview['stock_ticker_symbol'] == ticker, 'return_rate_' + str(years) + 'y_avg'] = annualized_return

def set_volatility_by_years(df_overview, df_monthly_adj_close):
    # 1, 5, 10, 25 year returns
    # Loop through time spans
    for i, years in enumerate(variables.time_span_years):
        for i, ticker in enumerate(df_overview['stock_ticker_symbol']):
            if ticker in df_monthly_adj_close.columns:
                # Get date "years" ago
                date = pd.Timestamp.today() - pd.DateOffset(years=years)
                # Pick only stocks that are after this date
                adj_close_filtered = df_monthly_adj_close.loc[ pd.to_datetime(df_monthly_adj_close['Date']) >= date, ticker].dropna()
                std_deviation = adj_close_filtered.pct_change().std()

                if len(adj_close_filtered) >= 2:
                    df_overview.loc[df_overview['stock_ticker_symbol'] == ticker, 'volatility_' + str(years) + 'y'] = std_deviation


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

def efficient_frontier(df):
    returns = df['return_rate_5y_avg'].values
    volatilities = df['volatility_5y'].values
    num_assets = len(returns)
    results = []
    target_returns = np.linspace(min(returns), max(returns), 100)
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