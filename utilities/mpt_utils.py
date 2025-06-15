import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
import cvxpy as cp
#
import plotly.graph_objects as go
#
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier, EfficientSemivariance
from pypfopt import objective_functions
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy.optimize import minimize
import importlib
import utilities.variables as variables
import utilities.plots as plots
# Methods


def get_train_test_mean_pred(y_train_pred_1m, y_test_pred_1m, columns_count):
    train_pred_torch_list = torch.from_numpy(y_train_pred_1m)
    print(len(train_pred_torch_list)/columns_count)
    # Reshape to (num_samples, num_features) for normalization
    train_rows = int(len(train_pred_torch_list)/columns_count)
    train_pred_torch_view = train_pred_torch_list.view(train_rows, columns_count)
    y_train_pred = pd.DataFrame(train_pred_torch_view).mean(axis=1)
    #
    test_pred_torch_list = torch.from_numpy(y_test_pred_1m)
    test_rows = int(len(test_pred_torch_list) / columns_count)
    test_pred_torch_view = test_pred_torch_list.view(test_rows, columns_count)
    y_test_pred = pd.DataFrame(test_pred_torch_view).mean(axis=1)

    return y_train_pred, y_test_pred

def get_df_from_pred_list(df, train_list, test_list):
    '''
    It gets true values or in other words training values and on top of them
    it concats the predicted values (1m, 6m, 12m) to create the whole timeline
    so that it can then be used to compare the last year (true + pred) with the known true values
    '''
    train_time_steps = int(len(train_list) / len(df.columns))
    test_time_steps = int(len(test_list) / len(df.columns))
    #
    # Create an empty dictionary to hold each ticker's data
    data_dict = {}

    for i, ticker in enumerate(df.columns):
        train_start = i * train_time_steps
        train_end = train_start + train_time_steps
        #
        test_start = i * test_time_steps
        test_end = test_start + test_time_steps
        #
        train_data = train_list[train_start:train_end]
        test_data = test_list[test_start:test_end]
        #
        ticker_data = [*train_data, *test_data]
        #
        data_dict[ticker] = ticker_data

    return pd.DataFrame(data_dict, columns=df.columns)

# To create an allocation we keep a minimum of 12 months, for all 3 cases (1 month, 6 months, 12 months)
def get_prophet_portfolio_performance(forecasts, file_name ="weights.csv", min_avg_return=variables.MIN_AVG_RETURN, months=12):
    # Create DataFrame of forecasted prices
    # Collect 'ds' (date) and 'yhat' from each forecast
    forecast_dfs = [item[['ds', 'yhat']].rename(columns={'yhat': stock}) for stock, item in forecasts.items()]

    # Merge all forecasts on 'ds' (date)
    merged_forecast = forecast_dfs[0]
    for df in forecast_dfs[1:]:
        merged_forecast = merged_forecast.merge(df, on='ds', how='outer')

    merged_forecast = merged_forecast.set_index('ds')

    # Calculate expected returns and sample covariance
    mu_0 = expected_returns.mean_historical_return(merged_forecast)

    # Get only tickers with a mean historical return of at least 5%
    optimal_tickers = mu_0[mu_0 > min_avg_return].index
    df_optimal = merged_forecast[optimal_tickers].tail(months)

    mu = expected_returns.mean_historical_return(df_optimal)
    mu = round(mu)
    S = risk_models.CovarianceShrinkage(df_optimal).ledoit_wolf()

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, solver=cp.CLARABEL)
    # ef_new = EfficientFrontier(mu, S, solver=cp.CLARABEL)

    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    # volatility = ef.min_volatility()
    ef.save_weights_to_file(file_name)  # saves to file
    #
    p_mu, p_sigma, p_sharpe = ef.portfolio_performance(verbose=True)

    return df_optimal, cleaned_weights, mu, S, p_sigma, p_sharpe

def get_portfolio_performance(df, file_name = "weights.csv", min_avg_return=variables.MIN_AVG_RETURN):
    # Calculate expected returns and sample covariance
    mu_0 = expected_returns.mean_historical_return(df, frequency=12)

    # Get only tickers with a mean historical return of at least 5%
    optimal_tickers = mu_0[mu_0 > min_avg_return].index
    df_optimal = df[optimal_tickers]

    mu = expected_returns.mean_historical_return(df_optimal, frequency=12)
    S = risk_models.CovarianceShrinkage(df_optimal, frequency=12).ledoit_wolf() # Ledoit-Wolf shrinkage (df_optimal, frequency=12), # Exponential Covariance

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, solver=cp.CLARABEL) # cp.ECOS
    ef.min_volatility()
    #ef.add_objective(objective_functions.L2_reg, gamma=0.1)

    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    ef.save_weights_to_file(file_name)  # saves to file
    #
    p_mu, p_sigma, p_sharpe = ef.portfolio_performance(verbose=True)
    return df_optimal, cleaned_weights, mu, S, p_sigma, p_sharpe


'''
Post Modern Portfolio Theory
'''
def get_semivariance_portfolio_performance(df, file_name = "weights.csv", min_avg_return=variables.MIN_AVG_RETURN):
    # Calculate expected returns and sample covariance
    mu_0 = expected_returns.mean_historical_return(df, frequency=12)

    # Get only tickers with a mean historical return of at least 5%
    optimal_tickers = mu_0[mu_0 > min_avg_return].index
    df_optimal = df[optimal_tickers]

    mu = expected_returns.mean_historical_return(df_optimal, frequency=12)
    historical_returns = expected_returns.returns_from_prices(df)

    #mu = expected_returns.mean_historical_return(df)

    # Optimize for maximal Sharpe ratio
    es = EfficientSemivariance(mu, historical_returns, solver=cp.CLARABEL, frequency=12)
    es.max_quadratic_utility()

    cleaned_weights = es.clean_weights()

    #es.save_weights_to_file(file_name)  # saves to file
    #
    p_mu, p_sigma, p_sortino = es.portfolio_performance(verbose=True)
    return df_optimal, cleaned_weights, np.NaN, mu, p_sigma, p_sortino


'''
Constructs an optimized portfolio
'''
def get_returns_portfolio_performance(df_pred, file_name ="weights.csv", min_avg_return=variables.MIN_AVG_RETURN, opt_months=12):
    # Create DataFrame of forecasted prices
    # Calculate expected returns and sample covariance
    mu_0 = expected_returns.mean_historical_return(df_pred, frequency=12, returns_data=True)
    # Get only tickers with a mean historical return of at least 5%
    optimal_tickers = mu_0[mu_0 > min_avg_return].index
    df_optimal = df_pred[optimal_tickers].tail(opt_months)
    #
    mu = expected_returns.mean_historical_return(df_optimal, frequency=12, returns_data=True)

    S = risk_models.CovarianceShrinkage(df_optimal, returns_data=True).ledoit_wolf()
    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, solver=cp.CLARABEL)
    ef.add_constraint(lambda w: w <= 0.05)

    raw_weights = ef.max_sharpe() # ef.min_volatility()
    cleaned_weights = ef.clean_weights()

    ef.save_weights_to_file(file_name)  # saves to file
    #
    p_mu, p_sigma, p_sharpe = ef.portfolio_performance(verbose=True)

    return df_optimal, cleaned_weights, mu, S, p_sigma, p_sharpe

def create_discrete_allocation(df, raw_weights, total_portfolio_value = 10000, is_greedy=True):
    latest_prices = get_latest_prices(df)

    da = DiscreteAllocation(raw_weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.greedy_portfolio() if is_greedy else da.lp_portfolio()

    # print("Discrete allocation:", allocation)
    # print("Funds remaining: €{:.2f}".format(leftover))
    return allocation, leftover

def get_full_prices_from_returns(df_returns_complete, df_complete, months):
    # Get the initial price for each ticker, first month of last time-horizon
    df_initial_prices = df_complete.tail(months).head(1)
    # Get last "time-horizon" return rates together with predicted value/s
    df_returns = df_returns_complete.tail(months)

    # Set indices
    df_returns.index = df_complete.tail(months).index
    # Create empty price DataFrame with same shape
    df_prices = pd.DataFrame(index=df_returns.index, columns=df_returns.columns)

    df_prices.iloc[0] = df_initial_prices.values[0]
    # Set the first row as initial prices * (1 + first return)
    first_date = df_prices.index[0]

    for i, ticker in enumerate(df_returns.columns):
        value = df_initial_prices[ticker][first_date] * (1 + df_returns.loc[first_date, ticker])
        df_prices.loc[df_prices.index == first_date, ticker] = value

    # Compute prices for all subsequent dates
    for i in range(1, len(df_returns)):
        current_date = df_returns.index[i]
        previous_date = df_returns.index[i - 1]

        for j, ticker in enumerate(df_returns.columns):
            value = df_prices[ticker][previous_date] * (1 + df_returns[ticker][current_date])
            df_prices.loc[df_prices.index == current_date, ticker] = value

    return df_prices

def get_full_prices_euro(df, df_overview):
    df_euro = df.copy()
    exchange_rate = {
        "yen_to_euro": 0.0062,
        "us_to_euro": 0.88,
        "pound_to_euro": 1.17
    }
    # Loop through all tickers, for each ticker we find its corresponding stock-exchange
    # Based on stock-exchange we grab the corresponding exchange-rate and apply to the whole column of that ticker
    for i, col in enumerate(df_euro.columns):
        stock_exchange = df_overview.loc[df_overview['stock_ticker_symbol'] == col, 'stock_exchange'].values[0]
        # Yen
        if stock_exchange == 'TKS':
            df_euro[col] = df_euro[col] * exchange_rate['yen_to_euro']
        # Dollar
        if stock_exchange == 'NAS' or stock_exchange == 'NYS':
            df_euro[col] = df_euro[col] * exchange_rate['us_to_euro']
        # Pound
        if stock_exchange == 'LON':
            df_euro[col] = df_euro[col] * exchange_rate['pound_to_euro']

    return df_euro

'''
Important!
Generalized method that constructs an optimized portfolio and plots it
gets a dataframe with forecasted values attached to training values 
and the equivalent in real prices (euro). 

df_forecast (DataFrame) in percentage values (0.02 for 2%), all training months + predicted months (e.g. 240 + 12 = 252 months)
df (DataFrame) real euro prices of stocks (as read df_monthly_prices_complete_euro), all 300 months 
opt_months (int) time-horizon of data that will be used to construct the portfolio (e.g. 60 months)
other_threshold (decimal) a minimum percentage value for a stock to be shown, otherwise they are grouped into "other"
'''
def portfolio_and_plot(df_forecast, df, opt_months=(variables.TEST_YEARS_NR * 12), plot_threshold=0.02, file_name="weights.csv"):
    importlib.reload(plots)

    df_prices_train_and_forecast = get_full_prices_from_returns(df_forecast, df[df_forecast.columns], opt_months)
    #
    df_portfolio, raw_weights, mu, S, sigma, sharpe = get_returns_portfolio_performance(df_forecast, file_name=file_name, min_avg_return=0, opt_months=opt_months)
    allocation, leftover = create_discrete_allocation(df_prices_train_and_forecast[df_portfolio.columns], raw_weights, is_greedy=True)
    #
    weights_grouped, weights_pct_grouped = get_grouped_weights(raw_weights, allocation, plot_threshold)

    print('-- Allocation --')
    print(allocation)
    print('-- Weights Percentage --')
    print(weights_pct_grouped)

    plots.plot_allocations(weights_grouped)

    return weights_grouped, mu, S, allocation

'''
Important!
Generalized method that constructs an optimized portfolio and plots it
gets a dataframe with forecasted values attached to training values 
and the equivalent in real prices (euro). 

df_forecast (DataFrame) in percentage values (0.02 for 2%), all training months + predicted months (e.g. 240 + 12 = 252 months)
df (DataFrame) real euro prices of stocks (as read df_monthly_prices_complete_euro), all 300 months 
opt_months (int) time-horizon of data that will be used to construct the portfolio (e.g. 60 months)
other_threshold (decimal) a minimum percentage value for a stock to be shown, otherwise they are grouped into "other"
'''
def benchmark_portfolio_and_plot(df, opt_months=60, plot_threshold=0.02, file_name="weights.csv", semivariance=False):
    importlib.reload(plots)

    # is semivariance flag is set SemivarianceFrontier willbe used, other EfficientFrontier
    df_portfolio, raw_weights, mu, S, sigma, sharpe = get_semivariance_portfolio_performance(df, file_name=file_name, min_avg_return=0) if semivariance else get_portfolio_performance(df, file_name=file_name, min_avg_return=0)
    allocation, leftover = create_discrete_allocation(df_portfolio, raw_weights, is_greedy=True)
    #
    weights_grouped, weights_pct_grouped = get_grouped_weights(raw_weights, allocation, plot_threshold)

    print('-- Allocation --')
    print(allocation)
    print('-- Weights Percentage --')
    print(weights_pct_grouped)

    plots.plot_allocations(weights_grouped)

    return weights_grouped, mu, S, allocation

'''
Given weights and allocation as parameter
It will return the percentage equivalent for each allocated item
Values under threshold are summed in "Other"
'''
def get_grouped_weights(weights, allocation, threshold):
    weights_filtered = OrderedDict((key, round(value, 4)) for key, value in weights.items() if key in allocation)
    sum_val = round(sum(weights_filtered.values()), 4)

    # Group all weights that pass threshold, mark the rest as other
    weights_grouped = {
        k: v for k, v in weights_filtered.items() if v/sum_val >= threshold
    }
    other_weights_list = [v for k, v in weights_filtered.items() if v/sum_val < threshold]
    other_total = round(sum(other_weights_list), 4)
    if other_total > 0:
        weights_grouped["Other("+str(len(other_weights_list))+")"] = other_total

    # Group all weights that pass threshold, mark the rest as other
    weights_pct_grouped = {
        k: round(v/sum_val, 4) for k, v in weights_filtered.items() if v/sum_val >= threshold
    }
    other_pct_total = round(sum(v/sum_val for k, v in weights_filtered.items() if v/sum_val < threshold), 4)
    if other_pct_total > 0:
        weights_pct_grouped["Other("+str(len(other_weights_list))+")"] = other_pct_total

    return weights_grouped, weights_pct_grouped

# Efficient-Frontier
def portfolio_performance(weights, returns, volatilities):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(volatilities), weights)))
    return portfolio_return, portfolio_volatility

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

def negative_sharpe_ratio(weights, returns, volatilities, risk_free_rate=0):
    p_return, p_volatility = portfolio_performance(weights, returns, volatilities)
    return -(p_return - risk_free_rate) / p_volatility

'''
This method is used to get the actual return-rate of a portfolio
given a dataframe of prices and the allocation dictionary of a portfolio
'''
def get_allocation_return_rate(df_prices, allocation):
    tickers = allocation.keys()
    df_subset = df_prices[list(tickers)]

    # get the first and last price row
    first_prices = df_subset.iloc[0]
    last_prices = df_subset.iloc[-1]

    # compute portfolio value at start and end
    initial_value = sum(allocation[ticker] * first_prices[ticker] for ticker in allocation)
    final_value = sum(allocation[ticker] * last_prices[ticker] for ticker in allocation)

    # calculate return rate
    return_rate = (final_value - initial_value) / initial_value
    return return_rate

def generate_overview_table(weights, mu, S, df_pct, format_nr=True):
    df_pct_train = df_pct.head(int(variables.ALL_YEARS_NR - variables.TEST_YEARS_NR) * 12)
    df_pct_test = df_pct.tail(variables.TEST_YEARS_NR * 12)

    tickers = [k for k, v in weights.items()]
    # 1. cCreate overview with Weight
    df_view = pd.DataFrame.from_dict(weights, orient='index', columns=['Share Count'])
    # 2. set average covariance
    S_f = round(S.loc[S.index.isin(tickers), tickers], 2)
    S_avg = {}
    for ticker in S_f.columns:
        cov_with_others = S_f.loc[ticker].drop(ticker)  # remove self-covariance
        S_avg[ticker] = cov_with_others.mean()
    df_view['Average Covariance'] = S_avg
    # 3. set annual average returns
    df_view['Average Returns'] = round(mu.loc[mu.index.isin(tickers)], 4).values
    # 4. set predicted last year return rate
    df_view['Return Last 12 Months'] = round((df_pct_train[tickers].tail(12).prod() - 1), 4).values
    # 5. set actual last year return rate
    df_view['Return (Actual) Next 12 Months'] = round((df_pct_test[tickers].head(12).prod() - 1), 4).values

    if format_nr:
        df_view['Average Returns'] = df_view['Average Returns'].map('{:.2%}'.format)
        df_view['Return Last 12 Months'] = df_view['Return Last 12 Months'].map('{:.2%}'.format)
        df_view['Return (Actual) Next 12 Months'] = df_view['Return (Actual) Next 12 Months'].map('{:.2%}'.format)

    return df_view