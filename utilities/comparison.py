import pandas as pd
import torch
import cvxpy as cp
#
import plotly.graph_objects as go
#
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import utilities.variables as variables

# Methods
def generate_plot(df, df_tabular, y_train_pred, y_test_pred):
    # Create the plot
    fig = go.Figure()
    skip = len(y_test_pred)
    indices = df_tabular['date'].unique()[skip:]
    min_date = pd.to_datetime(df_tabular['date'].max()) - pd.DateOffset(months=len(y_test_pred))
    min_datestr = min_date.strftime('%Y-%m-%d')

    # Add the timeseries line
    fig.add_trace(go.Scatter(x=indices, y=df.mean(axis=1), mode='lines', name='Actual returns',
                             line=dict(color='#5c839f', width=2)))
    # Add the training plot in red
    fig.add_trace(go.Scatter(x=indices[:len(y_train_pred)], y=y_train_pred,
                             mode='lines', name='Train returns',
                             line=dict(color='red', width=2)))

    # Add the testing plot in green
    fig.add_trace(go.Scatter(x=indices[len(y_train_pred) -1:],
                             y=[y_train_pred[len(y_train_pred)-1], *y_test_pred],
                             mode='lines', name='Test returns',
                             line=dict(color='green', width=2)))

    fig.add_vline(x=min_datestr, line_color='red', line_dash='dash', line_width=1)

    # Update layout with labels
    fig.update_layout(
        title='{0} Month Prediction vs Actual Plot'.format(len(y_test_pred)),
        xaxis=dict(
            title='Date'
        ),
        yaxis=dict(
            title='Day closing return (%)',
            tickformat='.0%',
            range=[0.75, 1.6]
        ),
        legend=dict(title="Legend"),
        template="plotly_white"
    )

    fig.show()

def get_train_test_mean_pred(y_train_pred_1m, y_test_pred_1m, columns_count):
    train_pred_torch_list = torch.from_numpy(y_train_pred_1m)
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

def get_portfolio_performance(df_pred, file_name = "weights.csv", min_avg_return=variables.MIN_AVG_RETURN, months=12):
    '''
    Constructs an optimized portfolio
    :param df_pred:
    :param file_name:
    :param min_avg_return:
    :param months:
    :return:
    '''
    # Create DataFrame of forecasted prices
    # Calculate expected returns and sample covariance
    mu_0 = expected_returns.mean_historical_return(df_pred, frequency=12, returns_data=True)
    # Get only tickers with a mean historical return of at least 5%
    optimal_tickers = mu_0[mu_0 > min_avg_return].index
    df_optimal = df_pred[optimal_tickers].tail(months)
    #
    mu = expected_returns.mean_historical_return(df_optimal, frequency=12, returns_data=True)

    S = risk_models.CovarianceShrinkage(df_optimal, returns_data=True).ledoit_wolf()
    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, solver=cp.CLARABEL)

    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    ef.save_weights_to_file(file_name)  # saves to file
    #
    p_mu, p_sigma, p_sharpe = ef.portfolio_performance(verbose=True)

    return df_optimal, cleaned_weights, mu, S, p_sigma, p_sharpe

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

    exchange_rate = {
        "yen_to_euro": 0.0062,
        "us_to_euro": 0.88,
        "pound_to_euro": 1.17
    }
    # Loop through all tickers, for each ticker we find its corresponding stock-exchange
    # Based on stock-exchange we grab the corresponding exchange-rate and apply to the whole column of that ticker
    for i, col in enumerate(df.columns):
        stock_exchange = df_overview.loc[df_overview['stock_ticker_symbol'] == col, 'stock_exchange'].values[0]
        # Yen
        if stock_exchange == 'TKS':
            df[col] = df[col] * exchange_rate['yen_to_euro']
        # Dollar
        if stock_exchange == 'NAS' or stock_exchange == 'NYS':
            df[col] = df[col] * exchange_rate['us_to_euro']
        # Pound
        if stock_exchange == 'LON':
            df[col] = df[col] * exchange_rate['pound_to_euro']

    return df

def create_discrete_allocation(df, raw_weights, total_portfolio_value = 10000, greedy=False):
    latest_prices = get_latest_prices(df)

    da = DiscreteAllocation(raw_weights, latest_prices, total_portfolio_value=total_portfolio_value)
    if greedy:
        allocation, leftover = da.greedy_portfolio()
    else:
        allocation, leftover = da.lp_portfolio()

    print("Discrete allocation:", allocation)
    print("Funds remaining: €{:.2f}".format(leftover))
    return allocation, leftover