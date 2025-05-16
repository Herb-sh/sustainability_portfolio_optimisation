#
from prophet import Prophet
import plotly.graph_objects as go
import pandas as pd
import numpy as np
#
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
#
import utilities.variables as variables
#
import cvxpy as cp

def train_predict(dataframe, months=12):
    df_train_long = dataframe.reset_index().melt(id_vars=['Date'], var_name='ticker', value_name='y')
    df_train_long.rename(columns={'Date': 'ds'}, inplace=True)

    # model
    models = {}
    forecasts = {}

    for ticker, data in df_train_long.groupby('ticker'):
        model = Prophet()
        model.fit(data[['ds', 'y']])  # Train model

        future = model.make_future_dataframe(periods=months, freq='ME')  # Forecast next given months
        forecast = model.predict(future)

        models[ticker] = model
        forecasts[ticker] = forecast

    return forecasts

def forecast_to_df(dataframe, forecasts, months=12):
    # Allocate the last 5 years of data for testing
    min_date = pd.to_datetime(dataframe.index[-1]).replace(day=1) - pd.DateOffset(months=12)
    min_datestr = min_date.strftime('%Y-%m-%d')

    # Collect 'ds' (date) and 'yhat' from each forecast
    forecast_dfs = [item[['ds', 'yhat']].rename(columns={'yhat': stock}) for stock, item in forecasts.items()]

    # Merge all forecasts on 'ds' (date)
    merged_forecast = forecast_dfs[0]
    for df in forecast_dfs[1:]:
        merged_forecast = merged_forecast.merge(df, on='ds', how='outer')

    # Compute the mean 'yhat' per time point
    y_pred = merged_forecast.tail(months)
    y_true = dataframe.loc[dataframe.index >= min_datestr].head(months)

    # re
    return y_pred, y_true

def plot(dataframe, forecasts, months=12):
    # Allocate the last 5 years of data for testing
    min_date = pd.to_datetime(dataframe.index[-1]).replace(day=1) - pd.DateOffset(months=12)
    min_datestr = min_date.strftime('%Y-%m-%d')

    X_train = dataframe.loc[dataframe.index < min_datestr]
    # df_test = dataframe.loc[dataframe.index >= min_datestr]

    # Collect 'ds' (date) and 'yhat' from each forecast
    forecast_dfs = [item[['ds', 'yhat']].rename(columns={'yhat': stock}) for stock, item in forecasts.items()]

    # Merge all forecasts on 'ds' (date)
    merged_forecast = forecast_dfs[0]
    for df in forecast_dfs[1:]:
        merged_forecast = merged_forecast.merge(df, on='ds', how='outer')

    # Compute the mean 'yhat' per time point
    y_pred = merged_forecast.iloc[:, 1:].mean(axis=1)
    y_true = dataframe.mean(axis=1)

    #
    train_true_list = y_pred[:len(X_train)]
    test_true_list = y_pred[len(X_train):]

    # Create the plot
    fig = go.Figure()

    # Add the timeseries line
    fig.add_trace(go.Scatter(y=y_true, x=dataframe.index.tolist(), mode='lines', name='Actual returns',
                             line=dict(color='#5c839f', width=2)))  #, line=dict(color='red'))
    # Add the training plot in red
    fig.add_trace(go.Scatter(y=train_true_list, x=dataframe.index.tolist()[:len(train_true_list)],
                             mode='lines', name='Train returns',
                             line=dict(color='red', width=2)))  #, line=dict(color='red')

    # Add the testing plot in green
    fig.add_trace(go.Scatter(y=test_true_list, x=dataframe.index.tolist()[len(train_true_list):],
                             mode='lines', name='Test returns',
                             line=dict(color='green', width=2)))  # , line=dict(color='green')

    fig.add_vline(x=min_datestr, line_color='red', line_dash='dash', line_width=1)

    # Update layout with labels
    fig.update_layout(
        title='1 Year Prediction vs Actual Plot',
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

# To create an allocation we keep a minimum of 12 months, for all 3 cases (1 month, 6 months, 12 months)
def get_portfolio_performance(forecasts, file_name = "weights.csv", min_avg_return=variables.MIN_AVG_RETURN, months=12):
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

def create_discrete_allocation(df, raw_weights, total_portfolio_value = 10000):
    latest_prices = get_latest_prices(df)

    da = DiscreteAllocation(raw_weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.greedy_portfolio()
    print("Discrete allocation:", allocation)
    print("Funds remaining: €{:.2f}".format(leftover))
