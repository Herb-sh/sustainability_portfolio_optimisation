from prophet import Prophet
import pandas as pd


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

'''
Merges a train dataframe with the forecast dataframe
Receives train dataframe as input, a prophet forecast and minimum time-horizon (default 12 months)
'''
def get_df_with_forecast(dataframe, forecasts, months=12):
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
    return merged_forecast, y_pred, y_true

'''
Convert Prophet forecast to a dataframe
'''
def get_df_from_forecast(forecast):
    # Collect 'ds' (date) and 'yhat' from each forecast
    forecast_dfs = [item[['ds', 'yhat']].rename(columns={'yhat': stock}) for stock, item in forecast.items()]

    # Merge all forecasts on 'ds' (date)
    merged_forecast = forecast_dfs[0]
    for df in forecast_dfs[1:]:
        merged_forecast = merged_forecast.merge(df, on='ds', how='outer')

    merged_forecast = merged_forecast.set_index('ds')

    return merged_forecast
