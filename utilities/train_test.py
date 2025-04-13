import pandas as pd
import torch
import numpy as np
#
import datetime
from sklearn.preprocessing import LabelEncoder

from utilities import variables


def get_dataframe_tabular(df):
    df_tabular = pd.DataFrame(columns=['ticker', 'month', 'year', 'monthly_return'])

    tickers = df.columns
    dates = df.index

    ticker_column = []
    month_column = []
    year_column = []
    date_column = []
    monthly_return_column_minus12 = []
    monthly_return_column_minus11 = []
    monthly_return_column_minus10 = []
    monthly_return_column_minus9 = []
    monthly_return_column_minus8 = []
    monthly_return_column_minus7 = []
    monthly_return_column_minus6 = []
    monthly_return_column_minus5 = []
    monthly_return_column_minus4 = []
    monthly_return_column_minus3 = []
    monthly_return_column_minus2 = []
    monthly_return_column_minus1 = []
    monthly_return_column = []
    monthly_return_column_plus1 = []
    monthly_return_column_plus2 = []
    monthly_return_column_plus3 = []
    monthly_return_column_plus4 = []
    monthly_return_column_plus5 = []
    monthly_return_column_plus6 = []
    monthly_return_column_plus7 = []
    monthly_return_column_plus8 = []
    monthly_return_column_plus9 = []
    monthly_return_column_plus10 = []
    monthly_return_column_plus11 = []
    monthly_return_column_plus12 = []


    for i, ticker in enumerate(tickers):
        for j, date in enumerate(dates):
            ticker_column.append(ticker)
            #
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")
            month_column.append(dt.month)
            year_column.append(dt.year)
            # Date should refer to t or current month, t+1 is the predicted month, t-11 is a year ago
            date_column.append(date)
            # Input
            monthly_return_column_minus11.append( getvalue(df, date, ticker, -11) )
            monthly_return_column_minus10.append( getvalue(df, date, ticker, -10) )
            monthly_return_column_minus9.append( getvalue(df, date, ticker, -9) )
            monthly_return_column_minus8.append( getvalue(df, date, ticker, -8) )
            monthly_return_column_minus7.append( getvalue(df, date, ticker, -7) )
            monthly_return_column_minus6.append( getvalue(df, date, ticker, -6) )
            monthly_return_column_minus5.append( getvalue(df, date, ticker, -5) )
            monthly_return_column_minus4.append( getvalue(df, date, ticker, -4) )
            monthly_return_column_minus3.append( getvalue(df, date, ticker, -3) )
            monthly_return_column_minus2.append( getvalue(df, date, ticker, -2) )
            monthly_return_column_minus1.append( getvalue(df, date, ticker, -1) )
            monthly_return_column.append(df.loc[date, ticker])
            # Future Output
            monthly_return_column_plus1.append( getvalue(df, date, ticker, 1) )
            monthly_return_column_plus2.append( getvalue(df, date, ticker, 2) )
            monthly_return_column_plus3.append( getvalue(df, date, ticker, 3) )
            monthly_return_column_plus4.append( getvalue(df, date, ticker, 4) )
            monthly_return_column_plus5.append( getvalue(df, date, ticker, 5) )
            monthly_return_column_plus6.append( getvalue(df, date, ticker, 6) )
            monthly_return_column_plus7.append( getvalue(df, date, ticker, 7) )
            monthly_return_column_plus8.append( getvalue(df, date, ticker, 8) )
            monthly_return_column_plus9.append( getvalue(df, date, ticker, 9) )
            monthly_return_column_plus10.append( getvalue(df, date, ticker, 10) )
            monthly_return_column_plus11.append( getvalue(df, date, ticker, 11) )
            monthly_return_column_plus12.append( getvalue(df, date, ticker, 12) )


    df_tabular = pd.DataFrame(data={
        'ticker': ticker_column,
        'month': month_column,
        'year': year_column,
        'date': date_column,
        # 'm_return(t-12)': monthly_return_column_minus12,
        'm_return(t-11)': monthly_return_column_minus11,
        'm_return(t-10)': monthly_return_column_minus10,
        'm_return(t-9)': monthly_return_column_minus9,
        'm_return(t-8)': monthly_return_column_minus8,
        'm_return(t-7)': monthly_return_column_minus7,
        'm_return(t-6)': monthly_return_column_minus6,
        'm_return(t-5)': monthly_return_column_minus5,
        'm_return(t-4)': monthly_return_column_minus4,
        'm_return(t-3)': monthly_return_column_minus3,
        'm_return(t-2)': monthly_return_column_minus2,
        'm_return(t-1)': monthly_return_column_minus1,
        'm_return(t)': monthly_return_column,
        'm_return_target(t+1)': monthly_return_column_plus1,
        'm_return_target(t+2)': monthly_return_column_plus2,
        'm_return_target(t+3)': monthly_return_column_plus3,
        'm_return_target(t+4)': monthly_return_column_plus4,
        'm_return_target(t+5)': monthly_return_column_plus5,
        'm_return_target(t+6)': monthly_return_column_plus6,
        'm_return_target(t+7)': monthly_return_column_plus7,
        'm_return_target(t+8)': monthly_return_column_plus8,
        'm_return_target(t+9)': monthly_return_column_plus9,
        'm_return_target(t+10)': monthly_return_column_plus10,
        'm_return_target(t+11)': monthly_return_column_plus11,
        'm_return_target(t+12)': monthly_return_column_plus12
    })

    le = LabelEncoder()
    df_tabular['stock_ticker_label'] = le.fit_transform(df_tabular['ticker'])
    df_tabular.drop(columns=['ticker'], inplace=True)

    # shift last month as it does not have a following month, for this reason it will NaN (which we do not want)
    # max_date below works as shift by removing last item
    #df_tabular = df_tabular.shift()

    # Exclude first year to avoid null values generated from tabular dataset
    min_date = pd.to_datetime(df.index).min() + pd.DateOffset(months=12)
    min_datestr = min_date.strftime('%Y-%m-%d')
    #
    max_date = pd.to_datetime(df.index).max()
    max_datestr = max_date.strftime('%Y-%m-%d')
    #
    df_tabular = df_tabular.loc[(df_tabular['date'] >= min_datestr) & (df_tabular['date'] < max_datestr)]
    # reorder index after filtering
    df_tabular.reset_index(drop=True, inplace=True)
    return df_tabular

def getvalue(df, date, ticker, months_add=1):
    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    try:
        offset = pd.DateOffset(months=months_add)
        return df.loc[(dt + offset).strftime('%Y-%m-%d'), ticker]
    except KeyError:
        return None

def get_dataframe_tabular_multi(df):
    df_overview = pd.read_csv('../../../data/df_overview.csv', index_col=0)
    #
    df_tabular = get_dataframe_tabular(df)
    #
    df_tabular_multi = df_tabular.merge(df_overview, on='stock_ticker_label')

    cols = df_tabular_multi.columns.tolist()
    #
    for col in ['return_rate_1y_avg', 'return_rate_5y_avg', 'return_rate_10y_avg', 'return_rate_25y_avg',
                'volatility_1y_avg', 'volatility_5y_avg', 'volatility_10y_avg', 'volatility_25y_avg',
                'score', 'company_name', 'company_esg_score_group', 'stock_ticker_symbol',
                'industry', 'stock_exchange']:
        while col in cols:
            cols.remove(col)

    return df_tabular_multi[cols]

def split_train_test_tabular(df_tabular, months=12, target_key='m_return_target(t+1)'):
    min_date = pd.to_datetime(df_tabular['date']).max() - pd.DateOffset(months=months)
    min_datestr = min_date.strftime('%Y-%m-%d')
    # filter nan values out
    df_tabular = df_tabular.loc[df_tabular[target_key].notna()]

    df_train, df_test = df_tabular.loc[df_tabular['date'] <= min_datestr], df_tabular.loc[df_tabular['date'] > min_datestr]
    columns = df_train.columns.to_list()
    # Remove all future columns after train/test split, date column should not be part of training
    for i, col in enumerate(df_train.columns):
        if (col.find('t+') != -1) or (col.find('date') != -1):
            columns.remove(col)

    #
    X_train = df_train[columns]
    y_train = df_train[[target_key]]

    X_test = df_test[columns]
    y_test = df_test[[target_key]]

    return X_train, y_train, X_test, y_test, min_datestr

def create_sequences(df_ts, df_static, seq_length, out_seq_length=1):
    """
    Create sequences of data for LSTM model.
    """
    x_ts, x_static, y = [], [], []
    for i in range(len(df_ts) - seq_length):
        x_ts_data = df_ts.iloc[i:i+seq_length].values
        x_ts_data_transposed = x_ts_data.transpose(1, 0)

        #
        if len(df_static) > 0:
            x_static.append(df_static[i]) # Sequences are NOT added to
        #
        y_data = df_ts.iloc[i+seq_length: i + seq_length + out_seq_length].values
        y_data_transposed = y_data.transpose(1, 0)

        # y_data = fix_array_length(y_data, out_seq_length)
        if len(y_data) == out_seq_length:
            x_ts.append(x_ts_data_transposed)  # Sequence of `seq_length` time points
            y.append(y_data_transposed)   # Target is the next time step

    return (torch.tensor(x_ts, dtype=torch.float32),
            torch.tensor(x_static, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))

def fix_array_length(arr, length):
    """
    Ensures each row of the array has a fixed length. If shorter, fills with NaN; if longer, truncates.

    Parameters:
    - arr (numpy.ndarray): The input 2D array.
    - length (int): The desired length of each row.

    Returns:
    - numpy.ndarray: A 2D array with the specified column length.
    """
    fixed_arr = np.full((arr.shape[0], length), np.nan)  # Initialize with NaN
    num_cols = min(arr.shape[1], length)  # Determine how many columns to copy
    fixed_arr[:, :num_cols] = arr[:, :num_cols]  # Copy existing data
    return fixed_arr
