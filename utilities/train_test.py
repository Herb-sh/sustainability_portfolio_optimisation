import pandas as pd
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

    for i, ticker in enumerate(tickers):
        for j, date in enumerate(dates):
            ticker_column.append(ticker)
            #
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")
            month_column.append(dt.month)
            year_column.append(dt.year)
            # Date should refer to t or current month, t+1 is the predicted month, t-11 is a year ago
            date_column.append(date)
            #
            # monthly_return_column_minus12.append( getvalue(df, date, ticker, -12) )
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
            #
            monthly_return_column.append(df.loc[date, ticker])
            #
            monthly_return_column_plus1.append( getvalue(df, date, ticker, 1) )

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
        'm_return_target(t+1)': monthly_return_column_plus1
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

    return df_tabular

def get_dataframe_tabular_multi(df):
    df_overview = pd.read_csv('../../../data/df_overview.csv', index_col=0)
    #
    df_tabular = get_dataframe_tabular(df)
    #
    X_train, y_train, X_test, y_test = get_train_test(df_tabular)
    X_train_multi = X_train.merge(df_overview, on='stock_ticker_label')
    X_test_multi = X_test.merge(df_overview, on='stock_ticker_label')

    cols = X_train_multi.columns.tolist()
    #
    for col in ['return_rate_1y_avg', 'return_rate_5y_avg', 'return_rate_10y_avg', 'return_rate_25y_avg',
                'volatility_1y', 'volatility_5y', 'volatility_10y', 'volatility_25y', 'score',
                'company_name', 'company_esg_score_group', 'stock_ticker_symbol',
                'industry', 'stock_exchange']:
        while col in cols:
            cols.remove(col)
    #
    X_train_multi = X_train_multi[cols]
    X_test_multi = X_test_multi[cols]

    return X_train_multi, X_test_multi

def get_train_test(df_tabular, months=12):
    min_date = pd.to_datetime(df_tabular['date']).max() - pd.DateOffset(months=months)
    min_datestr = min_date.strftime('%Y-%m-%d')

    train, test = df_tabular.loc[df_tabular['date'] <= min_datestr], df_tabular.loc[df_tabular['date'] > min_datestr]

    target_key = 'm_return_target(t+1)'

    columns = train.columns.to_list()
    columns.remove(target_key)
    columns.remove('date') # date column should not be part of training
    #
    X_train = train[columns]
    y_train = train[[target_key]]

    X_test = test[columns]
    y_test = test[[target_key]]
    return X_train, y_train, X_test, y_test

def getvalue(df, date, ticker, months_add=1):
    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    try:
        offset = pd.DateOffset(months=months_add)
        return df.loc[(dt + offset).strftime('%Y-%m-%d'), ticker]
    except KeyError:
        return None