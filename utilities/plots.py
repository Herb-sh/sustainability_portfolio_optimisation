import numpy as np
import pandas as pd
from networkx.algorithms.bipartite.basic import color
#
from pypfopt import EfficientFrontier
import cvxpy as cp
import cvxopt as opt
from cvxopt import blas, solvers
#
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#
import importlib
import utilities.utility as utility
importlib.reload(utility)


def plot_efficient_frontier(tickers, returns, volatility, mu, S):
    # Create a Plotly scatter plot
    fig = go.Figure()

    # Generate efficient frontier data @TODO check if correct!!!
    ef = EfficientFrontier(mu, S, solver=cp.CLARABEL)
    target_returns = np.linspace(mu.min(), mu.max(), 50)
    frontier_volatility = []
    frontier_returns = []
    '''
    for r in target_returns:
        ab = ef.efficient_return(target_return=r)
        weights = ef.clean_weights()
        std = np.sqrt(np.dot(np.dot(list(weights.values()), S), list(weights.values())))  # Compute std dev
        frontier_volatility.append(std)
        frontier_returns.append(r)'''

    # Marker: Add annotations for each point
    for i, (ret, vol, ticker) in enumerate(zip(returns, volatility, tickers)):
        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode='markers',
            text=ticker,
            textposition="top center",
            marker=dict(size=4, color='blue'),
            hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}<br>Return: %{y:.2f}<extra></extra>",
            showlegend=False
        ))

    # Marker: Add the maximum Sharpe ratio portfolio
    fig.add_trace(go.Scatter(
        x=[returns],
        y=[volatility],
        mode='markers',
        name='Max Sharpe Portfolio',
        marker=dict(color='red', size=10, symbol='star')
    ))

    # Line: Add the efficient frontier
    fig.add_trace(go.Scatter(
        x=frontier_volatility,
        y=frontier_returns,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=3)
    ))
    #
    fig.update_traces(hovertemplate=None)

    # Customize layout
    fig.update_layout(
        title='Efficient Frontier',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title='Risk (Standard Deviation)',
            tickformat='.0%',
            range=[min(volatility)-0.01, 0.8]
        ),
        yaxis=dict(
            title='Investment Return Average (last 25 years)',
            tickformat='.0%',
            range=[-0.2, 0.6]
        ),
        legend=dict(x=1, y=1),
        hovermode='closest',
        template='plotly',
    )

    fig.show()


def optimal_portfolio(returns): # @TODO check if useful!!
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def plot_diff(tickers, average_returns, actual_returns):
    # Create DataFrame
    df = pd.DataFrame({
        "Ticker": tickers,
        "Average Return (%)": average_returns,
        "Actual Return (Next Year) (%)": actual_returns
    })

    tickers_spread = utility.evenly_spaced_sample(tickers, 10)

    df_spread = df.loc[df['Ticker'].isin(tickers_spread)]

    # Reshape for Plotly (Long Format)
    df_melted = df_spread.melt(id_vars="Ticker",
                               var_name="Return Type",
                               value_name="Return (%)")

    # Create Bar Chart
    fig = px.bar(df_melted,
                 x="Ticker",
                 y="Return (%)",
                 color="Return Type",
                 barmode="group",
                 title="Comparison of average-returns (last 25 years) vs actual-returns (next test year)",
                 labels={"Return (%)": "Return (%)"},
                 text_auto=True)

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        yaxis=dict(
            title='Return rate (%)',
            tickformat='.0%'
        ),
        xaxis=dict(
            title='Company stock ticker'
        )
    )

    # Show the plot
    fig.show()

'''
Plots pie-chart given an allocation dictionary as input
'''
def plot_allocations(allocation):
    fig = go.Figure(data=[go.Pie(labels=pd.Series(allocation).index,
                                 values=pd.Series(allocation).values,
                                 textposition="inside",
                                 insidetextorientation="radial",
                                 textinfo='label+percent',
                                 hole=.3)])
    # Set background to white
    fig.update_layout(
        height=600,
        width=600,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        showlegend=False
    )
    fig.show()


'''
 PROPHET Plots
'''

def plot_lines_actual_vs_predicted(df_pct, forecasts, months=12):
    # Allocate the last 5 years of data for testing
    min_date = pd.to_datetime(df_pct.index[-1]).replace(day=1) - pd.DateOffset(months=12)
    min_datestr = min_date.strftime('%Y-%m-%d')

    X_train = df_pct.loc[df_pct.index < min_datestr]
    # df_test = dataframe.loc[dataframe.index >= min_datestr]

    # Collect 'ds' (date) and 'yhat' from each forecast
    forecast_dfs = [item[['ds', 'yhat']].rename(columns={'yhat': stock}) for stock, item in forecasts.items()]

    # Merge all forecasts on 'ds' (date)
    merged_forecast = forecast_dfs[0]
    for df in forecast_dfs[1:]:
        merged_forecast = merged_forecast.merge(df, on='ds', how='outer')

    # Compute the mean 'yhat' per time point
    y_pred = merged_forecast.iloc[:, 1:].mean(axis=1)
    y_true = df_pct.mean(axis=1)

    #
    train_true_list = y_pred[:len(X_train)]
    test_true_list = y_pred[len(X_train):]

    # Create the plot
    fig = go.Figure()

    # Add the timeseries line
    fig.add_trace(go.Scatter(y=y_true, x=df_pct.index.tolist(), mode='lines', name='Actual returns',
                             line=dict(color='#5c839f', width=2)))  #, line=dict(color='red'))
    # Add the training plot in red
    fig.add_trace(go.Scatter(y=train_true_list, x=df_pct.index.tolist()[:len(train_true_list)],
                             mode='lines', name='Train returns',
                             line=dict(color='red', width=2)))  #, line=dict(color='red')

    # Add the testing plot in green
    fig.add_trace(go.Scatter(y=test_true_list, x=df_pct.index.tolist()[len(train_true_list):],
                             mode='lines', name='Test returns',
                             line=dict(color='green', width=2)))  # , line=dict(color='green')

    fig.add_vline(x=min_datestr, line_color='red', line_dash='dash', line_width=1)

    # Update layout with labels
    fig.update_layout(
        title='1 Year Prediction vs Actual Plot',
        margin=dict(l=20, r=20, t=20, b=20),
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