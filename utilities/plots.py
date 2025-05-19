import numpy as np
import pandas as pd
#
from pypfopt import EfficientFrontier
import cvxpy as cp
#
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

def plot_diff(tickers, average_returns, actual_returns):
    # Create DataFrame
    df = pd.DataFrame({
        "Ticker": tickers,
        "Average Return (%)": average_returns,
        "Actual Return (Last Year) (%)": actual_returns
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
                 title="Comparison of average-returns (last 25 years) vs actual-returns (last year)",
                 labels={"Return (%)": "Return (%)"},
                 text_auto=True)

    fig.update_layout(
        template="plotly_white",
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
