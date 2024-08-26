from dash import Dash, dcc, html
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('../data/10_monthly_returns_complete.csv')
df_close = pd.read_csv('../data/10_monthly_adjacent_close.csv', index_col=0)
df_overview = pd.read_csv('../data/data_10_overview.csv', index_col=0)
# Create a scatter plot
fig = px.scatter(df_overview[['stock_ticker_symbol', 'return_rate_5y_avg', 'volatility_5y']],
                 x='volatility_5y',
                 y='return_rate_5y_avg',
                 text='stock_ticker_symbol',
                 title='Risk-Return Plot',
                 labels={'volatility_5y': 'Risk (Standard Deviation)',
                         'return_rate_5y_avg': 'Expected Return'},
                 template='plotly')

# Add labels to the points
fig.update_traces(textposition='top center')

# Show the plot
# fig.show()

st.plotly_chart(fig)