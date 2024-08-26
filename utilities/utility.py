import numpy as np
from scipy.optimize import minimize

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

#%%
