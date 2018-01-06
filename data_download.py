from yahoo_finance import Share
import pandas as pd
import numpy as np

def get_stock_returns(ticker, start_date, end_date):
    stock = Share(ticker)
    prices = stock.get_historical(start_date=start_date, end_date=end_date)
    prices = pd.to_numeric(pd.DataFrame(prices).set_index('Date')['Adj_Close'])
    prices = prices.sort_index(ascending=True)
    ret = prices.pct_change(1)
    return ret


def get_x_y(beta, size=1000000):
    eps = np.random.uniform(low=-1, high=1, size=size)
    x = np.random.normal(loc=0, scale=10, size=size)
    y = beta * x + eps
    return pd.DataFrame({
        'y':y,
        'x':x
    })


def get_estimated_betas(beta, sample_size=1000):
    betas = []
    for i in range(sample_size):
        print i
        df = get_x_y(beta=beta)
        model = pd.ols(y=df['y'], x={'x': df['x']})
        betas.append(model.beta['x'])
    return betas

def ols(size=1000):
    x1 = pd.Series(np.random.normal(size=size))
    x2 = pd.Series(np.random.normal(size=size))
    beta = pd.Series({
        'intercept': 2.0,
        'x1': 1.0,
        'x2': -1.0
    })
    X = pd.DataFrame({
        'intercept': 1.0,
        'x1': x1,
        'x2': x2
    })
    eps = pd.Series(np.random.normal(size=size))
    y = (X * beta).sum(1) + eps
    model = pd.ols(y=y, x={'x1': x1, 'x2': x2})

    # replicate OLS
    n = X.shape[0]
    K = X.shape[1]
    Sxx = X.transpose().dot(X)
    Sxx_inv = pd.DataFrame(np.linalg.pinv(Sxx.values), Sxx.columns, Sxx.index)
    Sxy = X.transpose().dot(y)
    b = Sxx_inv.dot(Sxy)
    e = y - X.dot(b)
    s2 = (e.transpose().dot(e)) / (n - K)
    s = s2 ** 0.5
    b_var = Sxx_inv * s2

    # check beta mean estimates
    np.testing.assert_array_almost_equal(b, model.beta[b.index])

    # check beta var-covar matrix
    np.testing.assert_array_almost_equal(b_var, model.var_beta.loc[b_var.index, b_var.columns])

    # check rmse
    np.testing.assert_almost_equal(s, model.rmse)

    # F test: beta1 + beta2 = 0
    X = pd.DataFrame({
        'intercept': 1,
        'x': x1 - x2
    })
    n = X.shape[0]
    K = X.shape[1]
    Sxx = X.transpose().dot(X)
    Sxx_inv = pd.DataFrame(np.linalg.pinv(Sxx.values), Sxx.columns, Sxx.index)
    Sxy = X.transpose().dot(y)
    b = Sxx_inv.dot(Sxy)
    e_constrained = y - X.dot(b)

    ee = e.T.dot(e)
    ee_constrained = e_constrained.T.dot(e_constrained)

    J = 1
    F = (ee_constrained - ee) / J / (ee / (n-K))



