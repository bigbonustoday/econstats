from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

def main():
    size = 10000

    # true DGP: y = alpha + beta * x + eps
    # Quantile(y, q) = alpha + beta * x + Quantile(eps, q)

    # Case 1
    # eps is iid N(0, 0.01)
    # quantile regression shouldn't show any variation in estimated beta to quantile
    # but there should be variation in the intercept
    eps = np.random.normal(0, 0.1, size=size)
    x = np.random.normal(0, 1, size=size)
    alpha = 1
    beta = 2
    y = alpha + beta * x + eps

    ols_params_1, quantile_params_1 = quantile_regression(y, x)

    # Case 2
    # eps is iid Normal mean zero but variance is correlated with exp(x)
    # quantile regression should show higher beta for higher quantiles
    eps = np.random.normal(0, 0.1, size=size)
    x = np.random.normal(0, 1, size=size)
    eps = eps * np.exp(x)
    alpha = 1
    beta = 2
    y = alpha + beta * x + eps

    ols_params_2, quantile_params_2 = quantile_regression(y, x)


def quantile_regression(y, x):
    data = pd.DataFrame({
        'y': y,
        'x': x
    })

    # ols
    model_ols = smf.ols('y ~ x', data)
    res_ols = model_ols.fit()
    ols_params = res_ols.params

    # quantile regression
    model_quant = smf.quantreg('y ~ x', data)
    params = pd.DataFrame(index=np.arange(0.05, 0.96, 0.1),
                          columns=['Intercept', 'x'])
    for q in np.arange(0.05, 0.96, 0.1):
        res = model_quant.fit(q=q)
        params.loc[q, :] = res.params

    return ols_params, params

