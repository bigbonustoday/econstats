import statsmodels.formula.api as smf
import pandas as pd
import numpy as np


def main():
    # true model: y = 2 * x + eps
    # x ~ N(0, 1)
    # eps ~ N(0, 0.1)

    # generate training set
    N = 1000
    bs_N = 10000
    beta = 2
    dat = pd.DataFrame({
        'x': np.random.normal(0, 1, N),
        'eps': np.random.normal(0, 0.1, N)
    })
    dat['y'] = beta * dat['x'] + dat['eps']

    # ols
    model = smf.ols('y ~ x', data=dat)
    ols_result = model.fit()

    beta_hat = ols_result.params
    var_beta = ols_result.cov_params().loc['x', 'x']
    sigma_hat = ols_result.mse_resid ** 0.5
    fitted_se_ols = dat['x'].abs() * var_beta ** 0.5
    fitted_y = ols_result.fittedvalues

    # parametric bootstrap
    fitted_y_parametric_bootstrap = pd.DataFrame()
    for sample in range(bs_N):
        bootstapped_eps = np.random.normal(0, sigma_hat, N)
        parametric_boostrapped_y = fitted_y + bootstapped_eps
        dat_parametric_bootstrap = pd.DataFrame({
            'y': parametric_boostrapped_y,
            'x': dat['x']
        })
        model_parametric_bootstrap = smf.ols('y ~ x', data=dat_parametric_bootstrap)
        fitted_y_parametric_bootstrap[sample] = model_parametric_bootstrap.fit().fittedvalues
    fitted_y_parambs = fitted_y_parametric_bootstrap.mean(1)
    fitted_se_parambs = fitted_y_parametric_bootstrap.std(1)

    # non-parametric bootstrap
    fitted_ys = pd.DataFrame()
    for sample in range(bs_N):
        draws = np.random.choice(dat.index, size=N, replace=True)
        dat_npbs = dat.ix[draws]
        model_npbs = smf.ols('y ~ x', data=dat_npbs)
        fitted_ys[sample] = model_npbs.fit().predict(exog=dat)
    fitted_y_npbs = fitted_ys.mean(1)
    fitted_se_npbs = fitted_ys.std(1)

    df = pd.DataFrame({
        'ols mean': fitted_y,
        'ols se': fitted_se_ols,
        'pbs mean': fitted_y_parambs,
        'pbs se': fitted_se_parambs,
        'npbs mean': fitted_y_npbs,
        'npbs se': fitted_se_npbs
    })
