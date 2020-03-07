import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    N = 100  # sample size
    bs_N = 1000  # bootstrap sample size

    # generate sample
    dat = pd.DataFrame({
        'x': np.random.uniform(low=0, high=1, size=N),
        'eps': np.random.normal(0, 0.3, size=N)
    })
    dat['y'] = (1 - dat['x'] ** 2) ** 0.5 + dat['eps']

    se = univariate_bootstrap(dat=dat, N=N, bs_N=bs_N)

    # observe that correlation between ols and pbs is very high (they should be asymptotically
    # identical!)
    print(se.corr())


def univariate_bootstrap(dat, N, bs_N):
    """
    computes boostrapped standard errors of a univariate linear regression model
    :param dat: DataFrame; must have 'x' and 'y' columns
    :return: DataFrame; standard errors using specified model, parametric bootstrap,
    non-parametric bootsrap
    """
    # plot raw data
    plt.scatter(dat['x'], dat['y'], marker='.', color='grey')

    # ols
    model = smf.ols('y ~ x', data=dat)
    ols_result = model.fit()
    var_beta = ols_result.cov_params()
    sigma_hat = ols_result.mse_resid ** 0.5
    x_matrix = pd.DataFrame({'x': dat['x'], 'Intercept': 1})
    fitted_se_ols = pd.Series({ind: x_matrix.loc[ind,:].dot(var_beta).dot(x_matrix.loc[ind,:])
                               for ind in x_matrix.index}).pow(0.5)
    fitted_y = ols_result.fittedvalues
    plt.plot(dat['x'], fitted_y)

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
    plt.plot(dat['x'], fitted_y + 2 * fitted_se_parambs, color='blue')
    plt.plot(dat['x'], fitted_y - 2 * fitted_se_parambs, color='blue')


    # non-parametric bootstrap
    fitted_ys = pd.DataFrame()
    for sample in range(bs_N):
        draws = np.random.choice(dat.index, size=N, replace=True)
        dat_npbs = dat.loc[draws, :]
        model_npbs = smf.ols('y ~ x', data=dat_npbs)
        fitted_ys[sample] = model_npbs.fit().predict(exog=dat)
    fitted_y_npbs = fitted_ys.mean(1)
    fitted_se_npbs = fitted_ys.std(1)
    plt.plot(dat['x'], fitted_y + 2 * fitted_se_npbs, color='green')
    plt.plot(dat['x'], fitted_y - 2 * fitted_se_npbs, color='green')
    plt.show()

    se = pd.DataFrame({
        'ols': fitted_se_ols,
        'pbs': fitted_se_parambs,
        'npbs': fitted_se_npbs
    })
    return se
