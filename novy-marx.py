import numpy as np
import pandas as pd

# returns a T x size pandas matrix
def simulate_ar1(rho, T, size, sigma, x0=0):
    eps = np.transpose(np.matrix([np.random.normal(size=T) for ind in range(size)])) * sigma

    x = np.matrix([[x0] * size])
    for ind in range(T - 1):
        x_last = x[ind, ...]
        x_next = rho * x_last + eps[ind, ...]
        x = np.append(x, x_next, axis=0)

    return pd.DataFrame(x)


def simulate_persistent_regressor():
    rho = 0.995
    sigma_x = 0.01
    sigma_eps = 0.05
    T = 10000
    size = 100
    x0 = 0
    beta = 0

    x = simulate_ar1(rho, T, size, sigma_x, x0)
    eps = pd.DataFrame({ind: np.random.normal(size=T) for ind in range(size)}).multiply(sigma_eps)
    y = x.multiply(beta) + eps

    beta_hat = []
    for ind in range(size):
        model = pd.ols(y=y[ind], x=x[ind])
        beta_hat.append(model.beta['x'])


def simulate_novymarx():
    rho = 0.985
    sigma_x = 0.01
    sigma_lamda = sigma_x
    sigma_r = 0.16
    T = 10000
    size = 100
    x0 = 0
    lamda0 = 0

    lamda = simulate_ar1(rho, T, size, sigma_lamda, lamda0)
    x = simulate_ar1(rho, T, size, sigma_x, x0)
    eps = pd.DataFrame({ind: np.random.normal(size=T) for ind in range(size)}).multiply(sigma_r)
    y = lamda + eps

    print('Running legit regression...')
    beta_hat = []
    for ind in range(size):
        model = pd.ols(y=y[ind], x=lamda[ind])
        beta_hat.append(model.beta['x'])
    print('mean(beta)=' + str(np.mean(beta_hat)))
    print('std(beta)=' + str(np.std(beta_hat)))

    print('Running spurious regression...')
    beta_hat = []
    for ind in range(size):
        model = pd.ols(y=y[ind], x=x[ind])
        beta_hat.append(model.beta['x'])
    print('mean(beta)=' + str(np.mean(beta_hat)))
    print('std(beta)=' + str(np.std(beta_hat)))
