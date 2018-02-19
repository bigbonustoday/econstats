import pandas as pd
import numpy as np

theta0 = {
    'mu1': 1,
    'mu2': 8,
    'sigma1': 5,
    'sigma2': 5,
    'p': 0.5
}

theta1 = {
    'mu1': 2,
    'mu2': 6,
    'sigma1': 2,
    'sigma2': 4,
    'p': 0.9
}


def simulate_mixture_gaussian(size, theta):
    x1 = np.random.normal(loc=theta['mu1'], scale=theta['sigma1'], size=size)
    x2 = np.random.normal(loc=theta['mu2'], scale=theta['sigma2'], size=size)
    w = np.random.binomial(n=1, p=theta['p'], size=size)
    x = x1 * w + x2 * (1 - w)
    return x


def _log_likelihood_mixed_normal(x, theta):
    p1 = 1.0 / (2 * np.pi * theta['sigma1'] ** 2) ** 0.5 * np.exp(
        - 1.0 / 2 / theta['sigma1'] ** 2 * (x - theta['mu1']) ** 2)
    p2 = 1.0 / (2 * np.pi * theta['sigma2'] ** 2) ** 0.5 * np.exp(
        - 1.0 / 2 / theta['sigma2'] ** 2 * (x - theta['mu2']) ** 2)
    l = np.sum(np.log(p1 * theta['p'] + p2 * (1 - theta['p'])))

    # find posterior distribution of Z
    prob_z = p1 * theta['p'] / (p1 * theta['p'] + p2 * (1 - theta['p']))
    return {'logL': l,
            'prob_z': prob_z}


# use EM algorithm to fit mixed normal distribution
def fit_mixture_gaussian(x, theta0=theta0, max_attempt=10):
    theta = theta0

    print('Attempt #0')
    l = _log_likelihood_mixed_normal(x, theta)['logL']
    print(list(theta.values()))
    print('LogL = ' + str(l))

    for attempt in range(max_attempt):
        print('Attempt #' + str(attempt + 1))

        # E step
        prob_z = _log_likelihood_mixed_normal(x, theta)['prob_z']

        # M step
        prob_z_1 = 1 - prob_z
        mu1 = np.sum(prob_z * x) / np.sum(prob_z)
        mu2 = np.sum(prob_z_1 * x) / np.sum(prob_z_1)
        sigma1 = (np.sum(prob_z * (x - mu1) ** 2) / np.sum(prob_z)) ** 0.5
        sigma2 = (np.sum(prob_z_1 * (x - mu2) ** 2) / np.sum(prob_z_1)) ** 0.5
        p = np.mean(prob_z)

        # update
        theta = {
            'mu1': mu1,
            'mu2': mu2,
            'sigma1': sigma1,
            'sigma2': sigma2,
            'p': p
        }
        # log likelihood should increase at each iteration!!
        l = _log_likelihood_mixed_normal(x, theta)['logL']
        print(list(theta.values()))
        print('LogL = ' + str(l))
    return theta


def main():
    x = simulate_mixture_gaussian(size=1000000, theta=theta1)
    theta_fittd = fit_mixture_gaussian(x, max_attempt=100)
    return