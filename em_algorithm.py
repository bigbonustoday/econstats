import pandas as pd
import numpy as np

theta0 = {
    'mu1': 0,
    'mu2': 1,
    'sigma1': 1,
    'sigma2': 1,
    'p': 0.5
}

theta1 = {
    'mu1': 2,
    'mu2': 6,
    'sigma1': 2,
    'sigma2': 4,
    'p': 0.3
}

def simulate_mixed_normal(size, theta):
    x1 = np.random.normal(loc=theta['mu1'], scale=theta['sigma1'], size=size)
    x2 = np.random.normal(loc=theta['m2'], scale=theta['sigma2'], size=size)
    w = np.random.binomial(n=1, p=theta['p'], size=size)
    return x1 * w + x2 * (1 - w)

def _log_likelihood_mixed_normal(x, theta):
    p1 = 1 / (2 * np.pi * theta['sigma1'] ** 2) * np.exp(- 1.0 / 2 / theta['sigma1'] ** 2 * (x - theta['mu1']) ** 2)
    p2 = 1 / (2 * np.pi * theta['sigma2'] ** 2) * np.exp(- 1.0 / 2 / theta['sigma2'] ** 2 * (x - theta['mu2']) ** 2)
    l = np.log(p1) * theta['p'] + np.log(p2) * (1 - theta['p'])

    # find posterior distribution of Z
    prob_z = p1 * theta['p'] / (p1 * theta['p'] + p2 * (1 - theta['p']))
    return {'logL': np.sum(l),
    'prob_z': prob_z}


# use EM algorithm to fit mixed normal distribution
def fit_mixed_normal(x, theta0=theta0, max_attempt=10):
    theta = theta0
    for attempt in range(max_attempt):
        print 'Attempt #', attempt + 1
        print  '...', theta.values()
        prob_z = _log_likelihood_mixed_normal(x, theta)['prob_z']
