import numpy as np
import pandas as pd


class SmoothingCubicSpline():
    def __init__(self, y, lamda=0):
        if not isinstance(y, pd.Series):
            raise TypeError('y is not pandas series!')
        self.y = y
        self.K = len(y)
        self.lamda = lamda

        self._construct_cubic_spline()

    # compute basis function coefficients
    def _compute_theta(self):


    # compute the N matrix (K by K)
    def _construct_cubic_spline(self):
        y = self.y
        K = self.K

        x = pd.Series(y.index, index=y.index)
        N = pd.DataFrame(np.nan, index=y.index, columns=range(K))
        N[0] = 1
        N[1] = x

        ind = K
        d2_1 = (x - x.iloc[ind - 2]).apply(lambda x: x if x >= 0 else 0) ** 3
        d2_2 = (x - x.iloc[K - 1]).apply(lambda x: x if x >= 0 else 0) ** 3
        d2_3 = x.iloc[ind - 2] - x.iloc[K - 1]
        d2 = (d2_1 - d2_2) / d2_3
        for ind in range(2, K):
            d1_1 = (x - x.iloc[ind - 2]).apply(lambda x: x if x >= 0 else 0) ** 3
            d1_2 = (x - x.iloc[K - 1]).apply(lambda x: x if x >= 0 else 0) ** 3
            d1_3 = x.iloc[ind - 2] - x.iloc[K - 1]
            d1 = (d1_1 - d1_2) / d1_3
            N[ind] = d1 - d2
        self.N = N

    # compute the sigma_N matrix (K by K)

def test_fit_natural_cubic_spline():
    y = pd.Series({
        0.1: 0.3,
        0.3: 0.4,
        0.5: 0.2,
        0.7: 0.1,
        0.9: 0.5
    })
