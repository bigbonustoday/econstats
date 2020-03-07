import numpy as np


def main():
    size = 10000
    theta = [0.5]
    sigma = 1
    print('Simulate MA(1) time series for nobs=' + str(size) + ', theta=' + str(theta[0]) + ', sigma_e=' + str(sigma))
    y = simulateMA(theta, sigma, size)
    print('Compute log likelihood for theta=-1, -0.9, .., +1.0 and sigma_e=' + str(sigma))
    kl = KalmanFilterUnivariateMA(y, p=1)
    maxlogl = -1e10
    maxth = 0
    for th in np.linspace(start=-1.0, stop=1.0, num=21):
        logl = kl.computeLogLikelihood(theta=[th], sigma2=sigma ** 2)
        if logl > maxlogl:
            maxlogl = logl
            maxth = th
        print('Log likelihood(theta=' + str(round(th, 1)) + ')=' + str(round(logl, 2)))
    print('Max likelihood=' + str(round(maxlogl, 2)) + ', achieved when theta=' + str(round(maxth, 1)))


def simulateMA(theta, sigma, size):
    p = len(theta)
    e = np.random.normal(0, sigma, size=size + p)
    y = []
    w = np.insert(theta, 0, 1)
    for t in range(size): y.append(e[t:(t+p+1)].dot(w))
    y = np.array(y)
    return y


class KalmanFilterUnivariateMA(object):
    """
    This is a Python 3.6 implementation Kalman Filter for univariate MA series
    Notation follows Recitation 11 Page 2 of MIT Time Series Analysis OpenCourseware
    """
    def __init__(self, y, p):
        self.y = np.copy(y)  # observed data series
        self.p = p  # order of MA model
        # initialize MA parameters
        self.theta = np.zeros(p)
        self.sigma2 = np.var(y)

    def computeLogLikelihood(self, theta, sigma2):
        p = self.p
        y = self.y
        # initial values of alpha and P
        alpha = [np.zeros(p + 1).reshape(p + 1, 1)]
        P = [np.diag(np.ones(p + 1)) * np.var(y)]  # P_{t|t-1}
        alphau = []  # updated alpha, or alpha_{t|t}
        Pu = []  # updated P, or P_{t|t}
        # matrix objects
        Z = np.insert(theta, 0, 1).reshape(1, p + 1)
        T = np.array([[0] * (p + 1)] + [[0] * i + [1] + [0] * (p - i) for i in range(p)])
        R = np.array([1] + [0] * p).reshape(p + 1, 1)
        Q = sigma2
        RQR = R.dot(Q).dot(R.T)

        yhat = []
        F = []
        for t in range(len(y)):
            yhat.append(Z.dot(alpha[t]))
            F.append(Z.dot(P[t]).dot(Z.T))
            # update step
            alphau.append(alpha[t] + P[t].dot(Z.T) / F[t] * (y[t] - yhat[t]))
            Pu.append(P[t] - 1 / F[t] * P[t].dot(Z.T).dot(Z).dot(P[t]))
            # predict step
            alpha.append(T.dot(alphau[t]))
            P.append(T.dot(Pu[t]).dot(T.T) + RQR)
        # log likelihood
        L = 0
        for t in range(len(y)):
            L -= (0.5 * np.log(2 * np.pi * F[t]) + (y[t] - yhat[t]) ** 2 / F[t] / 2)
        return L[0][0]
