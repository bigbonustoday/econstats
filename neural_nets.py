import numpy as np
import pandas as pd

def textbook_exercise_11_3():
    N = 1000
    N_test = 100000
    x = np.array([np.random.normal(size=2) for i in range(N)])
    x_test = np.array([np.random.normal(size=2) for i in range(N_test)])
    z = np.array(np.random.normal(size=N))
    z_test = np.array(np.random.normal(size=N_test))
    alpha1 = np.array([3, 3])
    alpha2 = np.array([3, -3])
    y = 1 / (1 + np.exp(-x.dot(alpha1))) + (x.dot(alpha2)) ** 2 + 0.3 * z
    y_test = 1 / (1 + np.exp(-x_test.dot(alpha1))) + (x_test.dot(alpha2)) ** 2 + 0.3 * z_test

    df = {}
    M = 5
    for decay in np.linspace(0.0, 0.2, num=11):
        print decay
        y_predict = []
        y_predict_test = []
        for i in range(10):
            self = SingleLayerNN(y, x, M, decay, 0.0001, max_tries=10 ** 6)
            y_predict.append(self.predict(x))
            y_predict_test.append(self.predict(x_test))
        y_predict = np.array(y_predict).mean(0)
        y_predict_test = np.array(y_predict_test).mean(0)
        mse_training = ((y - y_predict) ** 2).mean()
        mse_test = ((y_test - y_predict_test) ** 2).mean()
        df[decay] = pd.Series({'training': mse_training, 'test': mse_test})
    df = pd.DataFrame(df)


# unit test: test error should be between 1x and 2x 0.09
def test():
    N = 100
    N_test = 10000
    x = np.array([np.random.normal(size=2) for i in range(N)])
    x_test = np.array([np.random.normal(size=2) for i in range(N_test)])
    z = np.array(np.random.normal(size=N))
    z_test = np.array(np.random.normal(size=N_test))
    alpha1 = np.array([3, 3])
    alpha2 = np.array([3, -3])
    y = 1 / (1 + np.exp(-x.dot(alpha1))) + 1 / (1 + np.exp(-x.dot(alpha2))) + z
    y_test = 1 / (1 + np.exp(-x_test.dot(alpha1))) + 1 / (1 + np.exp(-x_test.dot(alpha2))) + z_test

    y_predict = []
    y_predict_test = []
    for i in range(5):
        self = SingleLayerNN(y, x, hidden_units=5, weight_decay=0.0,
                             learning_rate=0.0001, max_tries=10 ** 3)
        y_predict.append(self.predict(x))
        y_predict_test.append(self.predict(x_test))
    y_predict = np.array(y_predict).mean(0)
    y_predict_test = np.array(y_predict_test).mean(0)
    mse_training = ((y - y_predict) ** 2).mean()
    mse_test = ((y_test - y_predict_test) ** 2).mean()
    errs = pd.Series({'training': mse_training, 'test': mse_test})


class SingleLayerNN:
    y = None
    x = None
    M = 0  # number of hidden units
    lamda = 0  # weight decay parameter; 0 = no decay / shrinkage
    gamma = 0  # learning rate
    p = 0  # number of features
    max_tries = 0  # number of max tries
    N = 0  # nobs

    # neural net coefs
    beta = None
    alpha = None

    def __init__(self, y, x, hidden_units, weight_decay, learning_rate, max_tries=100,
                 ssr_reduction_threshold=0.0001):
        # standardize input
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.x_mean = x.mean(0)
        self.x_std = x.std(0)

        self.y = (y - y.mean()) / y.std()
        x = (x - x.mean(0)) / x.std(0)
        self.x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

        self.M = hidden_units
        self.lamda = weight_decay
        self.gamma = learning_rate
        self.N = x.shape[0]
        self.p = x.shape[1]
        self.max_tries = max_tries
        self.ssr_reduction_threshold = ssr_reduction_threshold

        self.fit()

    def predict(self, x):
        x = (x - self.x_mean) / self.x_std
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        Z = 1 / (1 + np.exp(-x.dot(self.alpha.T)))
        Z = np.concatenate([np.ones((x.shape[0], 1)), Z], axis=1)
        f = Z.dot(self.beta)
        f = f * self.y_std + self.y_mean
        return f

    # fit neural net
    def fit(self):
        # initialize coefs
        self.beta = np.array(np.random.normal(0, 0.7, size=self.M + 1))
        self.alpha = np.array([np.random.normal(0, 0.7, size=self.p + 1) for i in range(self.M)])
        ssr = self._compute_sum_of_sqrd_residuals()

        for i in range(self.max_tries):
            beta_gradient, alpha_gradient = self._compute_gradient()
            self.beta -= self.gamma * beta_gradient
            self.alpha -= self.gamma * alpha_gradient
            ssr_update = self._compute_sum_of_sqrd_residuals()
            ssr_reduction = 1 - ssr_update / ssr
            if ssr_reduction <= self.ssr_reduction_threshold:
                break
            ssr = ssr_update
        return

    def _sigma_prime(self, x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    def _compute_sum_of_sqrd_residuals(self):
        Z = 1 / (1 + np.exp(-self.x.dot(self.alpha.T)))
        Z = np.concatenate([np.ones((self.N, 1)), Z], axis=1)
        f = Z.dot(self.beta)
        return np.sum((self.y - f) ** 2)

    def _compute_gradient(self):
        Z = 1 / (1 + np.exp(-self.x.dot(self.alpha.T)))
        Z = np.concatenate([np.ones((self.N, 1)), Z], axis=1)
        f = Z.dot(self.beta)
        delta = -2.0 * (self.y - f)
        beta_gradient = Z.T.dot(delta) + 2 * self.lamda * self.beta
        sigma_term = self._sigma_prime(self.alpha.dot(self.x.T))
        beta_delta = np.array([self.beta]).T.dot(np.array([delta]))[1:, :]  # remove first row of beta
        s = sigma_term * beta_delta
        alpha_gradient = s.dot(self.x) + 2 * self.lamda * self.alpha
        return beta_gradient, alpha_gradient
