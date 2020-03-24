from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt


def main():
    # replicate Figure 10.2
    plot_errors()

    # part (d)
    plot_errors(threshold=12)


def plot_errors(threshold=9.34):
    data = simulate(nobs=12000, threshold=threshold)
    train = data[0:2000]
    train_y = train[:, 10]
    test = data[2000:]
    test_y = test[:, 10]
    iter_range = range(1, 3001, 100)

    models = {
        'single tree': DecisionTreeClassifier(),  # single tree with no constraint on depth
        'stump': DecisionTreeClassifier(max_depth=1)  # stump (two-leaf tree)
    }
    for iter in iter_range:
        models[str(iter)] = BoostedDecisionTreeClassifier(boost_iter=iter, max_depth=1)  # boosted stump

    err_test = {}
    err_train = {}
    for label, model in models.items():
        print(label)
        model.fit(X=train[:, 0:10], y=train[:, 10])
        # test error
        y = model.predict(X=test[:, 0:10])
        err_test[label] = (y != test_y).mean()
        # train error
        y = model.predict(X=train[:, 0:10])
        err_train[label] = (y != train_y).mean()

    # plot for (b) and (c)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    test_plot, = ax.plot(list(iter_range), [err_test[str(ind)] for ind in iter_range], '-')
    train_plot, = ax.plot(list(iter_range), [err_train[str(ind)] for ind in iter_range], '-')
    single_tree_test = ax.axhline(y=err_test['single tree'], linestyle='--', color='blue')
    single_tree_train = ax.axhline(y=err_train['single tree'], linestyle='--', color='green')
    ax.legend([test_plot, train_plot, single_tree_test, single_tree_train],
              ['Boosting test error', 'Boosting train error',
               'single tree test error', 'single tree train error'])
    plt.show()
    plt.savefig('boosting.png', format='png')


def simulate(nobs, threshold):
    """
    simulate test and train data
    :param nobs: sample size
    :param threshold: threshold for sum(X^2); base case is 9.34 which is median(chi2(10)); part (d) changes
    this to 12
    :return: nobs x 11 2d np array, X -> columns 0:10, y -> column 10
    """
    data = []
    x = np.random.normal(0, 1, size=nobs * 10)
    for ind in range(nobs):
        entry = x[ind * 10: ((ind + 1) * 10)]
        y = 1 if np.sum(entry ** 2) > threshold else -1
        data.append(np.append(entry, y))
    return np.array(data)


class BoostedDecisionTreeClassifier(ClassifierMixin):
    """
    class for Boosted classification tree
    we only need the fit(X, y) and predict(X) functions
    """
    def __init__(self, boost_iter=100, **kwargs):
        self.boost_iter = boost_iter
        self.kwargs = kwargs
        self.alpha_vec = None
        self.trees = None

    def fit(self, X, y):
        boost_iter = self.boost_iter
        nobs = len(y)
        weights = np.ones(nobs)
        alpha_vec = []
        trees = []

        for iter in range(boost_iter):
            tree = DecisionTreeClassifier(**self.kwargs)
            weights = weights / weights.sum()
            tree.fit(X=X, y=y, sample_weight=weights)
            err = np.sum((tree.predict(X=X) != y) * weights) / np.sum(weights)
            alpha = np.log((1 - err) / err)
            weights = weights * np.exp(alpha * (tree.predict(X=X) != y))
            alpha_vec.append(alpha)
            trees.append(tree)
        self.alpha_vec = np.array(alpha_vec)
        self.trees = trees

    def predict(self, X):
        boost_iter = self.boost_iter
        alpha_vec = self.alpha_vec
        trees = self.trees
        pred_mat = []
        for iter in range(boost_iter):
            pred_mat.append(trees[iter].predict(X=X))
        pred_mat = np.array(pred_mat).transpose()
        return np.sign(pred_mat.dot(alpha_vec))
