import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LogisticRegression, Lasso, Ridge


def main():
    n_train = 50
    n_test = 5000
    n_runs = 20

    r2 = {}
    for n_coef in [10, 30, 300]:
        for noise_to_signal in np.linspace(0.1, 0.5, 5):
            label = str(n_coef) + '#' + str(noise_to_signal)
            print(label)
            r2[label + '#Lasso#regression'] = run_regression(
                Lasso, n_train, n_test, noise_to_signal, n_coef, n_runs)
            r2[label + '#Ridge#regression'] = run_regression(
                Ridge, n_train, n_test, noise_to_signal, n_coef, n_runs)
            r2[label + '#Lasso#classification'] = run_classification(
                LogisticRegression, n_train, n_test, noise_to_signal, n_coef, n_runs, 'l1')
            r2[label + '#Ridge#classification'] = run_classification(
                LogisticRegression, n_train, n_test, noise_to_signal, n_coef, n_runs, 'l2')

    # rearrange data
    df = pd.DataFrame(r2).unstack()
    df = pd.DataFrame({'Percentage explained': df})
    labels = list(df.index.get_level_values(0))
    df.loc[:, 'subset'] = [(x.split('#')[0]) for x in labels]
    df.loc[:, 'NSR'] = [(x.split('#')[1]) for x in labels]
    df.loc[:, 'model'] = [(x.split('#')[2] + '#' + x.split('#')[3]) for x in labels]

    g = sns.FacetGrid(df, row='subset', col='model', margin_titles=True, row_order=['300', '10', '30'])
    g.map(sns.boxplot, 'NSR', 'Percentage explained')


def run_classification(reg_model, n_train, n_test, noise_to_signal, n_coef, n_runs, penalty):
    c_range = np.array([np.exp(c) for c in
               np.linspace(np.log(0.001), np.log(1000), 50)])
    r2_list = []
    for i in range(n_runs):
        r2 = -1
        for c in c_range:
            model = reg_model(C=c, penalty=penalty, solver='saga', max_iter=1000)
            r2 = max(r2, compute_r2(
                model, n_coef, n_train, n_test, noise_to_signal, classification=True))
        r2_list.append(r2)
    return np.array(r2_list)


def run_regression(reg_model, n_train, n_test, noise_to_signal, n_coef, n_runs):
    c_range = np.array([np.exp(c) for c in
               np.linspace(np.log(0.001), np.log(1000), 50)])
    alpha_range = 1.0 / n_train / c_range
    r2_list = []
    for i in range(n_runs):
        r2 = -1
        for alpha in alpha_range:
            model = reg_model(alpha=alpha)
            r2 = max(r2, compute_r2(
                model, n_coef, n_train, n_test, noise_to_signal, classification=False))
        r2_list.append(r2)
    return np.array(r2_list)


# compute r2 of a model without tuning alpha or C
def compute_r2(model, n_coef, n_train, n_test, noise_to_signal, classification):
    data = simulate(n_coef, n_train + n_test, noise_to_signal, classification=classification)
    model.fit(data['x'][0: n_train, :], data['y'][0: n_train])
    y_pred = model.predict(data['x'][n_train:, :])
    y_test = data['y'][n_train:]
    if classification:
        misclassification_rate = (y_test != y_pred).mean()
        r2 = 1 - misclassification_rate / 0.5
    else:
        resid = y_test - y_pred
        r2 = 1 - np.var(resid) / np.var(y_test)
    return r2


def simulate(n_coef, n_sample, noise_to_signal,
             total_n_coef=300, classification=False):
    x = np.random.normal(
        0, 1, size=n_sample * total_n_coef
    ).reshape([n_sample, total_n_coef])
    coef = np.zeros(total_n_coef)
    coef[0:n_coef] = np.random.normal(0, 1, size=n_coef)
    # E(y|x)
    ey = x.dot(coef)
    sigma_eps = np.std(ey) * noise_to_signal ** 0.5
    eps = np.random.normal(
        0, sigma_eps, size=n_sample
    )
    y = ey + eps

    if classification:
        y = np.round(sigmoid(y))

    return {'y': y, 'x': x}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x)))