import numpy as np
import pandas as pd

def basis_function(ind, knots_including_bounds, order):
    """
    returns a list of f(x), basis functions in BSpline functional space, specifying knots and order
    :param ind: returns ith basis function
    :param knots: list of float, knots, including bounds
    :param order: int, order of function
    :return: a list of functions, each function takes one float
    """

    # check input validity
    if not isinstance(knots_including_bounds, list):
        raise TypeError('input k not a list!')
    if not isinstance(knots_including_bounds[0], float):
        raise TypeError('input k not a list of float!')
    if not (isinstance(order, int) or isinstance(order, float)):
        raise TypeError('input t not an int or float')

    knots_including_bounds.sort()
    order = int(order)
    num_of_knots = len(knots_including_bounds) - 2
    lower_bound = knots_including_bounds[0]
    upper_bound = knots_including_bounds[-1]
    knots = [knots_including_bounds[i] for i in range(1, num_of_knots + 1)]
    knots_expanded = [lower_bound] * order + knots + [upper_bound] * order

    def B(x, i, m):
        """
        returns the ith basis function of order m
        :param x: float
        :param i: int : 0 <= i <= m + num_of_knots - 1
        :param m: int
        :return:
        """
        if (i < 0) or (i > 2 * order + num_of_knots - m):
            raise ValueError('{index}th basis function does not exist!'.format(index=i))
        if m == 1:
            return 1.0 if (x >= knots_expanded[i]) and (x < knots_expanded[i + m]) else 0.0
        elif m > 1:
            if knots_expanded[i + m - 1] - knots_expanded[i] > 0:
                c1 = (x - knots_expanded[i]) / (knots_expanded[i + m - 1] - knots_expanded[i]) * B(x, i, m - 1)
            else:
                c1 = 0
            if knots_expanded[i + m] - knots_expanded[i + 1] > 0:
                c2 = (knots_expanded[i + m] - x) / (knots_expanded[i + m] - knots_expanded[i + 1]) * B(x, i + 1, m - 1)
            else:
                c2 = 0
            return c1 + c2
        else:
            raise ValueError('order cannot be negative!')

    func = (lambda x: B(x, ind, order))

    return func


def test_B_list():
    y = {}
    for order in [1, 2, 3, 4, 5]:
        knots_including_bounds = [0.0, 0.3, 0.5, 0.7, 1.0]
        x_list = np.linspace(start=0.0, stop=0.99, num=100)
        num_of_cols = len(knots_including_bounds) - 2 + order
        y[order] = pd.DataFrame(np.nan, index=x_list, columns=range(num_of_cols))
        for i in range(num_of_cols):
            func = basis_function(i, knots_including_bounds, order)
            y[order][i] = [func(x) for x in x_list]

    return y
