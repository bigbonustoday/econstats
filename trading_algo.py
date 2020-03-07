import pandas as pd
import numpy as np

def find_max_trade(prices):
    if not isinstance(prices, pd.Series):
        raise TypeError('this function only takes pd.Series')
    pl = pd.Series({ind: prices.loc[ind:].max() - prices.loc[:ind].min()
                    for ind in prices.index})
    return pl.max()


def find_max_pl_trade_list(prices):
    if not isinstance(prices, pd.Series):
        raise TypeError('this function only takes pd.Series')

    current_t = prices.first_valid_index()
    pl = 0
    while current_t < prices.last_valid_index():
        current_t, buy = find_local_maxmin(prices, start=current_t, sign=-1)
        current_t, sell = find_local_maxmin(prices, start=current_t, sign=1)
        pl += max(sell - buy, 0)
    return pl


def find_local_maxmin(prices, start, sign):
    for i in range(prices.index.get_loc(start), len(prices)):
        if i == 0:
            if np.sign(prices.iloc[i] - prices.iloc[i+1]) == sign:
                return prices.index[i], prices.iloc[i]
            else:
                continue
        if i == len(prices) - 1:
            if np.sign(prices.iloc[i] - prices.iloc[i-1]) == sign:
                return prices.index[i], prices.iloc[i]
            else:
                continue

        if (np.sign(prices.iloc[i] - prices.iloc[i+1]) == sign) and (np.sign(prices.iloc[i] - prices.iloc[i-1]) == sign):
            return prices.index[i], prices.iloc[i]
    return prices.last_valid_index(), None
