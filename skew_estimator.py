import pandas as pd
import numpy as np

def est_skewness():
    data = np.random.normal(0, 1, size=1000000)
    data = pd.Series(data)
    df = {}
    for window in [60, 120, 260, 20 * 1440]:
        m4 = pd.rolling_sum(data ** 4, window=window, min_periods=window)
        m3 = pd.rolling_sum(data ** 3, window=window, min_periods=window)
        m2 = pd.rolling_sum(data ** 2, window=window, min_periods=window)
        m1 = pd.rolling_sum(data ** 1, window=window, min_periods=window)

        raw_skew = m3 / m2 ** 1.5
        adj_skew = (m3 - m4 / m2 * m1) / m2 ** 1.5

        df[window] = pd.Series({
            'Raw skew mean': raw_skew.mean(),
            'Adj skew mean': adj_skew.mean(),
            'Raw skew stdev': raw_skew.std(),
            'Adj skew stdev': adj_skew.std()
        })
        return df

