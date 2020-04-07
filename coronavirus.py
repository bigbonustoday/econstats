import pandas as pd
import datetime
from numpy import nan
import numpy as np
import matplotlib. pyplot as plt
import matplotlib.dates as mdates


DATA_URL = 'http://covidtracking.com/api/states/daily.csv'


def load():
    df = pd.read_csv(DATA_URL)
    df.loc[:, 'date'] = df.loc[:, 'date'].apply(str).apply(datetime.datetime.strptime, args=('%Y%m%d',))
    df.loc[:, 'dateChecked'] = df.loc[:, 'dateChecked'].apply(pd.Timestamp).dt.date
    df.loc[:, 'total'] = df.loc[:, 'total'].astype(float)
    return df


def plot_total(usa_total, region_label):
    print('===Latest snapshot for ' + region_label + ' as of ' + usa_total.last_valid_index().strftime('%Y-%m-%d') +
          '===')
    snapshot = usa_total.iloc[-1, :]
    print('Total positive = ' + str(snapshot['positive']))
    print('Total tested = ' + str(snapshot['posNeg']))
    print('Death rate = ' + str(round(snapshot['death'] / snapshot['positive'], 3)))
    print('Hospitalization rate = ' + str(round(snapshot['hospitalizedCumulative'] / snapshot['positive'], 3)))
    print('ICU rate = ' + str(round(snapshot['inIcuCumulative'] / snapshot['positive'], 3)))
    print('Ventilator rate = ' + str(round(snapshot['onVentilatorCumulative'] / snapshot['positive'], 3)))

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 24))
    date_format = mdates.DateFormatter('%m-%d')

    # plot total case count
    ax[0].plot((usa_total['positive']))
    ax[0].set_yscale('log')
    ax[0].set_title(region_label + ' total positives')

    # plot daily new case count
    ax[1].plot(usa_total['positive'].diff(1))
    ax[1].set_yscale('log')
    ax[1].set_title(region_label + ' new cases')

    # plot daily new tests
    ax[2].plot(usa_total['total'].diff(1))
    ax[2].set_yscale('log')
    ax[2].set_title(region_label + ' new tests')

    # plot daily positive rate
    ax[3].plot(usa_total['positive'].diff(1) / usa_total['total'].diff(1))
    ax[3].set_title(region_label + ' new positive rate')

    for i in range(3):
        ax[i].xaxis.set_major_formatter(date_format)

    fig.show()
    return


def main():
    df = load()
    usa_total = df.groupby('date').sum().replace(0.0, nan)
    ny_total = df[df.state == 'NY'].groupby('date').sum().replace(0.0, nan)
    ca_total = df[df.state == 'CA'].groupby('date').sum().replace(0.0, nan)
    plot_total(usa_total, 'USA')
    plot_total(ny_total, 'NY State')
    plot_total(ca_total, 'CA State')


if __name__ == "__main__":
    main()
