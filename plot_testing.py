from preprocess import tseries
from model import model
from postprocess import forecast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from helpers._helpers import (acf, trend, detrend, seasonality, remainder)


plt.style.use('seaborn-muted')

data = (
    pd.read_csv(
        'https://data.seattle.gov/resource/4xy5-26gy.csv',
        parse_dates=True,
        index_col='date'
    )
    .resample('1M')
    .sum()
)
data = data.loc[(data.index >= '2013-01-31') & (data.index <= '2016-12-31')]
y = np.array(data['fremont_bridge_nb'])
x = np.array(data.index)
y[0] = -1
'''

y = y2.loc[(y2.index >= '2013-01-31') & (y2.index <= '2016-12-31')]

data = np.array([1, 2, 2, 3, 4, 5, 6, 5, np.nan,
                 7, 5, 9, 10, 12, 11, 9, 8, 11])
date = np.array(np.arange(len(data)))
'''

#series = tseries(data, date).impute_linear_interp().transform_natural_log()
series = tseries(y, x, season=12).impute_linear_interp()

series.transform_square_root().y_transformed


trend = series.decompose_trend()
detrend = series.decompose_detrend()
seasonality = series.decompose_seasonality()
random = series.decompose_remainder()
decomposition = series.decompose()

plt.show(series.plot_series_original())
plt.show(series.plot_series_transformed())
plt.show(series.plot_acf())
plt.show(series.plot_trend(overlay=True))
plt.show(series.plot_seasonality())
plt.show(series.plot_random())
plt.show(series.plot_series_decomposition())
