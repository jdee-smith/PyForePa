from PyForePa import tseries
from PyForePa.models import model
from PyForePa.postprocess import forecast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from helpers._helpers import (
    acf_corr, pacf_ols, trend, detrend, seasonality, remainder)


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
#data = data.loc[(data.index >= '2013-01-31') & (data.index <= '2016-12-31')]
y = np.array(data['fremont_bridge_nb'], dtype=np.float64)
y[[30, 31]] = np.nan
x = np.array(data.index)

'''

y = y2.loc[(y2.index >= '2013-01-31') & (y2.index <= '2016-12-31')]

data = np.array([1, 2, 2, 3, 4, 5, 6, 5, np.nan,
                 7, 5, 9, 10, 12, 11, 9, 8, 11])
date = np.array(np.arange(len(data)))
'''

#series = tseries(data, date).impute_linear_interp().transform_natural_log()
series = tseries(y, x, season=12)
series.impute_mean(mode='split').y_transformed
#series.y_original[30] = np.nan
#series = series.impute_seasonal_decomposition(how="mean")
series2 = series.impute_seasonal_split(how="mean", order=12)
np.column_stack((series2.y_original, series2.y_transformed))

series.transform_detrend().y_transformed

pacf(y)


trend = series.decompose_trend()
detrend = series.decompose_detrend(
    order=12, center=True, model='multiplicative')
seasonality = series.decompose_seasonality()
random = series.decompose_remainder()
decomposition = series.decompose(model='multiplicative')

plt.show(series.plot_series_original())
plt.show(series.plot_series_transformed())
plt.show(series.plot_acf())
plt.show(series.plot_pacf())
plt.show(series.plot_trend(overlay=True))
plt.show(series.plot_seasonality())
plt.show(series.plot_random())
plt.show(series.plot_series_decomposition(model='multiplicative'))
plt.show(series.plot_nan_distribution())

# 1949 1950 1951 1952 1953 1954 1955 1956 1957 1958 1959 1960
series = [112, 115, 145, 171, 196, 204, 242, 284, 315, 340, 360, 417,
          118, 126, 150, 180, 196, 188, 233, 277, 301, 318, 342, 391,
          132, 141, 178, 193, 236, 235, 267, 317, 356, 362, 406, 419,
          129, 135, 163, 181, 235, 227, 269, 313, 348, 348, 396, 461,
          121, 125, 172, 183, 229, 234, 270, 318, 355, 363, 420, 472,
          135, 149, 178, 218, 243, 264, 315, 374, 422, 435, 472, 535,
          148, 170, 199, 230, 264, 302, 364, 413, 465, 491, 548, 622,
          148, 170, 199, 242, 272, 293, 347, 405, 467, 505, 559, 606,
          136, 158, 184, 209, 237, 259, 312, 355, 404, 404, 463, 508,
          119, 133, 162, 191, 211, 229, 274, 306, 347, 359, 407, 461,
          104, 114, 146, 172, 180, 203, 237, 271, 305, 310, 362, 390,
          118, 140, 166, 194, 201, 229, 278, 306, 336, 337, 405, 432]
import itertools
import datetime
from dateutil.relativedelta import relativedelta
y = []
for i in range(12):
    y.append(series[i::12])
y = list(itertools.chain.from_iterable(y))
nob_1 = datetime.datetime(1949, 1, 1)
#x = np.array([nob_1 + relativedelta(months=+i) for i in range(len(y))])
x = np.arange('1949-01', '1961-01', dtype='datetime64[M]')

# Source:
# Hyndman, R.J., Time Series Data Library,
# http://www-personal.buseco.monash.edu.au/~hyndman/TSDL/.
# Copied in October, 2005.
