from preprocess import tseries
from model import model
from postprocess import forecast

import matplotlib.pyplot as plt
import numpy as np

from helpers._helpers import (
    acf, decompose_trend, decompose_detrend, decompose_seasonality,
    decompose_remainder
)


plt.style.use('bmh')

data = np.array([1, 2, 2, 3, 4, 5, 6, 5, np.nan, 7, 5, 9, 10, 12, 11, 9, 8, 11])
date = np.array(np.arange(len(data)))
#series = tseries(data, date).impute_linear_interp().transform_natural_log()
series = tseries(data, date, season=3).impute_linear_interp()

plt.show(series.plot_series_original())
plt.show(series.plot_series_transformed())
plt.show(series.plot_acf())
plt.show(series.plot_trend())
plt.show(series.plot_seasonality())
plt.show(series.plot_random())