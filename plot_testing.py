from preprocess import tseries
from model import model
from postprocess import forecast

import matplotlib.pyplot as plt
import numpy as np

from helpers._helpers import acf


plt.style.use('bmh')

data = np.array([1, 2, 2, 3, 4, 5, 6, 5, np.nan, 7, 5, 9, 10])
date = np.array(np.arange(len(data)))
#series = tseries(data, date).impute_linear_interp().transform_natural_log()
series = tseries(data, date).impute_linear_interp()

plt.show(series.plot_series())
plt.show(series.plot_series_original())
plt.show(series.plot_series_transformed())

acf(series.y_transformed)