import numpy as np

from preprocess import tseries


class model(tseries):
	def __init__(self, y_original, date_original, season, y_transformed):
	    tseries.__init__(
	        self, y_original, date_original, season, y_transformed
	    )
	    super(model, self).__init__()

	from ._mean_model import mean_model
	from ._random_model import random_model
	from ._naive_model import naive_model
	from ._drift_model import drift_model
	from ._sma_model import sma_model
	from ._ema_model import ema_model
	from ._wma_model import wma_model