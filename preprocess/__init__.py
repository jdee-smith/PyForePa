import numpy as np


class tseries(object):
    def __init__(self, y, date, season=1):
        super(tseries, self).__init__()
        self.y_original = np.array(y, dtype=np.float64)
        self.date_original = np.array(date)
        self.season = season
        self.y_transformed = np.array(y, dtype=np.float64)

    @property
    def date(self):
        return self.__date

    @date.setter
    def date(self, v):
        """
        Validate that date is datetime object.
        """
        if np.issubdtype(v.dtype, np.datetime64):
            self.__date = v
        else:
            raise ValueError('Date must be np.datetime64 object.')

    @property
    def season(self):
        return self.__season

    @season.setter
    def season(self, v):
        """
        Validate that season is an integer greater than 0.
        """
        if float(v).is_integer() and float(v) > 0:
            self.__season = v
        else:
            raise ValueError('Season must be positive integer.')

    from ._impute import (
        impute_mean, impute_median, impute_random, impute_value, impute_locf,
        impute_nocb, impute_linear_interp, impute
    )
    from ._transform import transform_square_root, transform_natural_log
    from ._plot import (
        plot_series_original, plot_series_transformed, plot_acf, plot_trend,
        plot_seasonality, plot_random
    )
