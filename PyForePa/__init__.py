import numpy as np


class tseries(object):
    def __init__(self, start, end, time_unit, series=None, frequency=None):
        super(tseries, self).__init__()
        index = np.arange(start, end, dtype="datetime64[" + time_unit + "]")
        X = np.full(len(index), np.nan) if series is None else series
        dtypes = np.dtype([('index', index.dtype), ('X', X.dtype)])
        self.values = np.empty(len(index), dtype=dtypes)
        self.values['index'] = index
        self.values['X'] = X
        self.frequency = frequency

    @property
    def frequency(self):
        return self.__frequency

    @frequency.setter
    def frequency(self, v):
        """
        Validate frequency.
        """
        if v is not None:
            if float(v).is_integer() and float(v) > 0:
                self.__frequency = v
            else:
                raise ValueError("Frequency must be positive integer.")

        if v is not None:
            if (float(v) * 2) < len(self.values['index']):
                self.__frequency = v
            else:
                raise ValueError("Series has no or less than 2 periods.")

    from PyForePa.preprocess.impute import (
        impute_mean,
        impute_median,
        impute_random,
        impute_value,
        impute_locf,
        impute_nocb,
        impute_interpolation,
    )
    from PyForePa.preprocess.transform import (
        transform_square_root,
        transform_natural_log,
        transform_detrend,
    )
    from PyForePa.preprocess.plot import (
        plot_series,
        plot_acf,
        plot_pacf,
        plot_trend,
        plot_seasonality,
        plot_random,
        plot_series_decomposition,
        plot_nan_distribution
    )
    from PyForePa.preprocess.decompose import (
        decompose_trend,
        decompose_detrend,
        decompose_seasonality,
        decompose_remainder,
        decompose,
    )

    from PyForePa.preprocess.describe import (
        describe_acf,
        describe_pacf
    )


class forecast:
    def __init__(
        self, model_info=None, forecasts=None, series=None, series_info=None
    ):
        super(forecast, self).__init__()
        self.model_info = model_info
        self.forecasts = forecasts
        self.series = series
        self.series_info = series_info

    from PyForePa.postprocess.accuracy import (
        accuracy_me,
        accuracy_rmse,
        accuracy_mae,
        accuracy_mse,
        accuracy_mape,
        accuracy_smape,
        accuracy_mdae,
        accuracy_mmr
    )

    from PyForePa.postprocess.plot import (
        plot_forecast
    )


class drift_model(object):
    def __init__(self, tseries):
        super(drift_model, self).__init__()
        dtypes = np.dtype(
            [("index", tseries.values["index"].dtype), ("X", tseries.values["X"].dtype)])
        self.values = np.empty(len(tseries.values["index"]), dtype=dtypes)
        self.values["index"] = tseries.values["index"]
        self.values["X"] = tseries.values["X"]
        self.frequency = tseries.frequency

    from PyForePa.models.drift_model import (
        forecast
    )


class ema_model(object):
    def __init__(self, tseries):
        super(ema_model, self).__init__()
        dtypes = np.dtype(
            [("index", tseries.values["index"].dtype), ("X", tseries.values["X"].dtype)])
        self.values = np.empty(len(tseries.values["index"]), dtype=dtypes)
        self.values["index"] = tseries.values["index"]
        self.values["X"] = tseries.values["X"]
        self.frequency = tseries.frequency

    from PyForePa.models.ema_model import (
        forecast
    )


class mean_model(object):
    def __init__(self, tseries):
        super(mean_model, self).__init__()
        dtypes = np.dtype(
            [("index", tseries.values["index"].dtype), ("X", tseries.values["X"].dtype)])
        self.values = np.empty(len(tseries.values["index"]), dtype=dtypes)
        self.values["index"] = tseries.values["index"]
        self.values["X"] = tseries.values["X"]
        self.frequency = tseries.frequency

    from PyForePa.models.mean_model import (
        forecast
    )


class naive_model(object):
    def __init__(self, tseries):
        super(naive_model, self).__init__()
        dtypes = np.dtype(
            [("index", tseries.values["index"].dtype), ("X", tseries.values["X"].dtype)])
        self.values = np.empty(len(tseries.values["index"]), dtype=dtypes)
        self.values["index"] = tseries.values["index"]
        self.values["X"] = tseries.values["X"]
        self.frequency = tseries.frequency

    from PyForePa.models.naive_model import (
        forecast
    )


class random_model(object):
    def __init__(self, tseries):
        super(random_model, self).__init__()
        dtypes = np.dtype(
            [("index", tseries.values["index"].dtype), ("X", tseries.values["X"].dtype)])
        self.values = np.empty(len(tseries.values["index"]), dtype=dtypes)
        self.values["index"] = tseries.values["index"]
        self.values["X"] = tseries.values["X"]
        self.frequency = tseries.frequency

    from PyForePa.models.random_model import (
        forecast
    )


class sma_model(object):
    def __init__(self, tseries):
        super(sma_model, self).__init__()
        dtypes = np.dtype(
            [("index", tseries.values["index"].dtype), ("X", tseries.values["X"].dtype)])
        self.values = np.empty(len(tseries.values["index"]), dtype=dtypes)
        self.values["index"] = tseries.values["index"]
        self.values["X"] = tseries.values["X"]
        self.frequency = tseries.frequency

    from PyForePa.models.sma_model import (
        forecast
    )


class wma_model(object):
    def __init__(self, tseries):
        super(wma_model, self).__init__()
        dtypes = np.dtype(
            [("index", tseries.values["index"].dtype), ("X", tseries.values["X"].dtype)])
        self.values = np.empty(len(tseries.values["index"]), dtype=dtypes)
        self.values["index"] = tseries.values["index"]
        self.values["X"] = tseries.values["X"]
        self.frequency = tseries.frequency

    from PyForePa.models.wma_model import (
        forecast
    )
