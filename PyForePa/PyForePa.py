import numpy as np
from scipy import stats


def boot_sd_residuals(data, n_samples):
    """
    Returns bootstrapped standard deviation of the residuals.
    """
    sample_num = 1
    sd_residuals_array = np.empty([0, 1])

    while sample_num <= n_samples:
        sample = np.random.choice(data, len(data))
        residuals = np.diff(sample)
        residuals_sd = np.std(residuals)
        sd_residuals_array = np.vstack((sd_residuals_array, residuals_sd))
        sample_num += 1

    bootstrap_sd_residuals = np.mean(sd_residuals_array)

    return bootstrap_sd_residuals


class tseries:
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

    def square_root_transformation(self):
        """
        Returns square root transformed y.
        """
        self.y_transformed = np.sqrt(np.float64(self.y_transformed))

        return self

    def natural_log_transformation(self):
        """
        Returns log transformed y.
        """
        self.y_transformed = np.log(np.float64(self.y_transformed))

        return self


class model(tseries):
    def __init__(self, y_original, date_original, season, y_transformed):
        tseries.__init__(
            self, y_original, date_original, season, y_transformed
        )
        super(model, self).__init__()

    def mean_forecast(
        self, h=1, ci=True, level=0.95, bootstrap=False, n_samples=500
    ):
        """
        Returns a forecast object based on mean forecaster.
        """
        model = 'mean_forecast'
        y_train = self.y_transformed
        i = 1
        j = len(y_train)
        k = j + (h - 1)
        y_point = np.empty([0, 1])
        y_lb = np.empty([0, 1])
        y_ub = np.empty([0, 1])
        residuals = np.diff(y_train)

        if bootstrap is False:
            sd_residuals = np.std(residuals)
        else:
            sd_residuals = boot_sd_residuals(y_train, n_samples)

        while j <= k:
            pred = np.mean(y_train)
            y_point = np.vstack((y_point, pred))
            if ci is False:
                y_lb = np.vstack((y_lb, np.nan))
                y_ub = np.vstack((y_ub, np.nan))
            else:
                se_pred = sd_residuals * np.sqrt(i)
                t_crit = stats.t.ppf(q=level, df=(j - 1))
                pred_lb = pred - (t_crit * se_pred)
                pred_ub = pred + (t_crit * se_pred)
                y_lb = np.vstack((y_lb, pred_lb))
                y_ub = np.vstack((y_ub, pred_ub))
            y_train = np.append(y_train, pred)
            i += 1
            j += 1

        model_info = np.array(
            [(model, ci, level, h, bootstrap, n_samples)],
            dtype=[
                ('model', 'S20'), ('ci', 'S10'), ('level', np.float64),
                ('h', np.int8), ('bootstrap', 'S10'), ('n_samples', np.float64)
            ]
        )

        forecast_obj = forecast(
            model_info, y_point, y_lb, y_ub, residuals, self.y_original,
            self.date_original, self.season, self.y_transformed
        )

        return forecast_obj

    def random_forecast(
        self, h=1, ci=True, level=0.95, bootstrap=False, n_samples=500
    ):
        """
        Returns a forecast object based on random forecaster.
        """
        model = 'random_forecast'
        y_train = self.y_transformed
        i = 1
        j = len(y_train)
        k = j + (h - 1)
        y_point = np.empty([0, 1])
        y_lb = np.empty([0, 1])
        y_ub = np.empty([0, 1])
        residuals = np.diff(y_train)

        if bootstrap is False:
            sd_residuals = np.std(residuals)
        else:
            sd_residuals = boot_sd_residuals(y_train, n_samples)

        while j <= k:
            pred = np.random.choice(y_train)
            y_point = np.vstack((y_point, pred))
            if ci is False:
                y_lb = np.vstack((y_lb, np.nan))
                y_ub = np.vstack((y_ub, np.nan))
            else:
                se_pred = sd_residuals * np.sqrt(i)
                t_crit = stats.t.ppf(q=level, df=(j - 1))
                pred_lb = pred - (t_crit * se_pred)
                pred_ub = pred + (t_crit * se_pred)
                y_lb = np.vstack((y_lb, pred_lb))
                y_ub = np.vstack((y_ub, pred_ub))
            y_train = np.append(y_train, pred)
            i += 1
            j += 1

        model_info = np.array(
            [(model, ci, level, h, bootstrap, n_samples)],
            dtype=[
                ('model', 'S20'), ('ci', 'S10'), ('level', np.float64),
                ('h', np.int8), ('bootstrap', 'S10'), ('n_samples', np.float64)
            ]
        )

        forecast_obj = forecast(
            model_info, y_point, y_lb, y_ub, residuals, self.y_original,
            self.date_original, self.season, self.y_transformed
        )

        return forecast_obj

    def naive_forecast(
        self, h=1, ci=True, level=0.95, seasonal=False, bootstrap=False,
        n_samples=500
    ):
        """
        Returns an forecast object based on naive forecaster.
        """
        model = 'naive_forecast'
        y_train = self.y_transformed
        i = 1
        s = np.negative(self.season)
        j = len(y_train)
        k = j + (h - 1)
        y_point = np.empty([0, 1])
        y_lb = np.empty([0, 1])
        y_ub = np.empty([0, 1])
        residuals = np.diff(y_train)

        if bootstrap is False:
            sd_residuals = np.std(residuals)
        else:
            sd_residuals = boot_sd_residuals(y_train, n_samples)

        while j <= k:
            if seasonal is True:
                pred = y_train[s]
            else:
                pred = y_train[-1]
            y_point = np.vstack((y_point, pred))
            if ci is False:
                y_lb = np.vstack((y_lb, np.nan))
                y_ub = np.vstack((y_ub, np.nan))
            else:
                se_pred = sd_residuals * np.sqrt(i)
                t_crit = stats.t.ppf(q=level, df=(j - 1))
                pred_lb = pred - (t_crit * se_pred)
                pred_ub = pred + (t_crit * se_pred)
                y_lb = np.vstack((y_lb, pred_lb))
                y_ub = np.vstack((y_ub, pred_ub))
            y_train = np.append(y_train, pred)
            i += 1
            j += 1

        model_info = np.array(
            [(model, ci, level, h, seasonal, bootstrap, n_samples)],
            dtype=[
                ('model', 'S20'), ('ci', 'S10'), ('level', np.float64),
                ('h', np.int8), ('seasonal', np.int8), ('bootstrap', 'S10'),
                ('n_samples', np.float64)
            ]
        )

        forecast_obj = forecast(
            model_info, y_point, y_lb, y_ub, residuals, self.y_original,
            self.date_original, self.season, self.y_transformed
        )

        return forecast_obj

    def drift_forecast(
        self, h=1, ci=True, level=0.95, bootstrap=False, n_samples=500
    ):
        """
        Returns a forecast object based on drift forecaster.
        """
        model = 'drift_forecast'
        y_train = self.y_transformed
        i = 1
        j = len(y_train)
        k = j + (h - 1)
        y_point = np.empty([0, 1])
        y_lb = np.empty([0, 1])
        y_ub = np.empty([0, 1])
        residuals = np.diff(y_train)

        if bootstrap is False:
            sd_residuals = np.std(residuals)
        else:
            sd_residuals = boot_sd_residuals(y_train, n_samples)

        while j <= k:
            drift = (y_train[-1] - y_train[0]) / (j - 1)
            pred = y_train[-1] + drift
            y_point = np.vstack((y_point, pred))
            if ci is False:
                y_lb = np.vstack((y_lb, np.nan))
                y_ub = np.vstack((y_ub, np.nan))
            else:
                se_pred = sd_residuals * np.sqrt(i)
                t_crit = stats.t.ppf(q=level, df=(j - 2))
                pred_lb = pred - (t_crit * se_pred)
                pred_ub = pred + (t_crit * se_pred)
                y_lb = np.vstack((y_lb, pred_lb))
                y_ub = np.vstack((y_ub, pred_ub))
            y_train = np.append(y_train, pred)
            i += 1
            j += 1

        model_info = np.array(
            [(model, ci, level, h, bootstrap, n_samples)],
            dtype=[
                ('model', 'S20'), ('ci', 'S10'), ('level', np.float64),
                ('h', np.int8), ('bootstrap', 'S10'), ('n_samples', np.float64)
            ]
        )

        forecast_obj = forecast(
            model_info, y_point, y_lb, y_ub, residuals, self.y_original,
            self.date_original, self.season, self.y_transformed
        )

        return forecast_obj

    def sma_forecast(
        self, h=1, ci=True, level=0.95, n_periods=2, bootstrap=False,
        n_samples=500
    ):
        """
        Returns a forecast object based on simple moving average
        forecaster.
        """
        model = 'sma_forecast'
        y_train = self.y_transformed
        i = 1
        j = len(y_train)
        k = j + (h - 1)
        y_point = np.empty([0, 1])
        y_lb = np.empty([0, 1])
        y_ub = np.empty([0, 1])
        residuals = np.diff(y_train)

        if bootstrap is False:
            sd_residuals = np.std(residuals)
        else:
            sd_residuals = boot_sd_residuals(y_train, n_samples)

        while j <= k:
            pred = np.mean(y_train[-(np.absolute(n_periods)):])
            y_point = np.vstack((y_point, pred))
            if ci is False:
                y_lb = np.vstack((y_lb, np.nan))
                y_ub = np.vstack((y_ub, np.nan))
            else:
                se_pred = sd_residuals * np.sqrt(i)
                t_crit = stats.t.ppf(q=level, df=(j - 1))
                pred_lb = pred - (t_crit * se_pred)
                pred_ub = pred + (t_crit * se_pred)
                y_lb = np.vstack((y_lb, pred_lb))
                y_ub = np.vstack((y_ub, pred_ub))
            y_train = np.append(y_train, pred)
            i += 1
            j += 1

        model_info = np.array(
            [(model, ci, level, h, n_periods, bootstrap, n_samples)],
            dtype=[
                ('model', 'S20'), ('ci', 'S10'), ('level', np.float64),
                ('h', np.int8), ('n_periods', np.float64),
                ('bootstrap', 'S10'), ('n_samples', np.float64)
            ]
        )

        forecast_obj = forecast(
            model_info, y_point, y_lb, y_ub, residuals, self.y_original,
            self.date_original, self.season, self.y_transformed
        )

        return forecast_obj

    def ema_forecast(
        self, h=1, ci=True, level=0.95, n_periods=2, bootstrap=False,
        n_samples=500
    ):
        """
        Returns a forecast object based on exponential moving average
        forecaster.
        """
        model = 'ema_forecast'
        y_train = self.y_transformed
        i = 1
        j = len(y_train)
        k = j + (h - 1)
        y_point = np.empty([0, 1])
        y_lb = np.empty([0, 1])
        y_ub = np.empty([0, 1])
        residuals = np.diff(y_train)

        weights = np.exp(np.linspace(-1, 0, n_periods))
        weights /= np.sum(weights)

        if bootstrap is False:
            sd_residuals = np.std(residuals)
        else:
            sd_residuals = boot_sd_residuals(y_train, n_samples)

        while j <= k:
            pred = np.sum(
                y_train[np.negative(np.absolute(n_periods)):] * weights)
            y_point = np.vstack((y_point, pred))
            if ci is False:
                y_lb = np.vstack((y_lb, np.nan))
                y_ub = np.vstack((y_ub, np.nan))
            else:
                se_pred = sd_residuals * np.sqrt(i)
                t_crit = stats.t.ppf(q=level, df=(j - 1))
                pred_lb = pred - (t_crit * se_pred)
                pred_ub = pred + (t_crit * se_pred)
                y_lb = np.vstack((y_lb, pred_lb))
                y_ub = np.vstack((y_ub, pred_ub))
            y_train = np.append(y_train, pred)
            i += 1
            j += 1

        model_info = np.array(
            [(model, ci, level, h, n_periods, bootstrap, n_samples)],
            dtype=[
                ('model', 'S20'), ('ci', 'S10'), ('level', np.float64),
                ('h', np.int8), ('n_periods', np.float64),
                ('bootstrap', 'S10'), ('n_samples', np.float64)
            ]
        )

        forecast_obj = forecast(
            model_info, y_point, y_lb, y_ub, residuals, self.y_original,
            self.date_original, self.season, self.y_transformed
        )

        return forecast_obj

    def wma_forecast(
        self, h=1, ci=True, level=0.95, bootstrap=False, n_samples=500,
        weights=[0.5, 0.5]
    ):
        """
        Returns a forecast object based on weighted moving average
        forecaster.
        """
        model = 'wma_forecast'
        y_train = self.y_transformed
        i = 1
        j = len(y_train)
        k = j + (h - 1)
        y_point = np.empty([0, 1])
        y_lb = np.empty([0, 1])
        y_ub = np.empty([0, 1])
        residuals = np.diff(y_train)

        n_periods = len(weights)
        weights = np.array(weights)

        if bootstrap is False:
            sd_residuals = np.std(residuals)
        else:
            sd_residuals = boot_sd_residuals(y_train, n_samples)

        while j <= k:
            pred = np.sum(y_train[np.negative(n_periods):] * weights)
            y_point = np.vstack((y_point, pred))
            if ci is False:
                y_lb = np.vstack((y_lb, np.nan))
                y_ub = np.vstack((y_ub, np.nan))
            else:
                se_pred = sd_residuals * np.sqrt(i)
                t_crit = stats.t.ppf(q=level, df=(j - 1))
                pred_lb = pred - (t_crit * se_pred)
                pred_ub = pred + (t_crit * se_pred)
                y_lb = np.vstack((y_lb, pred_lb))
                y_ub = np.vstack((y_ub, pred_ub))
            y_train = np.append(y_train, pred)
            i += 1
            j += 1

        model_info = np.array(
            [(model, ci, level, h, n_periods, bootstrap, n_samples, weights)],
            dtype=[
                ('model', 'S20'), ('ci', 'S10'), ('level', np.float64),
                ('h', np.int8), ('n_periods', np.float64),
                ('bootstrap', 'S10'), ('n_samples', np.float64),
                ('weights', object)
            ]
        )

        forecast_obj = forecast(
            model_info, y_point, y_lb, y_ub, residuals, self.y_original,
            self.date_original, self.season, self.y_transformed
        )

        return forecast_obj


class forecast:
    def __init__(
        self, model_info=None, y_point=None, y_lb=None, y_ub=None,
        residuals=None, y_original=None, date_original=None, season=None,
        y_transformed=None
    ):
        super(forecast, self).__init__()
        self.model_info = model_info
        self.y_point = y_point
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.residuals = residuals
        self.y_original = y_original
        self.date_original = date_original
        self.season = season
        self.y_transformed = y_transformed

    def accuracy(self, y_true):
        """
        Returns structured Numpy array of accuracy measures.
        """

        if len(self.y_point) != len(y_true):
            raise Exception('Length of y_point and y_true must be the same.')
        else:
            pass

        me = np.mean(self.y_point - y_true)
        rmse = np.sqrt(np.mean((self.y_point - y_true) ** 2))
        mae = np.mean(np.absolute(self.y_point - y_true))
        mse = np.mean((self.y_point - y_true) ** 2)
        mape = np.mean(np.absolute(((self.y_point - y_true) / y_true) * 100))
        smape = np.mean(np.absolute(self.y_point - y_true) /
                        (self.y_point + y_true) * 200)
        mdae = np.median(np.absolute(self.y_point - y_true))
        mmr = mae / np.mean(self.y_point)

        accuracy_measures = np.array(
            [(me, rmse, mae, mse, mape, smape, mdae, mmr)],
            dtype=[
                ('ME', np.float64), ('RMSE', np.float64), ('MAE', np.float64),
                ('MSE', np.float64), ('MAPE', np.float64),
                ('SMAPE', np.float64), ('MDAE', np.float64),
                ('MMR', np.float64)
            ]
        )

        return accuracy_measures
