import numpy as np
from scipy import stats


class PyForePa:
    def __init__(self, y, season=1):
        """
        Creates a PyForePa object.
        """
        self.y = np.array(y)
        self.season = season

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
            raise ValueError(f'Season must be positive integer.')

    def mean_forecast(self, h=1, ci=True, level=0.95):
        """
        Returns a PyForePa.forecast object that holds:
        1. predicted value of y based on mean forecaster
        2. lower bound of ci
        3. upper bound of ci
        """
        j = 1
        y_train = self.y
        preds = np.empty([0, 3])
        sd_diff = np.std(np.diff(y_train))
        while j <= h:
            pred = np.mean(y_train)
            if ci is False:
                result = np.array([pred, None, None]).reshape((1, 3))
                preds = np.vstack((preds, result))
            else:
                se_pred = sd_diff * np.sqrt(j)
                t_crit = stats.t.ppf(q=level, df=(len(y_train) - 1))
                lb_pred = pred - (t_crit * se_pred)
                ub_pred = pred + (t_crit * se_pred)
                result = np.array([pred, lb_pred, ub_pred]).reshape((1, 3))
                preds = np.vstack((preds, result))
            y_train = np.append(y_train, pred)
            j += 1

        preds = PyForePa.forecast(preds)

        return preds

    def random_forecast(self, h=1, ci=True, level=0.95):
        """
        Returns a PyForePa.forecast object that holds:
        1. predicted value of y based on random forecaster
        2. lower bound of ci
        3. upper bound of ci
        """
        j = 1
        y_train = self.y
        preds = np.empty([0, 3])
        sd_diff = np.std(np.diff(y_train))
        while j <= h:
            pred = np.random.choice(y_train)
            if ci is False:
                result = np.array([pred, None, None]).reshape((1, 3))
                preds = np.vstack((preds, result))
            else:
                se_pred = sd_diff * np.sqrt(j)
                t_crit = stats.t.ppf(q=level, df=(len(y_train) - 1))
                lb_pred = pred - (t_crit * se_pred)
                ub_pred = pred + (t_crit * se_pred)
                result = np.array([pred, lb_pred, ub_pred]).reshape((1, 3))
                preds = np.vstack((preds, result))
            y_train = np.append(y_train, pred)
            j += 1

        preds = PyForePa.forecast(preds)

        return preds

    def naive_forecast(self, h=1, ci=True, level=0.95, seasonal=False):
        """
        Returns a PyForePa.forecast object that holds:
        1. predicted value of y based on naive forecaster
        2. lower bound of ci
        3. upper bound of ci
        """
        s = np.negative(self.season)
        j = 1
        y_train = self.y
        preds = np.empty([0, 3])
        rmse_diff = np.sqrt(np.mean(np.diff(y_train)**2))
        while j <= h:
            if seasonal is True:
                pred = y_train[s]
            else:
                pred = y_train[-1]
            if ci is False:
                preds = np.append(preds, pred)
            else:
                se_pred = rmse_diff * np.sqrt(j)
                t_crit = stats.t.ppf(q=level, df=(len(y_train) - 1))
                lb_pred = pred - (t_crit * se_pred)
                ub_pred = pred + (t_crit * se_pred)
                result = np.array([pred, lb_pred, ub_pred]).reshape((1, 3))
                preds = np.vstack((preds, result))
            y_train = np.append(y_train, pred)
            j += 1

        preds = PyForePa.forecast(preds)

        return preds

    def drift_forecast(self, h=1, ci=True, level=0.95):
        """
        Returns a PyForePa.forecast object that holds:
        1. predicted value of y based on drift forecaster
        2. lower bound of ci
        3. upper bound of ci
        """
        j = 1
        y_train = self.y
        preds = np.empty([0, 3])
        sd_diff = np.std(np.diff(y_train))
        while j <= h:
            drift = (y_train[-1] - y_train[0]) / (len(y_train) - 1)
            pred = y_train[-1] + drift
            if ci is False:
                result = np.array([pred, None, None]).reshape((1, 3))
                preds = np.vstack((preds, result))
            else:
                se_pred = sd_diff * np.sqrt(j)
                t_crit = stats.t.ppf(q=level, df=(len(y_train) - 2))
                lb_pred = pred - (t_crit * se_pred)
                ub_pred = pred + (t_crit * se_pred)
                result = np.array([pred, lb_pred, ub_pred]).reshape((1, 3))
                preds = np.vstack((preds, result))
            y_train = np.append(y_train, pred)
            j += 1

        preds = PyForePa.forecast(preds)

        return preds

    def sma_forecast(self, n_periods, h=1, ci=True, level=0.95):
        """
        Returns a PyForePa.forecast object that holds:
        1. predicted value of y based on simple moving average forecaster
        2. lower bound of ci
        3. upper bound of ci
        """
        j = 1
        y_train = self.y
        preds = np.empty([0, 3])
        sd_diff = np.std(np.diff(y_train))
        while j <= h:
            pred = np.mean(y_train[-(np.absolute(n_periods)):])
            if ci is False:
                result = np.array([pred, None, None]).reshape((1, 3))
                preds = np.vstack((preds, result))
            else:
                se_pred = sd_diff * np.sqrt(j)
                t_crit = stats.t.ppf(q=level, df=(len(y_train) - 1))
                lb_pred = pred - (t_crit * se_pred)
                ub_pred = pred + (t_crit * se_pred)
                result = np.array([pred, lb_pred, ub_pred]).reshape((1, 3))
                preds = np.vstack((preds, result))
            y_train = np.append(y_train, pred)
            j += 1

        preds = PyForePa.forecast(preds)

        return preds

    class forecast:
        def __init__(self, y_pred):
            """
            Creates a forecast object.
            """
            self.y_pred = y_pred[:, 0]
            self.y_pred_lb = y_pred[:, 1]
            self.y_pred_ub = y_pred[:, 2]

        def mean_error(self, y_true):
            """
            Returns mean error of forecast.
            """
            me = np.mean(self.y_pred - y_true)

            return me

        def root_mean_squared_error(self, y_true):
            """
            Returns root mean squared error of forecast.
            """
            rmse = np.sqrt(((self.y_pred - y_true) ** 2).mean())

            return rmse

        def mean_absolute_error(self, y_true):
            """
            Returns mean absolute error of forecast.
            """
            mae = np.mean(np.absolute(self.y_pred - y_true))

            return mae

        def mean_squared_error(self, y_true):
            """
            Returns mean squared error of forecast.
            """
            mse = np.mean((self.y_pred - y_true) ** 2)

            return mse
