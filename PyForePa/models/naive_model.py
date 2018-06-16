import numpy as np

from scipy import stats

from PyForePa import forecast as fore_obj
from PyForePa.helpers.helpers import boot_sd_residuals


def forecast(
    self, h=1, ci=True, level=0.95, seasonal=False, bootstrap=False, n_samples=500
):
    """
    Returns an forecast object based on naive forecaster.
    """
    model = "naive_model"
    y_train = self.values["X"]
    i = 1
    s = np.negative(self.frequency)
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

    dtypes = np.dtype(
        [("lower", y_lb.dtype), ("point", y_point.dtype), ("upper", y_ub.dtype)]
    )

    forecasts = np.empty(len(y_point), dtype=dtypes)
    forecasts["lower"] = y_lb.reshape(len(y_lb))
    forecasts["point"] = y_point.reshape(len(y_point))
    forecasts["upper"] = y_ub.reshape(len(y_ub))

    model_info = np.array(
        [(model, ci, level, h, bootstrap, n_samples)],
        dtype=[
            ("model", "S20"),
            ("ci", "S10"),
            ("level", np.float64),
            ("h", np.int8),
            ("bootstrap", "S10"),
            ("n_samples", np.float64),
        ],
    )

    series_info = np.array([(self.frequency)], dtype=[("frequency", np.float64)])

    forecast_obj = fore_obj(model_info, forecasts, self.values, series_info)

    return forecast_obj
