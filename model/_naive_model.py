import numpy as np

from scipy import stats

from postprocess import forecast
from ._model_helpers import boot_sd_residuals


def naive_model(
    self, h=1, ci=True, level=0.95, seasonal=False, bootstrap=False,
    n_samples=500
):
    """
    Returns an forecast object based on naive forecaster.
    """
    model = 'naive_model'
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
