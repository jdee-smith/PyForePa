import numpy as np

from PyForePa.helpers.helpers import (
    mean_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    median_absolute_error,
    mad_mean_ratio,
)


def accuracy_me(self, y_true):
    """
    Returns mean error of forecast.
    """
    y_point = self.forecasts["point"]
    me = mean_error(y_point, y_true)

    return me


def accuracy_rmse(self, y_true):
    """
    Returns root_mean_squared_error of forecast.
    """
    y_point = self.forecasts["point"]
    rmse = root_mean_squared_error(y_point, y_true)

    return rmse


def accuracy_mae(self, y_true):
    """
    Returns mean absolute error of forecast.
    """
    y_point = self.forecasts["point"]
    mae = mean_absolute_error(y_point, y_true)

    return mae


def accuracy_mse(self, y_true):
    """
    Returns mean squared error of forecast.
    """
    y_point = self.forecasts["point"]
    mse = mean_squared_error(y_point, y_true)

    return mse


def accuracy_mape(self, y_true):
    """
    Returns mean absolute percentage error of forecast.
    """
    y_point = self.forecasts["point"]
    mape = mean_absolute_percentage_error(y_point, y_true)

    return mape


def accuracy_smape(self, y_true):
    """
    Returns symmetric mean absolute percentage error of forecast.
    """
    y_point = self.forecasts["point"]
    smape = symmetric_mean_absolute_percentage_error(y_point, y_true)

    return smape


def accuracy_mdae(self, y_true):
    """
    Returns median absolute error of forecast.
    """
    y_point = self.forecasts["point"]
    mdae = median_absolute_error(y_point, y_true)

    return mdae


def accuracy_mmr(self, y_true):
    """
    Returns mad mean ratio of forecast.
    """
    y_point = self.forecasts["point"]
    mmr = mad_mean_ratio(y_point, y_true)

    return mmr


'''
def accuracy(self, y_true, how):
    """
    Returns array of accuracy measurements in the order they are listed
    in measure argument.
    """
    measure_dict = {
        "ME": accuracy_me,
        "RMSE": accuracy_rmse,
        "MAE": accuracy_mae,
        "MSE": accuracy_mse,
        "MAPE": accuracy_mape,
        "SMAPE": accuracy_smape,
        "MDAE": accuracy_mdae,
        "MMR": accuracy_mmr,
    }

    measurements = np.empty([0, 1])
    for i in how:
        measurement = measure_dict[i](self, y_true)
        measurements = np.vstack((measurements, measurement))

    return measurements
'''
