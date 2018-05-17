import numpy as np

def accuracy_me(self, y_true):
    """
    Returns mean error of forecast.
    """
    error = self.y_point - y_true
    me = np.mean(error)

    return me

def accuracy_rmse(self, y_true):
    """
    Returns root_mean_squared_error of forecast.
    """
    error = self.y_point - y_true
    rmse = np.sqrt(np.mean((error) ** 2))

    return rmse

def accuracy_mae(self, y_true):
    """
    Returns mean absolute error of forecast.
    """
    error = self.y_point - y_true
    mae = np.mean(np.absolute(error))

    return mae

def accuracy_mse(self, y_true):
    """
    Returns mean squared error of forecast.
    """
    error = self.y_point - y_true
    mse = np.mean((error) ** 2)

    return mse

def accuracy_mape(self, y_true):
    """
    Returns mean absolute percentage error of forecast.
    """
    error = self.y_point - y_true
    mape = np.mean(np.absolute(((error) / y_true) * 100))

    return mape

def accuracy_smape(self, y_true):
    """
    Returns symmetric mean absolute percentage error of forecast.
    """
    error = self.y_point - y_true
    smape = np.mean(np.absolute(error) / (self.y_point + y_true) * 200)

    return smape

def accuracy_mdae(self, y_true):
    """
    Returns median absolute error of forecast.
    """
    error = self.y_point - y_true
    mdae = np.median(np.absolute(error))

    return mdae

def accuracy_mmr(self, y_true):
    """
    Returns mad mean ratio of forecast.
    """
    error = self.y_point - y_true
    mmr = np.mean(np.absolute(error)) / np.mean(self.y_point)

    return mmr

def accuracy(self, y_true, how):
    """
    Returns array of accuracy measurements in the order they are listed
    in measure argument.
    """

    if len(self.y_point) != len(y_true):
        raise Exception('Length of y_point and y_true must be the same.')
    else:
        measure_dict = {
            'ME': accuracy_me,
            'RMSE': accuracy_rmse,
            'MAE': accuracy_mae,
            'MSE': accuracy_mse,
            'MAPE': accuracy_mape,
            'SMAPE': accuracy_smape,
            'MDAE': accuracy_mdae,
            'MMR': accuracy_mmr
        }

        measurements = np.empty([0, 1])
        for i in how:
            measurement = measure_dict[i](self, y_true)
            measurements = np.vstack((measurements, measurement))

        return measurements
