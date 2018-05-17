import numpy as np


def impute_mean(self):
    """
    Returns tseries object with missing values of the series filled
    with the mean up to that point.
    """
    for idx, value in enumerate(self.y_transformed, 0):
        if np.isnan(value) == True:
            self.y_transformed[idx] = np.mean(self.y_transformed[:idx])

    return self


def impute_median(self):
    """
    Returns tseries object with missing values of the series filled
    with the median up to that point.
    """
    for idx, value in enumerate(self.y_transformed, 0):
        if np.isnan(value) == True:
            self.y_transformed[idx] = np.median(
                self.y_transformed[:idx])

    return self


def impute_random(self):
    """
    Returns tseries object with missing values of the series filled
    with a random number from the series up to that point.
    """
    for idx, value in enumerate(self.y_transformed, 0):
        if np.isnan(value) == True:
            self.y_transformed[idx] = np.random.choice(
                self.y_transformed[:idx])

    return self


def impute_value(self, replacement):
    """
    Returns tseries object with missing values of the series filled
    with a specific value.
    """
    for idx, value in enumerate(self.y_transformed, 0):
        if np.isnan(value) == True:
            self.y_transformed[idx] = replacement

    return self


def impute_locf(self):
    """
    Returns tseries object with missing values of the series filled
    with the most recent non-missing value.
    """
    for idx, value in enumerate(self.y_transformed, 0):
        if np.isnan(value) == True:
            self.y_transformed[idx] = self.y_transformed[:idx][0]

    return self


def impute_nocb(self):
    """
    Returns tseries object with missing values of the series filled
    with the next non-missing value.
    """
    for idx, value in enumerate(self.y_transformed[::-1], 0):
        if np.isnan(value) == True:
            self.y_transformed[::-1][idx] = (
                self.y_transformed[::-1][:idx][-1]
            )

    return self


def impute_linear_interp(self):
    """
    Returns tseries object with missing values of the series filled
    via linear interpolation.
    """
    mask = np.logical_not(np.isnan(self.y_transformed))
    self.y_transformed = np.interp(
        np.arange(len(self.y_transformed)),
        np.arange(len(self.y_transformed))[mask],
        self.y_transformed[mask]
    )

    return self


def impute(self, how, replacement=None):
    """
    Returns tseries object with missing values of the series filled by
    the method specified in the "how" argument.
    """
    imputation_dict = {
        'mean': impute_mean,
        'median': impute_median,
        'random': impute_random,
        'value': impute_value,
        'locf': impute_locf,
        'nocb': impute_nocb,
        'linear_interpolation': impute_linear_interp
    }

    tseries_obj = imputation_dict[how](self, replacement)

    return tseries_obj
