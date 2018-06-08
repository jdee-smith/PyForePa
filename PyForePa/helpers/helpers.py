import numpy as np

from scipy import stats
from scipy.linalg import toeplitz


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


def acf_corr(data, max_lags="default", ci=True, level=0.95):
    """
    Returns autocorrelation coefficients and their bounds
    of length max_lags.
    """
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    if max_lags is "default":
        max_lags = int(10 * np.log10(n))
    else:
        max_lags = int(max_lags)

    def corr(h):
        acf_coeff = np.sum(
            ((data[: n - h] - mean) * (data[h:] - mean))) / n / c0
        return acf_coeff

    t_crit = stats.t.ppf(q=level, df=(n - 3))
    acf_coeff_lb = np.negative(t_crit) / np.sqrt(n)
    acf_coeff_ub = t_crit / np.sqrt(n)

    acf_coeffs = np.empty([0, 3])
    for k in np.arange(max_lags):
        acf_coeff = corr(k)
        if ci is False:
            acf_coeffs = np.vstack((acf_coeffs, (np.nan, acf_coeff, np.nan)))
        else:
            acf_coeffs = np.vstack(
                (acf_coeffs, (acf_coeff_lb, acf_coeff, acf_coeff_ub))
            )

    return acf_coeffs


def pacf_ols(data, max_lags="default", ci=True, level=0.95):
    """
    Returns partial autocorrelation coefficients estimated via OLS along with
    their bounds of length max_lags.
    """
    n = len(data)
    x0 = data
    #x0 = data[:, ]

    if max_lags is "default":
        max_lags = int(10 * np.log10(n))
    else:
        max_lags = int(max_lags)

    xlags = np.ones((n, max_lags))
    for i in range(1, max_lags):
        xlags[:, i] = np.roll(data, i)

    xlags[np.triu_indices(xlags.shape[1], 1)] = 0

    t_crit = stats.t.ppf(q=level, df=(n - 3))
    pacf_coeff_lb = np.negative(t_crit) / np.sqrt(n)
    pacf_coeff_ub = t_crit / np.sqrt(n)

    pacf_coeffs = np.empty([0, 3])
    for k in range(1, max_lags + 1):
        pacf_coeff = np.linalg.lstsq(xlags[k:, : k + 1], x0[k:])[0][-1]
        if ci is False:
            pacf_coeffs = np.vstack(
                (pacf_coeffs, (np.nan, pacf_coeff, np.nan)))
        else:
            pacf_coeffs = np.vstack(
                (pacf_coeffs, (pacf_coeff_lb, pacf_coeff, pacf_coeff_ub))
            )

    return pacf_coeffs


def yule_walker(data, order, method="unbiased", demean=True):
    """
    Returns partial autocorrelation coefficients obtained via Yule-Walker
    equations. Code mostly from statsmodels.
    """
    n = len(data)

    if demean is True:
        data = data - np.mean(data)
    else:
        pass

    if method == "unbiased":

        def denom(k):
            return n - k

    else:

        def denom(k):
            return n

    r = np.zeros(order + 1)
    r[0] = np.sum(data ** 2) / denom(0)

    for k in range(1, order + 1):
        r[k] = np.sum(data[0:-k] * data[k:]) / denom(k)

    R = toeplitz(r[:-1])
    rho = np.linalg.solve(R, r[1:])

    return rho


def pacf_yule_walker(data, max_lags="default", method="unbiased", ci=True, level=0.95):
    """
    Returns autocorrelation coefficients estimated via Yule-Walker equations
    and their bounds of length max_lags.
    """
    n = len(data)

    if max_lags is "default":
        max_lags = int(10 * np.log10(n))
    else:
        max_lags = int(max_lags)

    t_crit = stats.t.ppf(q=level, df=(n - 3))
    pacf_coeff_lb = np.negative(t_crit) / np.sqrt(n)
    pacf_coeff_ub = t_crit / np.sqrt(n)

    pacf_coeffs = np.empty([0, 3])
    for k in range(1, max_lags + 1):
        pacf_coeff = yule_walker(data, order=k, method=method, demean=True)[-1]
        if ci is False:
            pacf_coeffs = np.vstack(
                (pacf_coeffs, (np.nan, pacf_coeff, np.nan)))
        else:
            pacf_coeffs = np.vstack(
                (pacf_coeffs, (pacf_coeff_lb, pacf_coeff, pacf_coeff_ub))
            )

    return pacf_coeffs


def trend(data, order, center=True):
    """
    Returns array consisting of series trend.
    """
    even_order = order % 2 == 0
    trends = np.empty([0, 1])
    k = len(data)
    a = int(order / 2)

    if center is False:
        if even_order is True:
            b = int(a - 1)
        else:
            b = a
        trends = np.convolve(
            data.reshape((k,)), np.ones((order,)) / order, mode="valid"
        )
        trends = np.pad(trends, (b, a), "constant", constant_values=(np.nan,)).reshape(
            k, 1
        )

    else:
        j = order
        for i in np.arange(k):
            multiplier = 1 / order
            if even_order is True:
                w1 = multiplier * np.sum(data[i:j])
                w2 = multiplier * np.sum(data[i + 1: j + 1])
                trend = np.mean((w1, w2))
                trends = np.vstack((trends, trend))
            else:
                b = a
                trends = np.convolve(
                    data.reshape((k,)), np.ones((order,)) / order, mode="valid"
                )
                trends = np.pad(
                    trends, (b, a), "constant", constant_values=(np.nan,)
                ).reshape(k, 1)

            j += 1

        pad = int(order / 2)

        if order % 2 == 0:
            trends = np.roll(trends, pad)
        else:
            pass

        trends[:pad, ] = np.nan
        trends[-pad:, ] = np.nan

    return trends


def detrend(data, order, center=True, model="additive"):
    """
    Returns array of detrended series.
    """
    k = len(data)

    if model == "additive":
        data_detrended = data.reshape(k, 1) - trend(data, order, center)
    elif model == "multiplicative":
        data_detrended = data.reshape(k, 1) / trend(data, order, center)
    else:
        raise ValueError("Model must be additive or multiplicative.")

    return data_detrended


def seasonality(data, order, center=True, model="additive", median=False):
    """
    Returns array of series seasonality.
    """
    j = len(data)
    k = int(j / order)

    if j < (order * 2):
        raise ValueError("Series has no or less than 2 periods.")
    else:
        pass

    de_series = detrend(data, order, center, model)

    if median is False:
        arr1 = np.nanmean(np.resize(de_series, (k, order)), axis=0)
        arr2 = np.resize(arr1, (j, 1))
    else:
        arr1 = np.nanmedian(np.resize(de_series, (k, order)), axis=0)
        arr2 = np.resize(arr1, (j, 1))

    return arr2


def remainder(data, order, center=True, model="additive", median=False):
    """
    Returns array of left behind random noise.
    """
    k = len(data)
    trends = trend(data, order, center)
    avg_seasonality = seasonality(data, order, center, model, median)

    if model == "additive":
        remainder = data.reshape(k, 1) - trends - avg_seasonality
    elif model == "multiplicative":
        remainder = data.reshape(k, 1) / (trends * avg_seasonality)
    else:
        raise ValueError("Model must be additive or multiplicative.")

    return remainder


def nan_mean(data, trailing=True):
    """
    Fills missing values with mean of series up to that point.
    """
    for idx, value in enumerate(data, 0):
        if np.isnan(value):
            if trailing == True:
                data[idx] = np.mean(data[:idx])
            else:
                data[idx] = np.nanmean(data)

    return data


def nan_median(data, trailing=True):
    """
    Fills missing values with median of series up to that point.
    """
    for idx, value in enumerate(data, 0):
        if np.isnan(value):
            if trailing == True:
                data[idx] = np.median(data[:idx])
            else:
                data[idx] = np.nanmedian(data)

    return data


def nan_random(data, trailing=True):
    """
    Fills missing values with a random number from the series
    up to that point.
    """
    for idx, value in enumerate(data, 0):
        if np.isnan(value):
            if trailing == True:
                data[idx] = np.random.choice(data[:idx])
            else:
                data[idx] = np.random.choice(data)

    return data


def nan_value(data, replacement):
    """
    Fills missing values with a specific value.
    """
    for idx, value in enumerate(data, 0):
        if np.isnan(value) == True:
            data[idx] = replacement

    return data


def nan_locf(data):
    """
    Fills missing values with the most recent non-missing value.
    """
    for idx, value in enumerate(data, 0):
        if np.isnan(value) == True:
            # data[idx] = data[:idx][0]
            data[idx] = data[:idx][-1]

    return data


def nan_nocb(data):
    """
    Fills missing values with the next non-missing value.
    """
    for idx, value in enumerate(data[::-1], 0):
        if np.isnan(value) == True:
            data[::-1][idx] = data[::-1][:idx][-1]

    return data


def nan_linear_interpolation(data):
    """
    Fills missing values via linear interpolation.
    """
    mask = np.logical_not(np.isnan(data))
    data = np.interp(np.arange(len(data)), np.arange(
        len(data))[mask], data[mask])

    return data


def mean_error(y_point, y_true):
    """
    Returns mean error of forecast.
    """
    error = y_point - y_true
    me = np.mean(error)

    return me


def root_mean_squared_error(y_point, y_true):
    """
    Returns root_mean_squared_error of forecast.
    """
    error = y_point - y_true
    rmse = np.sqrt(np.mean((error) ** 2))

    return rmse


def mean_absolute_error(y_point, y_true):
    """
    Returns mean absolute error of forecast.
    """
    error = y_point - y_true
    mae = np.mean(np.absolute(error))

    return mae


def mean_squared_error(y_point, y_true):
    """
    Returns mean squared error of forecast.
    """
    error = y_point - y_true
    mse = np.mean((error) ** 2)

    return mse


def mean_absolute_percentage_error(y_point, y_true):
    """
    Returns mean absolute percentage error of forecast.
    """
    error = y_point - y_true
    mape = np.mean(np.absolute(((error) / y_true) * 100))

    return mape


def symmetric_mean_absolute_percentage_error(y_point, y_true):
    """
    Returns symmetric mean absolute percentage error of forecast.
    """
    error = y_point - y_true
    smape = np.mean(np.absolute(error) / (y_point + y_true) * 200)

    return smape


def median_absolute_error(y_point, y_true):
    """
    Returns median absolute error of forecast.
    """
    error = y_point - y_true
    mdae = np.median(np.absolute(error))

    return mdae


def mad_mean_ratio(y_point, y_true):
    """
    Returns mad mean ratio of forecast.
    """
    error = y_point - y_true
    mmr = np.mean(np.absolute(error)) / np.mean(y_point)

    return mmr
