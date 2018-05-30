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


def acf(data, max_lags="default", ci=True, level=0.95):
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
        """
        Returns autocorrelation coefficient and its bounds.
        """
        acf_coeff = np.sum(((data[: n - h] - mean) * (data[h:] - mean))) / n / c0

        return acf_coeff

    acf_coeffs = np.empty([0, 3])
    for i in np.arange(max_lags):
        acf_coeff = corr(i)
        if ci is False:
            acf_coeffs_ph = np.hstack((np.nan, acf_coeff, np.nan))
            acf_coeffs = np.vstack((acf_coeffs, acf_coeffs_ph))
        else:
            t_crit = stats.t.ppf(q=level, df=(n - 3))
            acf_coeff_lb = np.negative(t_crit) / np.sqrt(n)
            acf_coeff_ub = t_crit / np.sqrt(n)
            acf_coeffs_ph = np.hstack((acf_coeff_lb, acf_coeff, acf_coeff_ub))
            acf_coeffs = np.vstack((acf_coeffs, acf_coeffs_ph))

    return acf_coeffs


def pacf(data, max_lags="default", ci=True, level=0.95):
    """
    Returns partial autocorrelation coefficients and their bounds
    of length max_lags.
    """
    n = len(data)
    x0 = data[:,]

    if max_lags is "default":
        max_lags = int(10 * np.log10(n))
    else:
        max_lags = int(max_lags)

    xlags = np.ones((n, max_lags))
    for i in range(1, max_lags):
        xlags[:, i] = np.roll(data, i)

    xlags[np.triu_indices(xlags.shape[1], 1)] = 0
    
    pacf_coeffs = np.empty([0, 3])
    for k in range(1, max_lags):
        pacf_coeff = np.linalg.lstsq(xlags[k:, :k+1], x0[k:])[0][-1]
        if ci is False:
            pacf_coeffs_ph = np.hstack((np.nan, pacf_coeff, np.nan))
            pacf_coeffs = np.vstack((pacf_coeffs, pacf_coeffs_ph))
            if k + 1 == max_lags:
                pacf_coeffs = np.vstack([[np.nan, 1., np.nan], pacf_coeffs])
            else:
                continue
        else:
            t_crit = stats.t.ppf(q=level, df=(n - 3))
            pacf_coeff_lb = np.negative(t_crit) / np.sqrt(n)
            pacf_coeff_ub = t_crit / np.sqrt(n)
            pacf_coeffs_ph = np.hstack((pacf_coeff_lb, pacf_coeff, pacf_coeff_ub))
            pacf_coeffs = np.vstack((pacf_coeffs, pacf_coeffs_ph))
            if k + 1 == max_lags:
                pacf_coeffs = np.vstack([[pacf_coeff_lb, 1., pacf_coeff_ub], pacf_coeffs])
            else:
                continue
    
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
                w2 = multiplier * np.sum(data[i + 1 : j + 1])
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

        trends[:pad,] = np.nan
        trends[-pad:,] = np.nan

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


from statsmodels.tsa.seasonal import seasonal_decompose

"""
data = np.array([1, 2, 4, 5, 2, 3, 5, -5, 4, 5, 6, -5])
order = 4
detrended_series = detrend(data, order, center=True, model="additive")
j = np.pad(
    detrended_series.astype(float),
    (0, order - detrended_series.size % order),
    mode="constant",
    constant_values=np.NaN,
).reshape(-1, order)
arr = np.nanmean(j, axis=1)

np.tile(np.nanmean(np.resize(jason, (2, 5)), axis=0), 2)
np.resize(np.nanmean(np.resize(jason, (2, 5)), axis=0), (12, 1))
"""
