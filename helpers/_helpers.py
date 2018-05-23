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


def acf(data, max_lags='default', ci=True, level=0.95):
    """
    Returns autocorrelation coefficients and their bounds 
    of length max_lags.
    """
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    if max_lags is 'default':
        max_lags = int(10 * np.log10(n))
    else:
        max_lags = int(max_lags)

    def corr(h):
        """
        Returns autocorrelation coefficient and its bounds.
        """
        acf_coeff = np.sum(
            ((data[:n - h] - mean) * (data[h:] - mean))) / n / c0

        return acf_coeff

    acf_coeffs = np.empty([0, 3])
    for i in np.arange(max_lags):
        acf_coeff = corr(i)
        if ci is False:
            acf_coeffs_ph = np.hstack((np.nan, acf_coeff, np.nan))
            acf_coeffs = np.vstack((acf_coeffs, acf_coeffs_ph))
        else:
            t_crit = stats.t.ppf(q=level, df=(n-3))
            acf_coeff_lb = np.negative(t_crit) / np.sqrt(n)
            acf_coeff_ub = t_crit / np.sqrt(n)
            acf_coeffs_ph = np.hstack((acf_coeff_lb, acf_coeff, acf_coeff_ub))
            acf_coeffs = np.vstack((acf_coeffs, acf_coeffs_ph))

    return acf_coeffs


def decompose_trend(data, order):
    """
    Returns array consisting of series trends.
    """
    k = len(data)
    a = int(order / 2)
    
    if order % 2 == 0:
        b = int(a - 1)
    else:
        b = a
    
    trends = np.convolve(data.reshape((k, )), np.ones((order, )) / order, mode='valid')
    trends = np.pad(trends, (b, a), 'constant', constant_values=(np.nan, )).reshape(k, 1)
        
    return trends


def decompose_detrend(data, order, model='additive'):
    """
    Returns array of detrended series.
    """
    k = len(data)

    if model == 'additive':
        data_detrended = data.reshape(k, 1) - decompose_trend(data, order)
    elif model == 'multiplicative':
        data_detrended = data.reshape(k, 1) / decompose_trend(data, order)
    else:
        raise ValueError('Model must be additive or multiplicative.')
        
    return data_detrended


def decompose_seasonality(data, order, model='additive'):
    """
    Returns array of series seasonality.
    """
    n_rows = int(len(data) / order)
    n_cols = order
    
    detrended_series = decompose_detrend(data, order, model)
    
    mat = np.nanmean(np.matrix(detrended_series).reshape([n_rows, n_cols]), axis=0).T
    seasonality = np.array(np.tile(mat,(n_rows, 1)))
    
    return seasonality


def decompose_remainder(data, order, model='additive'):
    """
    Returns array of left behind random noise.
    """
    k = len(data)
    trend = decompose_trend(data, order)
    seasonality = decompose_seasonality(data, order, model)
    
    if model == 'additive':
        remainder = data.reshape(k, 1) - trend - seasonality
    elif model == 'multiplicative':
        remainder = data.reshape(k, 1) / (trend * seasonality)
    else:
        raise ValueError('Model must be additive or multiplicative.')
        
    return remainder
    
        