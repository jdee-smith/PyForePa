import numpy as np


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


def acf(data, max_lags='default'):
    """
    Returns array of autocorrelation coefficients of length max_lags.
    """    
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)
    
    if max_lags is 'default':
        max_lags = int(10 * np.log10(n))
    else:
        max_lags = max_lags

    def corr(h):
        """
        Returns autocorrelation coefficient.
        """
        acf_coeff = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / n / c0
        
        return acf_coeff
    
    acf_coeffs = np.empty([0, 1])
    for i in np.arange(max_lags):
        acf_coeff = corr(i)
        acf_coeffs = np.vstack((acf_coeffs, acf_coeff))
    
    return acf_coeffs