from helpers._helpers import acf_corr, pacf_ols, pacf_yule_walker


def describe_acf(self, max_lags="default", ci=True, level=0.95):
    """
    Returns autocorrelation coefficients and their bounds of length max_lags.
    """
    data = self.y_transformed

    acf = acf_corr(data, max_lags, ci, level)

    if ci is False:
        acf = acf[:, 1]

    return acf


def describe_pacf(self, method="yw_unbiased", max_lags="default", ci=True, level=0.95):
    """
    Returns partial autocorrelation coefficients and their bounds of length
    max_lags.
    """
    data = self.y_transformed

    if method == "yw_unbiased":
        pacf = pacf_yule_walker(data, max_lags, "unbiased", ci, level)
    elif method == "yw_mle":
        pacf = pacf_yule_walker(data, max_lags, "mle", ci, level)
    else:
        pacf = pacf_ols(data, max_lags, ci, level)

    if ci is False:
        pacf = pacf[:, 1]

    return pacf
