import matplotlib.pyplot as plt
import numpy as np

from datetime import timedelta

from PyForePa.helpers.helpers import (
    acf_corr,
    pacf_ols,
    pacf_yule_walker,
    trend,
    detrend,
    seasonality,
    remainder,
)


def plot_series(
    self,
    title="Series",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    **kwargs
):
    """
    Plots series.
    """
    x = self.values['index'].astype('O')
    y = self.values['X']

    plt.plot(x, y, **kwargs)

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()


def plot_acf(
    self,
    max_lags="default",
    level=0.95,
    title="ACF",
    x_lab="Lag",
    y_lab="ACF",
    **kwargs
):
    """
    Plots autocorrelation function of series up to max_lags.
    """
    data = self.values['X']
    res = acf_corr(data, max_lags, ci=True, level=level)
    coeffs = res[:, 1]
    coeffs_lb = res[0][0]
    coeffs_ub = res[0][2]
    x = np.arange(len(coeffs))

    plt.stem(x, coeffs, **kwargs)
    plt.hlines(
        (coeffs_lb, coeffs_ub),
        xmin=x[0],
        xmax=x[-1],
        linestyles=kwargs.pop("linestyles", "dotted"),
        **kwargs
    )

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.tight_layout()


def plot_pacf(
    self,
    method="yw_unbiased",
    max_lags="default",
    level=0.95,
    title="PACF",
    x_lab="Lag",
    y_lab="PACF",
    **kwargs
):
    """
    Plots autocorrelation function of series up to max_lags.
    """
    data = self.values['X']

    if method == "yw_unbiased":
        res = pacf_yule_walker(
            data, max_lags, "unbiased", ci=True, level=level)
    elif method == "yw_mle":
        res = pacf_yule_walker(data, max_lags, "mle", ci=True, level=level)
    else:
        res = pacf_ols(data, max_lags, ci=True, level=level)

    coeffs = res[:, 1]
    coeffs_lb = res[0][0]
    coeffs_ub = res[0][2]
    x = np.arange(1, len(coeffs) + 1, 1)

    plt.stem(x, coeffs, **kwargs)
    plt.hlines(
        (coeffs_lb, coeffs_ub),
        xmin=x[0],
        xmax=x[-1],
        linestyles=kwargs.pop("linestyles", "dotted"),
        **kwargs
    )

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.tight_layout()


def plot_trend(
    self,
    order="default",
    center=True,
    title="Trend",
    x_lab="Index",
    y_lab="Y",
    overlay=False,
    x_rotation=45,
    **kwargs
):
    """
    Plots series trend.
    """
    x = self.values['index'].astype('O')
    y = self.values['X']
    order = self.frequency if order is "default" else order

    trends = trend(y, order, center)

    plt.plot(x, trends, **kwargs)
    if overlay is True:
        plt.plot(x, y, linestyle="dashed")
        plt.legend(["Trend", "Series"], loc="upper left")
    else:
        pass

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()


def plot_seasonality(
    self,
    order="default",
    center=True,
    model="additive",
    median=False,
    title="Seasonality",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    **kwargs
):
    """
    Plots series seasonality.
    """
    x = self.values['index'].astype('O')
    y = self.values['X']
    order = self.frequency if order is "default" else order

    avg_seasonality = seasonality(y, order, center, model, median)

    plt.plot(x, avg_seasonality, **kwargs)

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()


def plot_random(
    self,
    order="default",
    center=True,
    model="additive",
    median=False,
    title="Random",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    **kwargs
):
    """
    Plots random component of series.
    """
    x = self.values['index'].astype('O')
    y = self.values['X']
    order = self.frequency if order is "default" else order

    random = remainder(y, order, center, model, median)

    plt.plot(x, random, **kwargs)

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()


def plot_series_decomposition(
    self,
    order="default",
    center=True,
    model="additive",
    median=False,
    title="Decomposed Series",
    x_lab="Index",
    y_lab=["Series", "Trend", "Seasonality", "Random"],
    x_rotation=45,
    **kwargs
):
    """
    Plots decomposition of series.
    """
    x = self.values['index'].astype('O')
    order = self.frequency if order is "default" else order

    decomposition = self.decompose()

    plots = {
        0: [decomposition[:, 0], y_lab[0]],
        1: [decomposition[:, 1], y_lab[1]],
        2: [decomposition[:, 2], y_lab[2]],
        3: [decomposition[:, 3], y_lab[3]],
    }

    f, axarr = plt.subplots(len(plots), sharex=True)
    f.suptitle(title, y=1)
    f.subplots_adjust(hspace=0)
    f.text(0.5, -.05, x_lab, ha="center")
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()

    for k, v in plots.items():
        axarr[k].plot(x, v[0], **kwargs)
        axarr[k].set_ylabel(v[1])
        axarr[k].yaxis.set_ticks_position("right")
        axarr[k].yaxis.set_label_position("left")


def plot_nan_distribution(
    self,
    title="Distribution of nans",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    vfillc="#FADBD8",
    **kwargs
):
    """
    Plots distribution of missing values.
    """
    a = self.values['index'].astype('O')
    b = self.values['X']

    plt.plot(a, b, **kwargs)

    for x, y in list(zip(a, b)):
        if np.isnan(y):
            if np.isnan(y - 1):
                plt.axvspan(x - timedelta(1), x + timedelta(1), color=vfillc)
            else:
                continue

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()


'''
def plot_imputed_values(
    self,
    title="Imputed Values",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    **kwargs
):
    """
    Plots imputed values.
    """
    markers = []
    for idx, (a, b) in enumerate(list(zip(self.y_original, self.y_transformed))):
        if np.isnan(a) and ~np.isnan(b):
            markers.append(idx)
        else:
            continue

    plt.plot(
        self.date_original,
        self.y_transformed,
        marker=kwargs.pop("marker", "."),
        markevery=markers,
        **kwargs
    )
'''
