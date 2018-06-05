import matplotlib.pyplot as plt
import numpy as np

from helpers._helpers import (
    acf_corr,
    pacf_ols,
    pacf_yule_walker,
    trend,
    detrend,
    seasonality,
    remainder,
)


def plot_series_original(
    self,
    title="Original Series",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    **kwargs
):
    """
    Plots original series.
    """
    plt.plot(self.date_original, self.y_original, **kwargs)

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()


def plot_series_transformed(
    self,
    title="Transformed Series",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    **kwargs
):
    """
    Plots transformed series.
    """
    plt.plot(self.date_original, self.y_transformed, **kwargs)

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
    res = acf_corr(data=self.y_transformed, max_lags=max_lags, ci=True, level=level)
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
    if method == "yw_unbiased":
        res = pacf_yule_walker(
            data=self.y_transformed,
            max_lags=max_lags,
            method="unbiased",
            ci=True,
            level=level,
        )
    elif method == "yw_mle":
        res = pacf_yule_walker(
            data=self.y_transformed,
            max_lags=max_lags,
            method="mle",
            ci=True,
            level=level,
        )
    else:
        res = pacf_ols(data=self.y_transformed, max_lags=max_lags, ci=True, level=0.95)

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
    order = self.season if order is "default" else order

    trends = trend(self.y_transformed, order, center)

    plt.plot(self.date_original, trends, **kwargs)
    if overlay is True:
        plt.plot(self.date_original, self.y_transformed, linestyle="dashed")
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
    order = self.season if order is "default" else order

    avg_seasonality = seasonality(self.y_transformed, order, center, model, median)

    plt.plot(self.date_original, avg_seasonality, **kwargs)

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
    order = self.season if order is "default" else order

    random = remainder(self.y_transformed, order, center, model, median)

    plt.plot(self.date_original, random, **kwargs)

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
    order = self.season if order is "default" else order

    trends = trend(self.y_transformed, order, center)
    avg_seasonality = seasonality(self.y_transformed, order, center, model, median)
    random = remainder(self.y_transformed, order, center, model, median)

    plots = {
        0: [self.y_transformed, y_lab[0]],
        1: [trends, y_lab[1]],
        2: [avg_seasonality, y_lab[2]],
        3: [random, y_lab[3]],
    }

    f, axarr = plt.subplots(len(plots), sharex=True)
    f.suptitle(title, y=1)
    f.subplots_adjust(hspace=0)
    f.text(0.5, -.05, x_lab, ha="center")
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()

    for k, v in plots.items():
        axarr[k].plot(self.date_original, v[0], **kwargs)
        axarr[k].set_ylabel(v[1])
        axarr[k].yaxis.set_ticks_position("right")
        axarr[k].yaxis.set_label_position("left")


def plot_nan_distribution(
    self,
    title="Distribution of nans",
    x_lab="Index",
    y_lab="Y",
    x_rotation=45,
    vfillcolor="#FADBD8",
    **kwargs
):
    """
    Plots distribution of missing values.
    """
    plt.plot(self.date_original, self.y_original, **kwargs)

    for x, y in list(zip(self.date_original, self.y_original)):
        if np.isnan(y):
            if np.isnan(y - 1):
                plt.axvspan(x - 1, x + 1, color=vfillcolor)
            else:
                continue

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()


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
