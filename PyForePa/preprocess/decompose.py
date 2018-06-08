import numpy as np

from PyForePa.helpers.helpers import trend, detrend, seasonality, remainder


def decompose_trend(self, order="default", center=True):
    """
    Returns array consisting of series trend.
    """
    data = self.values["X"]
    order = self.frequency if order is "default" else order

    trends = trend(data, order, center)

    return trends


def decompose_detrend(self, order="default", center=True, model="additive"):
    """
    Returns array of detrended series.
    """
    data = self.values["X"]
    order = self.frequency if order is "default" else order

    data_detrended = detrend(data, order, center, model)

    return data_detrended


def decompose_seasonality(
    self, order="default", center=True, model="additive", median=False
):
    """
    Returns array of series seasonality.
    """
    data = self.values["X"]
    order = self.frequency if order is "default" else order

    avg_seasonality = seasonality(data, order, center, model, median)

    return avg_seasonality


def decompose_remainder(
    self, order="default", center=True, model="additive", median=False
):
    """
    Returns array of left behind random noise.
    """
    data = self.values["X"]
    order = self.frequency if order is "default" else order

    random = remainder(data, order, center, model, median)

    return random


def decompose(self, order="default", center=True, model="additive", median=False):
    """
    Returns array of decomposition results in the order of:
    original series, trend, seasonality, random.
    """
    data = self.values["X"]
    order = self.frequency if order is "default" else order

    trends = trend(data, order, center)
    avg_seasonality = seasonality(data, order, center, model, median)
    random = remainder(data, order, center, model, median)

    decomp_series = np.column_stack((data, trends, avg_seasonality, random))

    return decomp_series
