import numpy as np

from itertools import zip_longest

from PyForePa.helpers.helpers import (
    nan_mean,
    nan_median,
    nan_random,
    nan_value,
    nan_locf,
    nan_nocb,
    nan_linear_interpolation,
    seasonality,
)


def impute_mean(self, trailing=True, mode=None, **kwargs):
    """
    Returns tseries object with missing values of the series filled
    with the mean up to that point.
    """
    data = self.values["X"]
    order = kwargs.pop("order", self.frequency)

    if mode is "decompose":
        seasonal = seasonality(data, order, **kwargs).reshape(len(data))
        de_seasonal = data - seasonal
        self.values["X"] = seasonal + nan_mean(de_seasonal, trailing)

    elif mode is "split":
        k = 0
        r = []
        while k < order:
            r.append(nan_mean(data[k::order], trailing))
            k += 1
        rz = np.array(list(zip_longest(*r, fillvalue=np.nan))).flatten()
        self.values["X"] = np.array([i for i in rz if not np.isnan(i).any()])

    else:
        self.values["X"] = nan_mean(data, trailing)

    return self


def impute_median(self, trailing=True, mode=None, **kwargs):
    """
    Returns tseries object with missing values of the series filled
    with the median up to that point.
    """
    data = self.values["X"]
    order = kwargs.pop("order", self.frequency)

    if mode is "decompose":
        seasonal = seasonality(data, order, **kwargs).reshape(len(data))
        de_seasonal = data - seasonal
        self.values["X"] = seasonal + nan_median(de_seasonal, trailing)

    elif mode is "split":
        k = 0
        r = []
        while k < order:
            r.append(nan_median(data[k::order], trailing))
            k += 1
        rz = np.array(list(zip_longest(*r, fillvalue=np.nan))).flatten()
        self.values["X"] = np.array([i for i in rz if not np.isnan(i).any()])

    else:
        self.values["X"] = nan_median(data, trailing)

    return self


def impute_random(self, trailing=True, mode=None, **kwargs):
    """
    Returns tseries object with missing values of the series filled
    with a random number from the series up to that point.
    """
    data = self.values["X"]
    order = kwargs.pop("order", self.frequency)

    if mode is "decompose":
        seasonal = seasonality(data, order, **kwargs).reshape(len(data))
        de_seasonal = data - seasonal
        self.values["X"] = seasonal + nan_random(de_seasonal, trailing)

    elif mode is "split":
        k = 0
        r = []
        while k < order:
            r.append(nan_random(data[k::order], trailing))
            k += 1
        rz = np.array(list(zip_longest(*r, fillvalue=np.nan))).flatten()
        self.values["X"] = np.array([i for i in rz if not np.isnan(i).any()])

    else:
        self.values["X"] = nan_random(data, trailing)

    return self


def impute_value(self, replacement, mode=None, **kwargs):
    """
    Returns tseries object with missing values of the series filled
    with a specific value.
    """
    data = self.values["X"]
    order = kwargs.pop("order", self.frequency)

    if mode is "decompose":
        seasonal = seasonality(data, order, **kwargs).reshape(len(data))
        de_seasonal = data - seasonal
        self.values["X"] = seasonal + nan_value(de_seasonal, replacement)

    elif mode is "split":
        k = 0
        r = []
        while k < order:
            r.append(nan_value(data[k::order], replacement))
            k += 1
        rz = np.array(list(zip_longest(*r, fillvalue=np.nan))).flatten()
        self.values["X"] = np.array([i for i in rz if not np.isnan(i).any()])

    else:
        self.values["X"] = nan_value(data, replacement)

    return self


def impute_locf(self, mode=None, **kwargs):
    """
    Returns tseries object with missing values of the series filled
    with the most recent non-missing value.
    """
    data = self.values["X"]
    order = kwargs.pop("order", self.frequency)

    if mode is "decompose":
        seasonal = seasonality(data, order, **kwargs).reshape(len(data))
        de_seasonal = data - seasonal
        self.values["X"] = seasonal + nan_locf(de_seasonal)

    elif mode is "split":
        k = 0
        r = []
        while k < order:
            r.append(nan_locf(data[k::order]))
            k += 1
        rz = np.array(list(zip_longest(*r, fillvalue=np.nan))).flatten()
        self.values["X"] = np.array([i for i in rz if not np.isnan(i).any()])

    else:
        self.values["X"] = nan_locf(data)

    return self


def impute_nocb(self, mode=None, **kwargs):
    """
    Returns tseries object with missing values of the series filled
    with the next non-missing value.
    """
    data = self.values["X"]
    order = kwargs.pop("order", self.frequency)

    if mode is "decompose":
        seasonal = seasonality(data, order, **kwargs).reshape(len(data))
        de_seasonal = data - seasonal
        self.values["X"] = seasonal + nan_nocb(de_seasonal)

    elif mode is "split":
        k = 0
        r = []
        while k < order:
            r.append(nan_nocb(data[k::order]))
            k += 1
        rz = np.array(list(zip_longest(*r, fillvalue=np.nan))).flatten()
        self.values["X"] = np.array([i for i in rz if not np.isnan(i).any()])

    else:
        self.values["X"] = nan_nocb(data)

    return self


def impute_interpolation(self, mode=None, **kwargs):
    """
    Returns tseries object with missing values of the series filled
    via linear interpolation.
    """
    data = self.values["X"]
    order = kwargs.pop("order", self.frequency)

    if mode is "decompose":
        seasonal = seasonality(data, order, **kwargs).reshape(len(data))
        de_seasonal = data - seasonal
        self.values["X"] = seasonal + nan_linear_interpolation(de_seasonal)

    elif mode is "split":
        k = 0
        r = []
        while k < order:
            r.append(nan_linear_interpolation(data[k::order]))
            k += 1
        rz = np.array(list(zip_longest(*r, fillvalue=np.nan))).flatten()
        self.values["X"] = np.array([i for i in rz if not np.isnan(i).any()])

    else:
        self.values["X"] = nan_linear_interpolation(data)

    return self
