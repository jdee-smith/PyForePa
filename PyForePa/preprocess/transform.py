import numpy as np

from PyForePa.helpers.helpers import detrend


def transform_square_root(self):
    """
    Returns tseries object with y_original transformed via square root
    transformation.
    """
    data = self.values['X']
    self.values['X'] = np.sqrt(data)

    return self


def transform_natural_log(self):
    """
    Returns tseries object with y_original transformed via natural
    log transformation.
    """
    data = self.values['X']
    self.values['X'] = np.log(data)

    return self


def transform_detrend(self, order="default", center=True):
    """
    Returns tseries object with trend removed from y_original.
    """
    data = self.values['X']
    order = self.frequency if order is "default" else order

    self.values['X'] = detrend(data, order, center).reshape(len(data), )

    return self
