import numpy as np

from helpers._helpers import detrend


def transform_square_root(self):
    """
    Returns tseries object with y_original transformed via square root
    transformation.
    """
    self.y_transformed = np.sqrt(np.float64(self.y_transformed))

    return self


def transform_natural_log(self):
    """
    Returns tseries object with y_original transformed via natural
    log transformation.
    """
    self.y_transformed = np.log(np.float64(self.y_transformed))

    return self


def transform_detrend(self, order="default", center=True):
    """
    Returns tseries object with trend removed from y_original.
    """
    data = self.y_transformed
    order = self.season if order is "default" else order

    self.y_transformed = detrend(data, order, center)

    return self
