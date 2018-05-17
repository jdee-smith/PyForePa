import numpy as np

def transform_square_root(self):
    """
    Returns square root transformed y.
    """
    self.y_transformed = np.sqrt(np.float64(self.y_transformed))

    return self

def transform_natural_log(self):
    """
    Returns log transformed y.
    """
    self.y_transformed = np.log(np.float64(self.y_transformed))

    return self
