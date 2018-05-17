import numpy as np


class forecast:
    def __init__(
        self, model_info=None, y_point=None, y_lb=None, y_ub=None,
        residuals=None, y_original=None, date_original=None, season=None,
        y_transformed=None
    ):
        super(forecast, self).__init__()
        self.model_info = model_info
        self.y_point = y_point
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.residuals = residuals
        self.y_original = y_original
        self.date_original = date_original
        self.season = season
        self.y_transformed = y_transformed

    
    from ._accuracy import (
    	accuracy_me, accuracy_rmse, accuracy_mae, accuracy_mse, accuracy_mape,
    	accuracy_smape, accuracy_mdae, accuracy_mmr, accuracy
    )