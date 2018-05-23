import matplotlib.pyplot as plt
import numpy as np

from helpers._helpers import (
    acf, decompose_trend, decompose_detrend,
    decompose_seasonality, decompose_remainder
)


def plot_series_original(
    self, title='Original Series', x_lab='Index', y_lab='Y', **kwargs
):
    """
    Plots original series.
    """
    plt.plot(self.date_original, self.y_original, **kwargs)
    
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)


def plot_series_transformed(
    self, title='Transformed Series', x_lab='Index', y_lab='Y', **kwargs
):
    """
    Plots transformed series.
    """
    plt.plot(self.date_original, self.y_transformed, **kwargs)
    
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    

def plot_acf(
    self, max_lags='default', level=0.95, title='ACF', x_lab='Lag',
    y_lab='ACF', **kwargs
):
    """
    Plots autocorrelation function of series up to max_lags.
    """
    res = acf(data=self.y_transformed, max_lags=max_lags, ci=True, level=level)
    coeffs = res[:,1]
    coeffs_lb = res[0][0]
    coeffs_ub = res[0][2]
    x = np.arange(len(coeffs))
    
    plt.stem(x, coeffs, **kwargs)
    plt.hlines(
        (coeffs_lb, coeffs_ub),
        xmin=x[0], 
        xmax=x[-1], 
        linestyles=kwargs.pop('linestyles', 'dotted'), 
        **kwargs
    )
    
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    
    
def plot_trend(
    self, title='Trend', x_lab='Index', y_lab='Y', overlay=False, **kwargs
):
    """
    Plots series trend.
    """
    trend = decompose_trend(self.y_transformed, self.season)
    
    plt.plot(self.date_original, trend, **kwargs)
    if overlay == True:
        plt.plot(self.date_original, self.y_transformed, linestyle='dashed')
        plt.legend(['Trend', 'Series'], loc='upper left')
    else:
        pass
    
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    
    
def plot_seasonality(
    self, title='Seasonality', x_lab='Index', y_lab='Y', model='additive',
    **kwargs
):
    """
    Plots series seasonality.
    """
    seasonality = decompose_seasonality(self.y_transformed, self.season, model)
    
    plt.plot(self.date_original, seasonality, **kwargs)
    
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    

def plot_random(
    self, title='Random', x_lab='Index', y_lab='Y', model='additive', **kwargs
):
    """
    Plots random component of series.
    """
    remainder = decompose_remainder(self.y_transformed, self.season, model)
    
    plt.plot(self.date_original, remainder, **kwargs)
    
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    