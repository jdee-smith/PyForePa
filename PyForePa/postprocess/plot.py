import matplotlib.pyplot as plt
import numpy as np


def plot_forecast(
    self,
    title="Forecast",
    x_lab="Index",
    y_lab="Y",
    add_series=True,
    vline=True,
    ci=True,
    time_index=True,
    x_rotation=45,
    **kwargs
):
    """
    Plots forecast. CODE IS GARBAGE!!! Needs complete re-do tbh.
    """
    series_len = len(self.series)
    h = len(self.forecasts)
    series_begin = self.series["index"][0]
    series_end = self.series["index"][-1]
    forecast_end = series_end + (h + 1)
    dtype = self.series["index"].dtype

    if add_series is True:
        if time_index is True:
            x = np.arange(series_begin, forecast_end, dtype=dtype).astype("O")
        else:
            x = np.arange(1, series_len + h + 1, 1)
        y = np.concatenate((self.series["X"], self.forecasts["point"]))

    else:
        if time_index is True:
            x = np.arange(series_end + 1, 6, dtype=dtype).astype("O")
        else:
            x = np.arange(1, h + 1, 1)
        y = self.forecasts["point"]

    plt.plot(x, y, **kwargs)

    if add_series is True and vline is True:
        if time_index is True:
            plt.axvline(x=series_end.astype("O"), linestyle="dotted")
        else:
            plt.axvline(x=series_len, linestyle="dotted")

    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()
