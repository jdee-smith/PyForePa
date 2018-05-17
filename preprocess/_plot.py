import matplotlib.pyplot as plt


def plot_series(self, **kwargs):
    """
    """
    plt.plot(
        self.date_original,
        self.y_transformed,
        linewidth=kwargs.get('linewidth', 3)
    )
