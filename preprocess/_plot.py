import matplotlib.pyplot as plt


def plot_series(self, **kwargs):
    """
    """
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    ax1.plot(
        self.date_original,
        self.y_original,
        color=kwargs.get('color_1', 'c'),
        linewidth=kwargs.get('linewidth_1', 1)
    )
    ax1.set_title('Original Series')

    ax2.plot(
        self.date_original,
        self.y_transformed,
        color=kwargs.get('color_2', 'm'),
        linewidth=kwargs.get('linewidth_2', 1)
    )
    ax2.set_title('Transformed Series')


def plot_original_series(self, **kwargs):
    """
    """
    plt.plot(
        self.date_original,
        self.y_original,
        color=kwargs.get('color', 'c'),
        linewidth=kwargs.get('linewidth', 1)
    )
    plt.title(kwargs.get('title', 'Original Series'))


def plot_transformed_series(self, **kwargs):
    """
    """
    plt.plot(
        self.date_original,
        self.y_transformed,
        color=kwargs.get('color', 'c'),
        linewidth=kwargs.get('linewidth', 1)
    )
    plt.title(kwargs.get('title', 'Transformed Series'))


def plot_acf(self, max_lags, **kwargs):
    """
    """
    plt.acorr(
        self.y_transformed,
        maxlags=max_lags,
        color=kwargs.get('color', 'c'),
        linewidth=kwargs.get('linewidth', 1)
    )
    plt.title(kwargs.get('title', 'ACF'))
