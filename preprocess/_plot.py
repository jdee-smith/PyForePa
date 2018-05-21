import matplotlib.pyplot as plt

def plot_series(self, **kwargs):
    """
    Plots original and transformed series.
    """
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False, sharex=True)

    plt.suptitle(kwargs.get('title_main', 'Original and Transformed Series'))
    plt.xlabel(kwargs.get('x_label', 'Index'))

    ax1.plot(
        self.date_original,
        self.y_original,
        color=kwargs.get('color_1', 'c'),
        linewidth=kwargs.get('linewidth_1', 1)
    )
    ax1.set_title(kwargs.get('title_1', ''))
    ax1.set_ylabel(kwargs.get('y_label_1', 'Y'))

    ax2.plot(
        self.date_original,
        self.y_transformed,
        color=kwargs.get('color_2', 'm'),
        linewidth=kwargs.get('linewidth_2', 1)
    )
    ax2.set_title(kwargs.get('title_2', ''))
    ax2.set_ylabel(kwargs.get('y_label_2', 'Y'))
    
    return f


def plot_series_original(self, **kwargs):
    """
    Plots original series.
    """
    plt.plot(
        self.date_original,
        self.y_original,
        color=kwargs.get('color', 'c'),
        linewidth=kwargs.get('linewidth', 1),
    )
    plt.title(kwargs.get('title', 'Original Series'))
    plt.ylabel(kwargs.get('y_label', 'Y'))
    plt.xlabel(kwargs.get('x_label', 'Index'))


def plot_series_transformed(self, **kwargs):
    """
    Plots transformed series.
    """
    plt.plot(
        self.date_original,
        self.y_transformed,
        color=kwargs.get('color', 'c'),
        linewidth=kwargs.get('linewidth', 1)
    )
    plt.title(kwargs.get('title', 'Transformed Series'))
    plt.ylabel(kwargs.get('y_label', 'Y'))
    plt.xlabel(kwargs.get('x_label', 'Index'))
