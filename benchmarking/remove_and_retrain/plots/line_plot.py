
"""
A module for plotting line plots with bands as confidence intervals.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as st

def _plot_vals(data,
               x,
               y,
               sd=None,
               label=None,
               **kwargs):
    line_values = data[y]
    x_values = data[x]

    plt.plot(x_values,
             line_values,
             lw=1.5,
             label=label,
             **kwargs)

    if sd is not None:
        lower_conf = line_values - data[sd]
        upper_conf = line_values + data[sd]
        plt.fill_between(x_values,
                         lower_conf,
                         upper_conf,
                         alpha=0.25,
                         **kwargs)

def line_bar_plot(x,
                  y,
                  data,
                  color_by=None,
                  ax=None,
                  dpi=100,
                  xlabel=None,
                  ylabel=None,
                  title=None,
                  sd=None,
                  legend_title=None,
                  **kwargs):
    if 'loc' not in kwargs:
        loc = 'upper right'
    else:
        loc = kwargs.pop('loc')

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    if color_by is not None:
        colors = ['royalblue',
                  'firebrick',
                  'forestgreen',
                  'darkorange',
                  'black']
        if 'colors' in kwargs:
            colors = kwargs.pop('colors')

        for c, color_value in enumerate(data[color_by].unique()):
            kwargs['color'] = colors[c]
            _plot_vals(data[data[color_by] == color_value],
                       x,
                       y,
                       sd=sd,
                       label=color_value,
                       **kwargs)
        plt.legend(loc=loc, title=legend_title)
    else:
        if 'color' not in kwargs:
            kwargs['color'] = 'royalblue'
        _plot_vals(data,
                   x,
                   y,
                   sd=sd,
                   **kwargs)

    ax.spines["top"].set_alpha(0.1)
    ax.spines["bottom"].set_alpha(1)
    ax.spines["right"].set_alpha(0.1)
    ax.spines["left"].set_alpha(1)

    ax.grid(alpha=0.5, linestyle='--')

    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    return fig, ax