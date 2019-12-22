"""
This module contains functions for plotting
attributions on text data.
"""
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import colors

def text_plot(word_array,
              attributions,
              include_legend=False,
              vmin=None,
              vmax=None,
              **kwargs):
    """
    A function to plot attributions on text data.
    Args:
        word_array: An array of strings.
        attributions: An array or iteratable of importance values.
                      Should be the same length as word_array.
        include_legend: If true, plots a color bar legend.
        **kwargs: Sent to matplotlib.pyplot.text
    """
    figsize = (0.1, 0.1)
    if include_legend:
        figsize=(10, 2)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    plt.axis('off')
    t = ax.transData

    space_text = plt.text(x=0.0, y=1.0, s='   ', transform=t, **kwargs)
    space_text.draw(fig.canvas.get_renderer())
    space_bounds = space_text.get_window_extent()

    if vmin is None and vmax is None:
        bounds = np.max(np.abs(attributions))
        vmin = -bounds
        vmax = bounds
    elif vmin is None:
        vmin = np.min(attributions)
    elif vmax is None:
        vmax = np.max(attributions)

    normalizer = mpl.colors.Normalize(vmin=vmin,
                                      vmax=vmax)

    color_mapper = mpl.cm.ScalarMappable(norm=normalizer,
                                         cmap=colors.green_white_gold())

    for word, importance in zip(word_array, attributions):
        color = color_mapper.to_rgba(importance)
        text = plt.text(x=0.0,
                        y=0.5,
                        s='{}'.format(word),
                        backgroundcolor=color,
                        transform=t,
                        **kwargs)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = mpl.transforms.offset_copy(text._transform, x=ex.width + space_bounds.width, units='dots')

    if include_legend:
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.1])
        color_bar = plt.colorbar(color_mapper, cax=cbar_ax, orientation='horizontal')
        color_bar.set_label('Attribution Value', fontsize=16)
        color_bar.outline.set_visible(False)