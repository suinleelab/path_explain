"""
This module contains functions for plotting
attributions on text data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import colors
from .scatter import _set_axis_config

def text_plot(word_array,
              attributions,
              include_legend=False,
              vmin=None,
              vmax=None,
              interaction_matrix=None,
              interaction_index=None,
              **kwargs):
    """
    A function to plot attributions on text data.
    Args:
        word_array: An array of strings.
        attributions: An array or iteratable of importance values.
                      Should be the same length as word_array.
        include_legend: If true, plots a color bar legend.
        interaction_matrix: Matrix of interactions, if you want additional explanations.
        interaction_index: Index to explain. Defaults to None.
        **kwargs: Sent to matplotlib.pyplot.text
    """
    figsize = (0.1, 0.1)
    if include_legend:
        figsize = (10, 2)

    fig = plt.figure(figsize=figsize)
    axis = fig.gca()
    plt.axis('off')
    axis_transform = axis.transData

    if interaction_index is not None:
        spacing = '     '
    else:
        spacing = '   '
    space_text = plt.text(x=0.0,
                          y=1.0,
                          s=spacing,
                          transform=axis_transform,
                          **kwargs)
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


    for i, (word, importance) in enumerate(zip(word_array, attributions)):
        fontweight = 500
        fontsize = 16
        y_pos = 0.5
        zorder=0
        if interaction_index is not None:
            y_pos = 0.7
            if i == interaction_index:
                fontweight = 700
                fontsize = 22
                zorder=10

        color = color_mapper.to_rgba(importance)
        text = plt.text(x=0.0,
                        y=y_pos,
                        s='{}'.format(word),
                        backgroundcolor=color,
                        fontsize=fontsize,
                        transform=axis_transform,
                        fontweight=fontweight,
                        zorder=zorder,
                        **kwargs)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        axis_transform = mpl.transforms.offset_copy(text._transform,
                                                    x=ex.width + space_bounds.width,
                                                    units='dots')

    if interaction_index is not None:
        axis_transform = axis.transData
        for word, importance in zip(word_array, interaction_matrix[interaction_index]):
            color = color_mapper.to_rgba(importance)

            text = plt.text(x=0.0,
                            y=0.2,
                            s='{}'.format(word),
                            backgroundcolor=color,
                            fontsize=16,
                            transform=axis_transform)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            axis_transform = mpl.transforms.offset_copy(text._transform,
                                                        x=ex.width + space_bounds.width,
                                                        units='dots')
    if include_legend:
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.1])
        color_bar = plt.colorbar(color_mapper, cax=cbar_ax, orientation='horizontal')
        color_bar.set_label('Attribution Value', fontsize=16)
        color_bar.outline.set_visible(False)

def matrix_interaction_plot(interaction_matrix,
                            tokens,
                            ax=None,
                            cbar_kw={},
                            cbarlabel="Interaction Value",
                            **kwargs):
    """
    A function to plot the text interaction matrix.

    Args:
        interaction_matrix: A len(tokens), len(tokens) sized matrix.
        tokens: A list of strings
        ax: An existing matplotlib axis object
        cbar_kw: Color bar kwargs
        cbarlabel: Label for the color bar
        kwargs: plt.imshow kwargs
    """
    if not ax:
        ax = plt.gca()

    bounds = np.max(np.abs(interaction_matrix))

    # Plot the heatmap
    if 'cmap' not in kwargs:
        kwargs['cmap'] = colors.green_white_gold()
    im = ax.imshow(interaction_matrix, vmin=-bounds, vmax=bounds, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=14)
    cbar.ax.tick_params(length=6, labelsize=12)
    cbar.outline.set_visible(False)

    # We want to show all ticks...
    ax.set_xticks(np.arange(interaction_matrix.shape[1]))
    ax.set_yticks(np.arange(interaction_matrix.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    ax.tick_params(length=6, labelsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(interaction_matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(interaction_matrix.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    color_threshold = np.quantile(interaction_matrix, 0.25)
    for i in range(interaction_matrix.shape[0]):
        for j in range(interaction_matrix.shape[1]):
            if i == j:
                continue
            if interaction_matrix[i, j] > color_threshold:
                color = 'black'
            else:
                color = 'white'

            if i > j:
                text = '{},\n{}'.format(tokens[j], tokens[i])
            else:
                text = '{},\n{}'.format(tokens[i], tokens[j])


            text = ax.text(j, i, text,
                           ha='center', va='center', color=color,
                           fontsize=12)

    return im, cbar

def bar_interaction_plot(interaction_matrix,
                         tokens,
                         top_k=5,
                         text_kwargs={},
                         pair_indices=None,
                         **kwargs):
    """
    A function to plot the word pairs with the largest
    absolute interaction values.

    Args:
        interaction_matrix: A len(tokens), len(tokens) sized matrix.
        tokens: A list of strings
        top_k: Number of top pairs to plot. Defaults to 5.
        text_kwargs: Passed to plt.text()
        pair_indices: Overrides top_k argument. A matrix of
                      size [top_k, 2] that lists the
                      indices you want to plot pairs of.
                      That is, pair_indices[i] is
                      the ith index into the matrix interaction_matrix
                      that you want plotted.
        **kwargs: Passed to plt.barh()
    """
    if pair_indices is None:
        pair_indices = np.argsort(np.triu(np.abs(interaction_matrix)).flatten())[::-1][:top_k]
        pair_indices = np.vstack(np.unravel_index(pair_indices, interaction_matrix.shape)).T
    else:
        top_k = len(pair_indices)

    token_labels = []
    interaction_values = []
    for index in pair_indices[::-1]:
        token_labels.append('{}, {} ({}, {})'.format(tokens[index[0]], tokens[index[1]], index[0], index[1]))
        interaction_values.append(interaction_matrix[index[0], index[1]])

    fig, ax = plt.subplots()

    bounds = np.max(np.abs(interaction_matrix))
    normalizer = mpl.colors.Normalize(vmin=-bounds, vmax=bounds)

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = colors.green_white_gold()

    ax.barh(np.arange(top_k),
            interaction_values,
            color=[cmap(normalizer(c)) for c in interaction_values],
            align='center',
            zorder=10,
            **kwargs)
    ax.set_xlabel('Interaction Value', fontsize=14)
    ax.set_ylabel('Strongest Interacting Pairs', fontsize=14)
    ax.set_yticks(np.arange(top_k))
    ax.tick_params(axis='y', which='both', left=False, labelsize=12)
    ax.set_yticklabels(token_labels)

    ax.grid(axis='x', zorder=0, linewidth=0.2)
    ax.grid(axis='y', zorder=0, linestyle='--', linewidth=1.0)
    _set_axis_config(ax,
                     linewidths=(0.0, 0.0, 0.0, 1.0))

    text_ax = fig.add_axes([0.1, 0.9, 0.8, 0.1])
    axis_transform = text_ax.transData
    _set_axis_config(text_ax, clear_y_ticks=True, clear_x_ticks=True)
    space_text = text_ax.text(x=0.0,
                              y=1.0,
                              s=' ',
                              transform=axis_transform)
    space_text.draw(fig.canvas.get_renderer())
    space_bounds = space_text.get_window_extent()

    for i, token in enumerate(tokens):
        text = text_ax.text(x=0.0,
                       y=0.6,
                       s=token,
                       transform=axis_transform,
                       fontsize=16,
                       **text_kwargs)
        text_ax.text(x=0.0,
                y=0.0,
                s=str(i),
                transform=axis_transform,
                fontsize=16,
                **text_kwargs)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        axis_transform = mpl.transforms.offset_copy(text._transform,
                                                    x=ex.width + space_bounds.width,
                                                    units='dots')
    return ax, text_ax