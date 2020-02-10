"""
Defines a function to plot individual feature-level importances
in a summary plot.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .scatter import _get_bounds, _color_bar, _get_shared_limits, _set_axis_config
from . import colors

def _get_jitter_array(feature_values,
                      select_attributions):
    """
    Helper function to get jitter in a summary plot.
    Args:
        feature_values: see summary_plot
        select_attributions: see summary_plot
    """
    jitter_array = np.zeros(feature_values.shape)
    for i in range(select_attributions.shape[1]):
        feature_attr = select_attributions[:, i]
        num_samples = feature_attr.shape[0]
        nbins = 100
        quant = np.round(nbins * (feature_attr - np.min(feature_attr)) / \
                         (np.max(feature_attr) - \
                          np.min(feature_attr) + 1e-8))
        inds = np.argsort(quant + np.random.randn(num_samples) * 1e-6)
        layer = 0
        last_bin = -1
        jitter_values = np.zeros(num_samples)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            jitter_values[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        jitter_values *= 0.9 * (1.0 / np.max(jitter_values + 1))
        jitter_array[:, i] = jitter_values
    return jitter_array

def _get_jitter_df(interactions, feature_values,
                   select_attributions, attributions,
                   interaction_feature, feature_order):
    """
    Helper function to call the jitter matrix function.
    """
    if interactions is None:
        jitter_array = _get_jitter_array(feature_values, select_attributions)
        jitter_df = pd.DataFrame(jitter_array)
    else:
        if interactions.shape == attributions.shape:
            select_interactions = interactions[:, feature_order]
        else:
            if interaction_feature is None:
                raise ValueError('Argument interaction was specified ' + \
                                 'but argument interaction_feature was not.')
            select_interactions = interactions[:, feature_order, interaction_feature]
        jitter_df = pd.DataFrame(select_interactions)
    return jitter_df

def summary_plot(attributions,
                 feature_values,
                 interactions=None,
                 interaction_feature=None,
                 feature_names=None,
                 plot_top_k=None,
                 standardize_features=True,
                 scale_x_ind=False,
                 scale_y_ind=False,
                 figsize=(8, 4),
                 dpi=150,
                 **kwargs):
    """
    Function to draw an interactive scatter plot of
    attribution values. Since this is built on top
    of altair, this function works best when the
    number of points is small (< 5000).

    Args:
        attributions: A matrix of attributions.
                      Should be of shape [batch_size, feature_dims].
        feature_values: A matrix of feature values.
                        Should the same shape as the attributions.
        interactions:  Either a matrix of the same shape as attributions representing
                       the interaction between interaction_feature and all other features,
                       or a matrix that can be indexed as
                       interactions[:, :, interaction_feature].
        interaction_feature: A feature to use for interactions if interactions
                             are provided as all pairwise interactions.
        feature_names: An optional list of length attributions.shape[1]. Each
                       entry should be a string representing the name of a feature.
        plot_top_k: The number of features to plot. If none, will plot all features.
                    This might take a while, depending on how many features you have.
        scale_x_ind: Set to True to scale the x axes of each plot independently.
                     Defaults to False.
        scale_y_ind: Set to True to scale the y axes of each plot independently.
                     Defaults to False.
        figsize: Figure size in matplotlib units. Each figure will be square.
        dpi: Resolution of each plot.
        kwargs: Passed to plt.scatter
    """
    if plot_top_k is None:
        plot_top_k = attributions.shape[1]
    mean_abs_attr = np.mean(np.abs(attributions), axis=0)
    max_order = np.argsort(mean_abs_attr)
    feature_order = max_order[::-1][:plot_top_k]

    if feature_names is None:
        feature_names = ['Feature {}'.format(i) for i in range(feature_values.shape[1])]

    feature_values = feature_values[:, feature_order]
    select_attributions = attributions[:, feature_order]
    feature_names = [feature_names[i] for i in feature_order]

    if standardize_features:
        standardized_feature_values = (feature_values - np.mean(feature_values,
                                                                axis=0,
                                                                keepdims=True))
        standardized_feature_values = standardized_feature_values / \
                                      (np.std(standardized_feature_values,
                                              axis=0,
                                              keepdims=True) + 1e7)
    else:
        standardized_feature_values = feature_values

    vmin, vmax = _get_bounds(standardized_feature_values)
    standardized_feature_values = np.clip(standardized_feature_values, vmin, vmax)

    attribution_names = ['Attribution to {}'.format(feature_names[i]) for \
                             i in range(len(feature_names))]
    feature_df = pd.DataFrame(standardized_feature_values)
    attribution_df = pd.DataFrame(select_attributions)
    feature_df.columns = feature_names
    attribution_df.columns = attribution_names

    feature_df = pd.melt(feature_df, var_name='Feature', value_name='Normalized Feature Value')
    attribution_df = pd.melt(attribution_df, var_name='Attribution', value_name='Attribution Value')
    attribution_df = attribution_df.drop(columns=['Attribution'])

    jitter_df = _get_jitter_df(interactions, feature_values,
                               select_attributions, attributions,
                               interaction_feature, feature_order)
    jitter_df = pd.melt(jitter_df, var_name='Variable', value_name='Jitter')
    jitter_df = jitter_df.drop(columns=['Variable'])
    melted_df = pd.concat([feature_df, attribution_df, jitter_df], axis=1)

    if 's' not in kwargs:
        kwargs['s'] = 4
    if 'cmap' not in kwargs:
        kwargs['cmap'] = colors.green_gold()

    x_limits, y_limits = _get_shared_limits(melted_df['Attribution Value'],
                                            melted_df['Jitter'],
                                            scale_x_ind,
                                            scale_y_ind)

    fig, axs = plt.subplots(plot_top_k, 1, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.2, hspace=0)
    for i in range(plot_top_k - 1):
        axis = axs[i]
        _set_axis_config(axis,
                         clear_x_ticks=True,
                         clear_y_ticks=True)
        trans = mpl.transforms.blended_transform_factory(axis.transData, axis.transAxes)
        axis.plot([0.0, 1.0], [0.5, 0.5], transform=axis.transAxes,
                  linewidth=0.5, color='black', alpha=0.3, zorder=1)
        axis.plot([0.0, 0.0], [-1.0, 1.0], transform=trans, clip_on=False,
                  linewidth=0.5, color='black', alpha=0.3, zorder=1)

    axis = axs[-1]
    _set_axis_config(axis,
                     [0.0, 0.0, 0.0, 0.5],
                     clear_x_ticks=False,
                     clear_y_ticks=True)
    trans = mpl.transforms.blended_transform_factory(axis.transData, axis.transAxes)
    axis.plot([0.0, 1.0], [0.5, 0.5], transform=axis.transAxes,
              linewidth=0.5, color='black', alpha=0.3, zorder=1)
    axis.plot([0.0, 0.0], [0.0, 1.0], transform=trans,
              linewidth=0.5, color='black', alpha=0.3, zorder=1)
    axis.tick_params(length=4, labelsize=8)
    axis.set_xlabel('Attribution Value')

    for i in range(plot_top_k):
        axis = axs[i]
        selected_df = melted_df.loc[melted_df['Feature'] == feature_names[i]]
        trans = mpl.transforms.blended_transform_factory(axis.transAxes, axis.transAxes)
        axis.text(-0.02, 0.5, feature_names[i],
                  horizontalalignment='right',
                  verticalalignment='center',
                  fontsize=8,
                  transform=trans)
        axis.scatter(x=selected_df['Attribution Value'],
                     y=selected_df['Jitter'],
                     c=selected_df['Normalized Feature Value'],
                     zorder=2,
                     **kwargs)
        if x_limits is not None:
            axis.set_xlim(x_limits)
        if y_limits is not None:
            axis.set_ylim(y_limits)

    _color_bar(fig, vmin, vmax, 'Feature Value', ticks=False, label_size=8, **kwargs)
