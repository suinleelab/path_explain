"""
Defines a function to plot individual feature-level importances
in a summary plot.
"""
import pandas as pd
import numpy as np
import altair as alt

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

def summary_plot(attributions,
                 feature_values,
                 interactions=None,
                 interaction_feature=None,
                 feature_names=None,
                 plot_top_k=None):
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
                    This might take a while!
    """
    if plot_top_k is None:
        plot_top_k = attributions.shape[1]
    mean_abs_attr = np.mean(np.abs(attributions), axis=0)
    max_order = np.argsort(mean_abs_attr)
    feature_order = max_order[:plot_top_k][::-1]

    if feature_names is None:
        feature_names = ['Feature {}'.format(i) for i in range(feature_values.shape[1])]

    feature_values = feature_values[:, feature_order]
    select_attributions = attributions[:, feature_order]
    feature_names = [feature_names[i] for i in feature_order]

    standardized_feature_values = (feature_values - np.mean(feature_values,
                                                            axis=0,
                                                            keepdims=True))
    standardized_feature_values = standardized_feature_values / \
                                  np.std(standardized_feature_values,
                                         axis=0,
                                         keepdims=True)
    vmin = np.nanpercentile(standardized_feature_values, 5)
    vmax = np.nanpercentile(standardized_feature_values, 95)
    if vmin == vmax:
        vmin = np.nanpercentile(standardized_feature_values, 1)
        vmax = np.nanpercentile(standardized_feature_values, 99)
        if vmin == vmax:
            vmin = np.min(standardized_feature_values)
            vmax = np.max(standardized_feature_values)
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

    jitter_df = pd.melt(jitter_df, var_name='Variable', value_name='Jitter')
    jitter_df = jitter_df.drop(columns=['Variable'])
    melted_df = pd.concat([feature_df, attribution_df, jitter_df], axis=1)

    chart = alt.Chart(melted_df, width=400, height=40)
    chart = chart.mark_point(filled=True, size=10)
    chart = chart.encode(x=alt.X('Attribution Value:Q',
                                 axis=alt.Axis(ticks=True,
                                               grid=False,
                                               labels=False)),
                         y=alt.Y('Jitter:Q',
                                 title=None,
                                 scale=alt.Scale(),
                                 axis=alt.Axis(values=[0],
                                               ticks=True,
                                               grid=True,
                                               labels=False)
                                ),
                         color=alt.Color('Normalized Feature Value:Q',
                                         legend=alt.Legend(direction='horizontal',
                                                           orient='bottom'),
                                         scale=alt.Scale(scheme='goldgreen')),
                         row=alt.Row('Feature:N',
                                     sort=feature_names,
                                     header=alt.Header(labelAngle=0,
                                                       labelAlign='right',
                                                       labelPadding=3)))

    chart = chart.configure_facet(spacing=0)
    chart = chart.configure_view(stroke=None)
    return chart
