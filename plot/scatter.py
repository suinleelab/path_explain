"""
Defines a function to plot individual feature-level importances
across a dataset.
"""
import pandas as pd
import numpy as np
import altair as alt

def scatter_plot(attributions,
                 feature_values,
                 feature_index,
                 interactions=None,
                 color_by=None,
                 feature_names=None):
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
        feature_index: The index of the feature to plot. If this is an integer,
                       indexes into attributions[:, feature_index]. If this is
                       a string, indexes into
                       attributions[:, where(features_names) == feature_index].
        interactions:  Either a matrix of the same shape as attributions representing
                       the interaction between feature_index and all other features,
                       or a matrix that can be indexed as
                       interactions[:, feature_index, color_by], which
                       provides the interaction between feature_index and color_by.
        color_by: An index of a feature to color the plot by. Follows the
                  same syntax as feature_index.
        feature_names: An optional list of length attributions.shape[1]. Each
                       entry should be a string representing the name of a feature.
    """
    if not isinstance(feature_index, int):
        if feature_names is None:
            raise ValueError('Provided argument feature_index {} was '.format(feature_index) + \
                             'not an integer but feature_names was not specified.')
        feature_index = feature_names.index(feature_index)
    if color_by and not isinstance(color_by, int):
        if feature_names is None:
            raise ValueError('Provided argument color_by {} was '.format(color_by) + \
                             'not an integer but feature_names was not specified.')
        color_by = feature_names.index(color_by)

    if feature_names is None:
        feature_names = ['Feature {}'.format(i) for i in range(attributions.shape[1])]

    x_name = 'Value of {}'.format(feature_names[feature_index])
    y_name = 'Attribution to {}'.format(feature_names[feature_index])
    data_df = pd.DataFrame({
        x_name: feature_values[:, feature_index],
        y_name: attributions[:, feature_index]
    })

    if color_by is not None:
        color_name = 'Value of {}'.format(feature_names[color_by])
        color_column = feature_values[:, color_by]
        vmin = np.nanpercentile(color_column, 5)
        vmax = np.nanpercentile(color_column, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(color_column, 1)
            vmax = np.nanpercentile(color_column, 99)
            if vmin == vmax:
                vmin = np.min(color_column)
                vmax = np.max(color_column)
        color_column = np.clip(color_column, vmin, vmax)

        data_df[color_name] = color_column

    chart = alt.Chart(data_df).mark_point(filled=True, size=20).encode(
        x=alt.X(x_name + ':Q'),
        y=alt.Y(y_name + ':Q')
    )

    if color_by is not None:
        chart = chart.encode(
            color=alt.Color(color_name + ':Q', scale=alt.Scale(scheme='goldgreen'))
        )

    if interactions is not None:
        if color_by is None:
            raise ValueError('Provided interactions but argument ' + \
                             'color_by was not specified')
        if interactions.shape == attributions.shape:
            interaction_column = 2.0 * interactions[:, color_by]
        else:
            interaction_column = interactions[:, feature_index, color_by] + interactions[:, color_by, feature_index]

        inter_name = 'Interaction between {} and {}'.format(feature_names[feature_index],
                                                            feature_names[color_by])
        main_name = 'Main effect of {} '.format(feature_names[feature_index])
        inter_df = pd.DataFrame({
            x_name: feature_values[:, feature_index],
            color_name: color_column,
            inter_name: interaction_column,
            main_name:  attributions[:, feature_index] - interaction_column
        })
        # TODO: WHY DO WE MULTIPLY BY TWO ABOVE? THINK ABOUT
        # WHY WE SHOULD SUBTRACT BY BOTH i,j entry and j,i
        # IN TERMS OF COMPLETENESS -

        inter_chart = alt.Chart(inter_df).mark_point(filled=True, size=20).encode(
            x=alt.X(x_name + ':Q'),
            y=alt.Y(inter_name + ':Q'),
            color=alt.Color(color_name + ':Q', scale=alt.Scale(scheme='goldgreen'))
        )

        main_chart = alt.Chart(inter_df).mark_point(filled=True, size=20).encode(
            x=alt.X(x_name + ':Q'),
            y=alt.Y(main_name + ':Q'),
            color=alt.Color(color_name + ':Q', scale=alt.Scale(scheme='goldgreen'))
        )

        chart = chart.properties(width=300, height=300) | \
                inter_chart.properties(width=300, height=300) | \
                main_chart.properties(width=300, height=300)
        chart = chart.configure_axis(labelFontSize=14,
                                     labelFontWeight=300,
                                     titleFontSize=16,
                                     titleFontWeight=400)

    return chart
