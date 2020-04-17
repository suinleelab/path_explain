"""
An implementation of the neural interaction detection, from
https://arxiv.org/pdf/1705.04977.pdf and
https://github.com/jander081/neural-interaction-detection/blob/master/neural_interaction_detector_demo.ipynb

This code ignores the fact that the NID framework can detect higher order interactions
than just pairs, and that it has a mechanism to detect strong higher
order interactions quickly using a greedy algorithm. Here
we are only interested in benchmarking detecting pairwise interactions.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

class NeuralInteractionDetectionExplainerTF():
    def __init__(self, model, ignore_untrainable=True):
        """
        Args:
            model: A tf.keras.Model instance. Assumes that
                   model is feedforward with a dense layer as the
                   first layer.
        """
        self.model = model
        self.unit_influence, self.first_layer_weight = self._preprocess_weights(ignore_untrainable=ignore_untrainable)

    def _preprocess_weights(self, ignore_untrainable=True):
        """
        Pre-computes the `influence` vector z^(l)
        used to weight hidden unit importances. See the
        original paper for more details.
        """
        dense_layer_weights = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                if not ignore_untrainable or \
                    layer.trainable:
                    dense_layer_weights.append(layer.weights[0])

        first_layer_weight = np.abs(dense_layer_weights[0])

        aggregate_influence = tf.abs(dense_layer_weights[-1])
        dense_layer_weights = dense_layer_weights[1:-1]

        for layer_weight in dense_layer_weights[::-1]:
            aggregate_influence = tf.matmul(tf.abs(layer_weight), aggregate_influence)

        return aggregate_influence, first_layer_weight

    def _single_interaction(self,
                            output_index,
                            batch_inputs,
                            i, j,
                            unit_influence):
        weight_i = np.expand_dims(self.first_layer_weight[i], axis=0)
        weight_j = np.expand_dims(self.first_layer_weight[j], axis=0)

        weight_times_input_i = weight_i * np.abs(np.expand_dims(batch_inputs[:, i], axis=1))
        weight_times_input_j = weight_j * np.abs(np.expand_dims(batch_inputs[:, j], axis=1))

        interaction_strength = np.minimum(weight_times_input_i,
                                          weight_times_input_j) * unit_influence
        summed_interaction_strength = np.sum(interaction_strength, axis=1)
        return summed_interaction_strength

    def interactions(self,
                     output_index=None,
                     verbose=False,
                     inputs=None,
                     batch_size=None,
                     interaction_index=None):
        """
        Generates all pairwise interactions using the NID framework.

        Args:
            output_index: Used to index into the output.
            verbose: Set to true to enable tqdm looping.
            inputs: If this argument is not None, we will attempt
                    to generate local interactions using the NID framework
                    by using first layer weights * inputs instead of first layer
                    weights alone in order to generate interactions. Although
                    this wasn't discussed in the original paper, I believe
                    it's the only coherent way to make NID into a local
                    framework.
            batch_size: Only used if inputs is not None. The batch size to
                        feed into a matrix multiplication.
            interaction_index: Either an integer or a tuple of integers.
        """
        num_features = self.first_layer_weight.shape[0]

        if output_index is not None:
            unit_influence = self.unit_influence[:, output_index]
        else:
            unit_influence = tf.reduce_sum(self.unit_influence, axis=1)

        if inputs is not None:
            num_samples = inputs.shape[0]
            unit_influence = np.expand_dims(unit_influence, axis=0)

            if interaction_index is not None:
                if isinstance(interaction_index, int):
                    interaction_matrix = np.zeros((num_samples, num_features))
                elif len(interaction_index) == 2:
                    interaction_matrix = np.zeros((num_samples,))
            else:
                interaction_matrix = np.zeros((num_samples, num_features, num_features))

            iterable = range(0, len(inputs), batch_size)
            if verbose:
                iterable = tqdm(iterable)

            for k in iterable:
                number_to_draw = min(batch_size, len(inputs) - k)
                batch_inputs = inputs[k:(k + number_to_draw)]

                if interaction_index is not None:
                    if isinstance(interaction_index, int):
                        i = interaction_index
                        for j in range(num_features):
                            summed_interaction_strength = self._single_interaction(output_index, batch_inputs, i, j, unit_influence)
                            interaction_matrix[k:(k + number_to_draw), i] = summed_interaction_strength

                    elif len(interaction_index) == 2:
                        i = interaction_index[0]
                        j = interaction_index[1]
                        summed_interaction_strength = self._single_interaction(output_index, batch_inputs, i, j, unit_influence)

                        interaction_matrix[k:(k + number_to_draw)] = summed_interaction_strength
                else:
                    for i in range(num_features):
                        for j in range(i + 1, num_features):
                            summed_interaction_strength =  self._single_interaction(output_index,
                                                                                    batch_inputs,
                                                                                    i, j,
                                                                                    unit_influence)

                            interaction_matrix[k:(k + number_to_draw), i, j] = summed_interaction_strength
                            interaction_matrix[k:(k + number_to_draw), j, i] = summed_interaction_strength
        else:
            interaction_matrix = np.zeros((num_features, num_features))

            iterable = range(num_features)
            if verbose:
                iterable = tqdm(iterable)

            for i in iterable:
                for j in range(i + 1, num_features):
                    weight_i = self.first_layer_weight[i]
                    weight_j = self.first_layer_weight[j]

                    interaction_strength = np.minimum(weight_i, weight_j) * unit_influence
                    summed_interaction_strength = np.sum(interaction_strength)
                    interaction_matrix[i, j] = summed_interaction_strength
                    interaction_matrix[j, i] = summed_interaction_strength

        return interaction_matrix
