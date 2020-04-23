"""
A module to explain models using the exact or approximate
shapley interaction index and shapley value. Exact computation
is NP-Hard in the number of features - sampling is
recommended for instances with larger numbers of features.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from itertools import chain, combinations
from scipy.special import comb

class SamplingExplainerTF():
    def __init__(self, model):
        """
        Args:
            model: A tf.keras.Model instance.
        """
        self.model = model

    @staticmethod
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    @staticmethod
    def weight_coeff(S, N, interaction=False):
        if interaction:
            return 1.0 / ((N - 1) * comb(N - 2, S))
        else:
            return 1.0 / (N * comb(N - 1, S))

    def _call_model(self,
                    inputs,
                    output_index=None):
        outputs = self.model(inputs)

        if output_index is not None:
            outputs = outputs[:, output_index]
        elif len(outputs.shape) > 1:
             outputs = np.sum(outputs, axis=-1)
        return outputs

    def _sampling_estimate(self,
                           batch_inputs,
                           batch_baselines,
                           feature_index,
                           number_of_samples,
                           output_index=None,
                           interaction=False):
        accumulated_differences = []
        num_features = batch_inputs.shape[1]

        if interaction:
            i, j = feature_index
            feature_indices = np.arange(num_features)
            feature_indices = np.delete(feature_indices, j)

            for _ in range(number_of_samples):
                np.random.shuffle(feature_indices)
                pos = np.where(feature_indices == i)[0][0]

                batch_with_S = batch_baselines.copy()
                batch_with_S[:, feature_indices[:pos]] = batch_inputs[:, feature_indices[:pos]]
                v_S = self._call_model(batch_with_S, output_index)

                batch_with_S[:, i] = batch_inputs[:, i]
                v_S_i = self._call_model(batch_with_S, output_index)
                batch_with_S[:, i] = batch_baselines[:, i]

                batch_with_S[:, j] = batch_inputs[:, j]
                v_S_j = self._call_model(batch_with_S, output_index)

                batch_with_S[:, i] = batch_inputs[:, i]
                v_S_ij = self._call_model(batch_with_S, output_index)

                discrete_derivative = v_S_ij - v_S_i - v_S_j + v_S
                accumulated_differences.append(discrete_derivative)
        else:
            feature_indices = np.arange(num_features)
            for _ in range(number_of_samples):
                np.random.shuffle(feature_indices)
                pos = np.where(feature_indices == feature_index)[0][0]

                batch_with_S = batch_baselines.copy()
                batch_with_S[:, feature_indices[:pos]] = batch_inputs[:, feature_indices[:pos]]

                v_S = self._call_model(batch_with_S, output_index)

                batch_with_S[:, feature_index] = batch_inputs[:, feature_index]
                v_S_i = self._call_model(batch_with_S, output_index)
                difference = v_S_i - v_S
                accumulated_differences.append(difference)

        accumulated_differences = np.stack(accumulated_differences, axis=1)
        return np.mean(accumulated_differences, axis=1)

    def _batch_attributions(self,
                            batch_inputs,
                            batch_baselines,
                            number_of_samples=None,
                            output_index=None):
        num_features = batch_inputs.shape[1]
        batch_attributions = np.zeros(batch_inputs.shape)

        if number_of_samples is None:
            feature_powerset = SamplingExplainerTF.powerset(range(num_features))
            for S in feature_powerset:
                # First we create the subset with the features in S
                batch_with_S = batch_baselines.copy()
                batch_with_S[:, S] = batch_inputs[:, S]
                v_S = self._call_model(batch_with_S, output_index)

                for feature in filter(lambda x: x not in S,
                                      range(num_features)):
                    # Then we create the subset S u {i}
                    batch_with_S[:, feature] = batch_inputs[:, feature]
                    v_S_i = self._call_model(batch_with_S, output_index)

                    # We return to just the subset S
                    batch_with_S[:, feature] = batch_baselines[:, feature]

                    # Now we compute the contributions
                    difference = v_S_i - v_S

                    # And weight them appropriately
                    batch_attributions[:, feature] += difference * \
                        SamplingExplainerTF.weight_coeff(len(S), num_features)
        else:
            for feature_index in range(num_features):
                batch_importance = self._sampling_estimate(batch_inputs,
                                                           batch_baselines,
                                                           feature_index,
                                                           number_of_samples,
                                                           output_index=None)
                batch_attributions[:, feature_index] = batch_importance

        return batch_attributions

    def _get_discrete_derivative(self, batch_with_S, batch_baselines, batch_inputs, output_index, i, j):
        # First we create the subset with the features in S
        v_S = self._call_model(batch_with_S, output_index)

        # Then we create the subset S u {i}
        batch_with_S[:, i] = batch_inputs[:, i]
        v_S_i = self._call_model(batch_with_S, output_index)
        batch_with_S[:, i] = batch_baselines[:, i]

        # Then we create the subset S u {j}
        batch_with_S[:, j] = batch_inputs[:, j]
        v_S_j = self._call_model(batch_with_S, output_index)

        # Then we create the subset S u {i, j}
        batch_with_S[:, i] = batch_inputs[:, i]
        v_S_ij = self._call_model(batch_with_S, output_index)
        batch_with_S[:, i] = batch_baselines[:, i]
        batch_with_S[:, j] = batch_baselines[:, j]

        discrete_derivative = v_S_ij - v_S_i - v_S_j + v_S
        return discrete_derivative

    def _batch_interactions(self,
                            batch_inputs,
                            batch_baselines,
                            number_of_samples=None,
                            feature_index=None,
                            output_index=None):
        num_samples = batch_inputs.shape[0]
        num_features = batch_inputs.shape[1]

        if feature_index is not None:
            batch_interactions = np.zeros(num_samples)
        else:
            batch_interactions = np.zeros((num_samples, num_features, num_features))

        if number_of_samples is None:
            if feature_index is not None:
                indices = np.arange(num_features)
                i, j = feature_index
                np.delete(indices, feature_index)

                feature_powerset = SamplingExplainerTF.powerset(indices)
                for S in feature_powerset:
                    batch_with_S = batch_baselines.copy()
                    batch_with_S[:, S] = batch_inputs[:, S]
                    discrete_derivative = self._get_discrete_derivative(batch_with_S, batch_baselines, batch_inputs, output_index, i, j)

                    # And weight them appropriately
                    batch_interactions += discrete_derivative * \
                        SamplingExplainerTF.weight_coeff(len(S),
                                                         num_features,
                                                         interaction=True)
            else:
                feature_powerset = SamplingExplainerTF.powerset(range(num_features))
                for S in feature_powerset:
                    batch_with_S = batch_baselines.copy()
                    batch_with_S[:, S] = batch_inputs[:, S]

                    for i in filter(lambda x: x not in S,
                                    range(num_features)):
                        for j in filter(lambda x: x not in S,
                                        range(i + 1, num_features)):
                            discrete_derivative = self._get_discrete_derivative(batch_with_S, batch_baselines, batch_inputs, output_index, i, j)

                            # And weight them appropriately
                            batch_interactions[:, i, j] += discrete_derivative * \
                                SamplingExplainerTF.weight_coeff(len(S),
                                                                 num_features,
                                                                 interaction=True)
                            batch_interactions[:, j, i] = batch_interactions[:, i, j]
        else:
            if feature_index is not None:
                i, j = feature_index
                batch_importance = self._sampling_estimate(batch_inputs,
                                                               batch_baselines,
                                                               feature_index=(i, j),
                                                               number_of_samples=number_of_samples,
                                                               output_index=output_index,
                                                               interaction=True)
                return batch_importance
            else:
                for i in range(num_features):
                    for j in range(i + 1, num_features):
                        batch_importance = self._sampling_estimate(batch_inputs,
                                                                   batch_baselines,
                                                                   feature_index=(i, j),
                                                                   number_of_samples=number_of_samples,
                                                                   output_index=output_index,
                                                                   interaction=True)
                        batch_interactions[:, i, j] = batch_importance
                        batch_interactions[:, j, i] = batch_importance
        return batch_interactions

    def attributions(self,
                     inputs,
                     baselines,
                     batch_size=50,
                     number_of_samples=None,
                     output_index=None,
                     verbose=False):
        """
        A function to compute attributions directly using the shapley value.
        This function can either churn through an exponential number of computations,
        or sample uniformly from the subsets to get approximate attributions.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            baselines: A tensor of inputs to the model, either of shape (k, ...)
                       or of shape (...). In the latter case, uses the same baseline for all inputs.
                       In the former case, we assume k == batch_size and each baseline is fixed to the batch.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            number_of_samples: The number of samples to approximate the shapley value
                               with. If this parameter is None, it computes them exactly.
            output_index: Set to an integer to index into the output. Otherwise sums the outputs.
            verbose: Set to true to print a progress bar during computation.
        """
        num_inputs   = inputs.shape[0]
        num_features = inputs.shape[1]

        attributions = np.zeros((num_inputs, num_features))

        iterable = range(0, num_inputs, batch_size)
        if verbose:
            iterable = tqdm(iterable)

        for i in iterable:
            effective_batch_size = min(batch_size, num_inputs - i)
            batch_inputs = inputs[i:i + effective_batch_size]

            if len(baselines.shape) == len(inputs[0].shape):
                batch_baselines = baselines.copy()
                batch_baselines = batch_baselines[np.newaxis, :]
                batch_baselines = np.tile(batch_baselines, reps=(effective_batch_size, 1))
            else:
                batch_baselines = baselines[i:i + effective_batch_size]

            batch_attributions = self._batch_attributions(batch_inputs,
                                                          batch_baselines,
                                                          number_of_samples=number_of_samples,
                                                          output_index=output_index)
            attributions[i:i + effective_batch_size] = batch_attributions
        return attributions

    def interactions(self,
                     inputs,
                     baselines,
                     batch_size=50,
                     number_of_samples=None,
                     output_index=None,
                     feature_index=None,
                     verbose=False):
        """
        A function to compute interactions directly using the shapley interaction index.
        This function can either churn through an exponential number of computations,
        or sample uniformly from the subsets to get approximate attributions.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            baselines: A tensor of inputs to the model, either of shape (k, ...)
                       or of shape (...). In the latter case, uses the same baseline for all inputs.
                       In the former case, we assume k == batch_size and each baseline is fixed to the batch.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            number_of_samples: The number of samples to approximate the shapley value
                               with. If this parameter is None, it computes them exactly.
            output_index: Set to an integer to index into the output. Otherwise sums the outputs.
            verbose: Set to true to print a progress bar during computation.
        """
        num_inputs   = inputs.shape[0]
        num_features = inputs.shape[1]

        if feature_index is not None:
            interactions = np.zeros(num_inputs)
        else:
            interactions = np.zeros((num_inputs, num_features, num_features))

        iterable = range(0, num_inputs, batch_size)
        if verbose:
            iterable = tqdm(iterable)

        for i in iterable:
            effective_batch_size = min(batch_size, num_inputs - i)
            batch_inputs = inputs[i:i + effective_batch_size]

            if len(baselines.shape) == len(inputs[0].shape):
                batch_baselines = baselines.copy()
                batch_baselines = batch_baselines[np.newaxis, :]
                batch_baselines = np.tile(batch_baselines, reps=(effective_batch_size, 1))
            else:
                batch_baselines = baselines[i:i + effective_batch_size]

            batch_interactions = self._batch_interactions(batch_inputs,
                                                          batch_baselines,
                                                          number_of_samples=number_of_samples,
                                                          output_index=output_index,
                                                          feature_index=feature_index)
            interactions[i:i + effective_batch_size] = batch_interactions
        return interactions
