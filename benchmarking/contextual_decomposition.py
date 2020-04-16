"""
An implementation of the generalized form of contextual decomposition from
https://arxiv.org/pdf/1806.05337.pdf,
https://arxiv.org/pdf/1801.05453.pdf and
https://github.com/csinva/hierarchical-dnn-interpretations (implementation in Pytorch)

This implementation ignores the hierarchical aspect of the attributions
and only focuses on implementing explanations for feed-forward neural networks,
even though contextual decomposition can be applied to many types of neural networks.
We also only focus on linear layers and activations (ignoring layers like dropout, batch norm etc.)
for simplicity. This is because we are only interested in benchmarking performance and runtime.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

class ContextualDecompositionExplainerTF():
    def __init__(self, model):
        """
        Args:
            model: A tf.keras.Model instance.
        """
        self.model = model

    def _cd_scores(self,
                   batch_inputs,
                   batch_masks,
                   batch_output_indices=None):
        batch_beta  = batch_inputs * batch_masks
        batch_gamma = batch_inputs * (1 - batch_masks)

        # Bulk of the CD code is done here
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense) or \
               hasattr(layer, 'w'):

                if hasattr(layer, 'w'):
                    layer_weight = layer.w
                    layer_bias = tf.zeros(layer_weight.shape[1])
                else:
                    layer_weight = layer.weights[0]

                    if len(layer.weights) > 1:
                        layer_bias   = layer.weights[1]
                    else:
                        layer_bias = tf.zeros(layer_weight.shape[1])

                weight_beta  = tf.matmul(batch_beta, layer_weight)
                weight_gamma = tf.matmul(batch_gamma, layer_weight)

                # We add a small constant to the divisor to avoid zero-division
                # Note that arithmetic operations are performed on GPU
                bias_weight_sum   = tf.abs(weight_beta) + tf.abs(weight_gamma) + 1e-20
                bias_weight_beta  = tf.abs(weight_beta)  / bias_weight_sum
                bias_weight_gamma = tf.abs(weight_gamma) / bias_weight_sum

                bias_beta  = bias_weight_beta  * layer_bias
                bias_gamma = bias_weight_gamma * layer_bias

                batch_beta  = weight_beta  + bias_beta
                batch_gamma = weight_gamma + bias_gamma

                if hasattr(layer, 'activation') and layer.activation is not None:
                    activation_beta  = layer.activation (batch_beta)
                    activation_gamma = layer.activation (batch_beta + batch_gamma) - \
                                       layer.activation (batch_beta)

                    batch_beta  = activation_beta
                    batch_gamma = activation_gamma
            elif isinstance(layer, tf.keras.layers.Activation):
                # Here we handle propagation through (ReLU) activations.
                activation_beta  = layer(batch_beta)
                activation_gamma = layer(batch_beta + batch_gamma) - layer(batch_beta)

                batch_beta  = activation_beta
                batch_gamma = activation_gamma
            else:
                pass
#                 raise ValueError('Layer type {} '.format(type(layer)) + \
#                                  'not implemented.')

        if batch_output_indices is not None:
            indices = np.arange(batch_inputs.shape[0])
            indices = np.stack([indices, batch_output_indices], axis=1)
            batch_beta  = tf.gather_nd(batch_beta,
                                       indices=indices)
            batch_gamma = tf.gather_nd(batch_gamma,
                                       indices=indices)
        elif len(batch_beta.shape) == 2:
            batch_beta  = tf.reduce_sum(batch_beta, axis=-1)
            batch_gamma = tf.reduce_sum(batch_gamma, axis=-1)

        return batch_beta, batch_gamma

    def mask_scores(self,
                    inputs,
                    masks,
                    batch_size=50,
                    output_indices=None,
                    verbose=False):
        """
        Generates CD attributions for a particular input and mask. Currently the implementation
        only works for feed forward neural networks with a single path.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            baseline: A tensor of inputs to the model of shape
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            output_indices:  If this is None, then this function returns the
                             attributions by summing the output. This is rarely
                             what you want for classification tasks. Pass an
                             integer tensor of shape [batch_size] to
                             index the output output_indices[i] for
                             the input inputs[i].
        """
        masks  = tf.convert_to_tensor(masks)
        inputs = tf.convert_to_tensor(inputs)

        if len(masks.shape) == len(inputs[0].shape):
            masks = tf.expand_dims(masks, axis=0)
        if masks.dtype != tf.float32:
            masks = tf.cast(masks, tf.float32)
        if inputs.dtype != tf.float32:
            inputs = tf.cast(inputs, tf.float32)

        betas  = []
        gammas = []

        iterable = range(0, len(inputs), batch_size)
        if verbose:
            iterable = tqdm(iterable)

        for i in iterable:
            number_to_draw = min(batch_size, len(inputs) - i)
            batch_inputs = inputs[i:(i + number_to_draw)]

            if masks.shape[0] == 1:
                # Tile the masks so they match the shape of the input
                reps = [1] * len(masks.shape)
                reps[0] = number_to_draw
                batch_masks = tf.tile(masks, multiples=reps)
            else:
                batch_masks = masks[i:(i + number_to_draw)]

            if isinstance(output_indices, int):
                batch_output_indices = [output_indices] * number_to_draw
            elif output_indices is not None:
                batch_output_indices = output_indices[i:(i + number_to_draw)]
            else:
                batch_output_indices = None

            batch_beta, batch_gamma = self._cd_scores(batch_inputs,
                                                      batch_masks,
                                                      batch_output_indices)
            betas.append(batch_beta)
            gammas.append(batch_gamma)

        betas  = tf.concat(betas,  axis=0).numpy()
        gammas = tf.concat(gammas, axis=0).numpy()

        return betas, gammas

    def attributions(self,
                     inputs,
                     batch_size,
                     output_indices=None,
                     verbose=False):
        """
        This function is an attempt to mimic the path explainer
        API using contextual decomposition attributions. It assumes
        that you want all attributions to individual features.
        For CD, the baseline (masking value) seems to
        typically be the all-zeros vector, so we hard-code that
        into the function.

        This function assumes that your input has a single dimensionality (e.g.
        tabular data). If you want to use on inputs with more axes (e.g. images),
        you'll have to flatten them first.
        """
        num_features = inputs.shape[1]
        beta_matrix  = []
        gamma_matrix = []

        iterable = range(num_features)
        if verbose:
            iterable = tqdm(iterable)

        for feature in iterable:
            mask = np.zeros((num_features,), dtype=int)
            mask[feature] = 1
            mask = tf.convert_to_tensor(mask)

            betas, gammas = self.mask_scores(inputs=inputs,
                                             masks=mask,
                                             batch_size=batch_size,
                                             output_indices=output_indices,
                                             verbose=False)

            beta_matrix.append(betas)
            gamma_matrix.append(gammas)

        # These matrices will be the same size as the input,
        # and represent attributions to each feature.
        beta_matrix  = np.stack(beta_matrix,  axis=1)
        gamma_matrix = np.stack(gamma_matrix, axis=1)

        return beta_matrix, gamma_matrix

    def interactions(self,
                     inputs,
                     batch_size,
                     output_indices=None,
                     verbose=False,
                     interaction_index=None):
        """
        An attempt to mimic the path explainer API using interactions
        generated by contextual decomposition.

        The interaction_index argument can either be an integer (to get
        all interactions with that feature index), or a pair of integers
        (to get a specific interaction pair). If it is None,
        this function will return ALL pairwise interactions.

        This function returns `interactions` in the sense of the attributions
        to pairs. If you want `interactions` in the true sense of the word,
        you need to take the attribution to the pair and subtract the attribution
        to the individual features (we do this by computing attributions using the
        attribution function and then performing the subtraction).

        Like the attribution function, this function assumes your input
        has a single dimensionality.
        """
        num_features = inputs.shape[1]
        num_samples  = inputs.shape[0]

        if interaction_index is not None:
            if isinstance(interaction_index, int):
                beta_pairs  = np.zeros((num_samples, num_features))
                gamma_pairs = np.zeros((num_samples, num_features))
                iterable = [interaction_index]

            elif len(interaction_index) == 2:
                i = interaction_index[0]
                j = interaction_index[1]

                mask = np.zeros((num_features,), dtype=int)
                mask[i] = 1
                mask[j] = 1
                betas, gammas = self.mask_scores(inputs=inputs,
                                                 masks=mask,
                                                 batch_size=batch_size,
                                                 output_indices=output_indices,
                                                 verbose=False)
                return betas, gammas
        else:
            beta_pairs  = np.zeros((num_samples, num_features, num_features))
            gamma_pairs = np.zeros((num_samples, num_features, num_features))

            iterable = range(num_features)
            if verbose:
                iterable = tqdm(iterable)

        # The O(d^2) loop is necessary to compute all pairwise interactions, as
        # far as I can tell. Of course, computing d^2 values always takes O(d^2) time,
        # but the hessian parallelizes the loop. I suppose a better implementation
        # would parallelize the forward calls here as well. With that said,
        # the batching behavior is already somewhat parallel.
        for i in iterable:
            for j in range(i + 1, num_features):
                mask = np.zeros((num_features,), dtype=int)
                mask[i] = 1
                mask[j] = 1
                mask = tf.convert_to_tensor(mask)
                betas, gammas = self.mask_scores(inputs=inputs,
                                                 masks=mask,
                                                 batch_size=batch_size,
                                                 output_indices=output_indices,
                                                 verbose=False)
                if interaction_index is not None:
                    beta_pairs[:, j] = betas
                    gamma_pairs[:, j] = gammas
                else:
                    beta_pairs[:, i, j]  = betas
                    beta_pairs[:, j, i]  = betas

                    gamma_pairs[:, i, j] = gammas
                    gamma_pairs[:, j, i] = gammas

        return beta_pairs, gamma_pairs
