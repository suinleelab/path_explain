"""
Contains a single import: the PathExplainerTF object, which is
used to get feature-level importances of TensorFlow
gradient-based models.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .explainer import Explainer

class PathExplainerTF(Explainer):
    """
    Explains a model using path attributions from the given baseline.
    """

    def __init__(self, model):
        """
        Initialize the TF explainer class. This class
        will handle both the eager and the
        old-school graph-based tensorflow models.

        Args:
            model: A tf.keras.Model instance if executing eagerly.
                   A tuple (input_tensor, output_tensor) otherwise.
        """
        self.model = model
        self.eager_mode = False
        try:
            self.eager_mode = tf.executing_eagerly()
        except AttributeError:
            pass

    def _sample_baseline(self, baseline, number_to_draw, use_expectation):
        """
        An internal function to sample the baseline.

        Args:
            baseline: The baseline tensor
            number_to_draw: The number of samples to draw
            use_expectation: Whether or not to sample baselines
                             or use a single baseline

        Returns:
            A tensor of shape (number_to_draw, ...) representing
            the baseline and a tensor of shape (number_to_draw) representing
            the sampled interpolation constants
        """
        if use_expectation:
            replace = baseline.shape[0] < number_to_draw
            sample_indices = np.random.choice(baseline.shape[0],
                                              size=number_to_draw,
                                              replace=replace)
            sampled_baseline = baseline[sample_indices]
        else:
            reps = np.ones(len(baseline.shape)).astype(int)
            reps[0] = number_to_draw
            sampled_baseline = np.tile(baseline, reps)
        return sampled_baseline

    def _sample_alphas(self, num_samples, use_expectation, use_product=False):
        """
        An internal function to sample the interpolation constant.

        Args:
            num_samples: Number of alphas to draw
            use_expectation: Whether or not to use
                             expected gradients-style sampling
            use_product: Set to true to sample from
                         the product distribution
        """
        if use_expectation:
            if use_product:
                alpha = np.random.uniform(low=0.0, high=1.0, size=num_samples)
                beta = np.random.uniform(low=0.0, high=1.0, size=num_samples)
                return alpha, beta
            else:
                return np.random.uniform(low=0.0, high=1.0, size=num_samples)
        else:
            if use_product:
                sqrt_samples = np.ceil(np.sqrt(num_samples)).astype(int)
                spaced_points = np.linspace(start=0.0, stop=1.0, num=sqrt_samples, endpoint=True)

                num_drawn = sqrt_samples * sqrt_samples
                slice_indices = np.round(np.linspace(start=0.0,
                                                     stop=num_drawn-1,
                                                     num=num_samples,
                                                     endpoint=True)).astype(int)

                ones_map = np.ones(sqrt_samples)
                beta = np.outer(spaced_points, ones_map).flatten()
                beta = beta[slice_indices]

                alpha = np.outer(ones_map, spaced_points).flatten()
                alpha = alpha[slice_indices]

                return alpha, beta
            else:
                return np.linspace(start=0.0, stop=1.0, num=num_samples, endpoint=True)

    def _single_attribution(self, current_input, current_baseline,
                            current_alphas, num_samples, batch_size,
                            use_expectation, output_index):
        """
        A helper function to compute path
        attributions for a single sample.

        Args:
            current_input: A single sample. Assumes that
                           it is of shape (...) where ...
                           represents the input dimensionality
            baseline: A tensor representing the baseline input.
            current_alphas: Which alphas to use when interpolating
            num_samples: The number of samples to draw
            batch_size: Batch size to input to the model
            use_expectation: Whether or not to sample the baseline
            output_index: Whether or not to index into a given class
        """
        current_input = np.expand_dims(current_input, axis=0)
        current_alphas = tf.reshape(current_alphas, (num_samples,) + \
                                    (1,) * (len(current_input.shape) - 1))

        attribution_array = []
        for j in range(0, num_samples, batch_size):
            number_to_draw = min(batch_size, num_samples - j)

            batch_baseline = self._sample_baseline(current_baseline,
                                                   number_to_draw,
                                                   use_expectation)
            batch_alphas = current_alphas[j:min(j + batch_size, num_samples)]

            reps = np.ones(len(current_input.shape)).astype(int)
            reps[0] = number_to_draw
            batch_input = tf.convert_to_tensor(np.tile(current_input, reps))

            batch_difference = batch_input - batch_baseline
            batch_interpolated = batch_alphas * batch_input + \
                                 (1.0 - batch_alphas) * batch_baseline

            with tf.GradientTape() as tape:
                tape.watch(batch_interpolated)

                batch_predictions = self.model(batch_interpolated)
                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]

            batch_gradients = tape.gradient(batch_predictions, batch_interpolated)

            batch_attributions = batch_gradients * batch_difference
            attribution_array.append(batch_attributions)
        attribution_array = np.concatenate(attribution_array, axis=0)
        attributions = np.mean(attribution_array, axis=0)
        return attributions

    def _init_array(self,
                    inputs,
                    output_indices,
                    interaction_index=None,
                    as_interactions=False):
        """
        Internal helper function to get an
        array of the proper shape.

        Args:
            See self.attributions for definitions.
        """
        test_output = self.model(inputs[0:1])
        is_multi_output = len(test_output.shape) > 1
        shape_tuple = inputs.shape
        num_classes = test_output.shape[-1]

        if as_interactions and interaction_index is None:
            shape_tuple = (inputs.shape[0], ) + \
                          2 * (inputs.shape[1:])

        if is_multi_output and output_indices is None:
            num_classes = test_output.shape[-1]
            attributions = np.zeros((num_classes,) + shape_tuple)
        elif not is_multi_output and output_indices is not None:
            raise ValueError('Provided output_indices but ' + \
                             'model is not multi output!')
        else:
            attributions = np.zeros(shape_tuple)

        return attributions, is_multi_output, num_classes

    def attributions(self, inputs, baseline,
                     batch_size=50, num_samples=100,
                     use_expectation=True, output_indices=None,
                     verbose=False):
        """
        A function to compute path attributions on the given
        inputs.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            baseline: A tensor of inputs to the model of shape
                      (num_refs, ...) where ... indicates the dimensionality
                      of the input.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            num_samples: The number of samples to use when computing the
                         expectation or integral.
            use_expectation: If True, this samples baselines and interpolation
                             constants uniformly at random (expected gradients).
                             If False, then this assumes num_refs=1 in which
                             case it uses the same baseline for all inputs,
                             or num_refs=batch_size, in which case it uses
                             baseline[i] for inputs[i] and takes 100 linearly spaced
                             points between baseline and input (integrated gradients).
            output_indices:  If this is None, then this function returns the
                             attributions for each output class. This is rarely
                             what you want for classification tasks. Pass an
                             integer tensor of shape [batch_size] to
                             index the output output_indices[i] for
                             the input inputs[i].
        """
        attributions, is_multi_output, num_classes = self._init_array(inputs,
                                                                      output_indices)

        input_iterable = enumerate(inputs)
        if verbose:
            input_iterable = enumerate(tqdm(inputs))

        for i, current_input in input_iterable:
            current_alphas = self._sample_alphas(num_samples, use_expectation)

            if not use_expectation and baseline.shape[0] > 1:
                current_baseline = np.expand_dims(baseline[i], axis=0)
            else:
                current_baseline = baseline

            if is_multi_output:
                if output_indices is not None:
                    output_index = output_indices[i]
                    current_attributions = self._single_attribution(current_input,
                                                                    current_baseline,
                                                                    current_alphas,
                                                                    num_samples,
                                                                    batch_size,
                                                                    use_expectation,
                                                                    output_index)
                    attributions[i] = current_attributions
                else:
                    for output_index in range(num_classes):
                        current_attributions = self._single_attribution(current_input,
                                                                        current_baseline,
                                                                        current_alphas,
                                                                        num_samples,
                                                                        batch_size,
                                                                        use_expectation,
                                                                        output_index)
                        attributions[output_index, i] = current_attributions
            else:
                current_attributions = self._single_attribution(current_input,
                                                                current_baseline,
                                                                current_alphas,
                                                                num_samples,
                                                                batch_size,
                                                                use_expectation,
                                                                None)
                attributions[i] = current_attributions
        return attributions

    def _get_diagonal_derivatives(self, batch_hessian, batch_gradients,
                                  interaction_index):
        """
        A helper function to get the diagonal derivatives
        for the interaction values.

        Args:
            batch_hessian: from _single_interaction
            batch_gradient: from _single_interaction
            interaction_index: from _single_interaction
        """
        if interaction_index is not None:
            diagonal_derivative = np.zeros(shape=batch_hessian.shape.as_list())
            diagonal_derivative[tuple([slice(None)] + interaction_index)] = \
                    batch_gradients
        else:
            gathered_indices = np.array(list(np.ndindex(*batch_gradients.shape.as_list())))
            gathered_indices = np.concatenate([gathered_indices, gathered_indices[:, 1:]],
                                              axis=1)
            diagonal_derivative = tf.scatter_nd(gathered_indices,
                                                tf.reshape(batch_gradients,
                                                           (-1,)),
                                                batch_hessian.shape)
            diagonal_derivative = diagonal_derivative.numpy()
        return diagonal_derivative


    def _single_interaction(self, current_input, current_baseline,
                            current_alphas, num_samples, batch_size,
                            use_expectation, output_index,
                            interaction_index):
        """
        A helper function to compute path
        interactions for a single sample.

        Args:
            current_input: A single sample. Assumes that
                           it is of shape (...) where ...
                           represents the input dimensionality
            baseline: A tensor representing the baseline input.
            current_alphas: Which alphas to use when interpolating
            num_samples: The number of samples to draw
            batch_size: Batch size to input to the model
            use_expectation: Whether or not to sample the baseline
            output_index: Whether or not to index into a given class
            interaction_index: The index to take the interactions with respect to.
        """
        current_input = np.expand_dims(current_input, axis=0)
        current_alpha, current_beta = current_alphas
        current_alpha = tf.reshape(current_alpha, (num_samples,) + \
                                    (1,) * (len(current_input.shape) - 1))
        current_beta = tf.reshape(current_beta, (num_samples,) + \
                                 (1,) * (len(current_input.shape) - 1))
        attribution_array = []
        for j in range(0, num_samples, batch_size):
            number_to_draw = min(batch_size, num_samples - j)

            batch_baseline = self._sample_baseline(current_baseline,
                                                   number_to_draw,
                                                   use_expectation)
            batch_alpha = current_alpha[j:min(j + batch_size, num_samples)]
            batch_beta = current_beta[j:min(j + batch_size, num_samples)]
            batch_product = batch_alpha * batch_beta

            reps = np.ones(len(current_input.shape)).astype(int)
            reps[0] = number_to_draw
            batch_input = tf.convert_to_tensor(np.tile(current_input, reps))

            batch_difference = batch_input - batch_baseline
            batch_interpolated = batch_product * batch_input + \
                                 (1.0 - batch_product) * batch_baseline
            with tf.GradientTape() as second_order_tape:
                second_order_tape.watch(batch_interpolated)
                with tf.GradientTape() as first_order_tape:
                    first_order_tape.watch(batch_interpolated)

                    batch_predictions = self.model(batch_interpolated)
                    if output_index is not None:
                        batch_predictions = batch_predictions[:, output_index]

                #Same shape as the input, e.g. [batch_size, ...]
                batch_gradients = first_order_tape.gradient(batch_predictions, batch_interpolated)
                if interaction_index is not None:
                    if isinstance(interaction_index, int):
                        interaction_index = [interaction_index]
                    else:
                        interaction_index = list(interaction_index)
                    batch_gradients = batch_gradients[tuple([slice(None)] + interaction_index)]

            if interaction_index is not None:
                #In this case, the hessian is the same size as the input because we
                #indexed into a particular gradient
                batch_differences_secondary = batch_difference[tuple([slice(None)] + \
                                                                     interaction_index)]

                for _ in range(len(current_input.shape) - 1):
                    batch_differences_secondary = tf.expand_dims(batch_differences_secondary,
                                                                 axis=-1)
                batch_hessian = second_order_tape.gradient(batch_gradients, batch_interpolated)
            else:
                #In this case, the hessian matrix duplicates the dimensionality of the
                #input. That is, if we have 5 CIFAR10 images for example, the
                #input is of shape [5, 32, 32, 3]. The hessian will be of
                #shape [5, 32, 32, 3, 5, 32, 32, 3]. This can
                #be very memory intesive - there are other strategies,
                #but this one sacrifices memory to be fast
                batch_hessian = second_order_tape.jacobian(batch_gradients, batch_interpolated)
                batch_hessian = batch_hessian.numpy()

                #We index into the non-zero elements to get
                #the shape [5, 32, 32, 3, 32, 32, 3]
                hessian_index_array = [np.arange(number_to_draw)] + \
                                      [slice(None)] * (len(current_input.shape) - 1)
                hessian_index_array = 2 * hessian_index_array
                batch_hessian = batch_hessian[tuple(hessian_index_array)]

                #Now we adjust the batch_difference tensor to
                #be ready to broadcast. We use a broadcasting trick
                #to get the pairwise-differences

                #Note: in eager mode, this copies the value,
                #not the reference
                batch_differences_secondary = batch_difference

                for _ in range(len(current_input.shape) - 1):
                    batch_product = tf.expand_dims(batch_product, axis=-1)
                    batch_difference = tf.expand_dims(batch_difference, axis=-1)
                    batch_differences_secondary = tf.expand_dims(batch_differences_secondary,
                                                                 axis=1)

            off_diagonal_attributions = batch_hessian * batch_product * \
                                        batch_difference * batch_differences_secondary
            diagonal_derivative = self._get_diagonal_derivatives(batch_hessian,
                                                                 batch_gradients,
                                                                 interaction_index)
            batch_attributions = off_diagonal_attributions + \
                                 batch_difference * diagonal_derivative
            attribution_array.append(batch_attributions)
        attribution_array = np.concatenate(attribution_array, axis=0)
        attributions = np.mean(attribution_array, axis=0)
        return attributions

    def interactions(self, inputs, baseline,
                     batch_size=50, num_samples=100,
                     use_expectation=True, output_indices=None,
                     verbose=False, explain_index=None,
                     interaction_index=None):
        """
        A function to compute path interactions (attributions of
        attributions) on the given inputs.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            baseline: A tensor of inputs to the model of shape
                      (num_refs, ...) where ... indicates the dimensionality
                      of the input.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            num_samples: The number of samples to use when computing the
                         expectation or integral.
            use_expectation: If True, this samples baselines and interpolation
                             constants uniformly at random (expected gradients).
                             If False, then this assumes num_refs=1 in which
                             case it uses the same baseline for all inputs,
                             or num_refs=batch_size, in which case it uses
                             baseline[i] for inputs[i] and takes 100 linearly spaced
                             points between baseline and input (integrated gradients).
            output_indices:  If this is None, then this function returns the
                             attributions for each output class. This is rarely
                             what you want for classification tasks. Pass an
                             integer tensor of shape [batch_size] to
                             index the output output_indices[i] for
                             the input inputs[i].
            interaction_index: Either None or an index into the input. If the latter,
                               will compute the interactions with respect to that
                               feature. This parameter should index into a batch
                               of inputs as inputs[(slice(None) + interaction_index)].
                               For example, if you had images of shape (32, 32, 3)
                               and you wanted interactions with respect
                               to pixel (i, j, c), you should pass
                               interaction_index=[i, j, c].
        """
        interactions, is_multi_output, num_classes = self._init_array(inputs,
                                                                      output_indices,
                                                                      interaction_index,
                                                                      True)

        input_iterable = enumerate(inputs)
        if verbose:
            input_iterable = enumerate(tqdm(inputs))

        for i, current_input in input_iterable:
            current_alphas = self._sample_alphas(num_samples,
                                                 use_expectation,
                                                 use_product=True)

            if not use_expectation and baseline.shape[0] > 1:
                current_baseline = np.expand_dims(baseline[i], axis=0)
            else:
                current_baseline = baseline

            if is_multi_output:
                if output_indices is not None:
                    output_index = output_indices[i]
                    current_interactions = self._single_interaction(current_input,
                                                                    current_baseline,
                                                                    current_alphas,
                                                                    num_samples,
                                                                    batch_size,
                                                                    use_expectation,
                                                                    output_index,
                                                                    interaction_index)
                    interactions[i] = current_interactions
                else:
                    for output_index in range(num_classes):
                        current_interactions = self._single_interaction(current_input,
                                                                        current_baseline,
                                                                        current_alphas,
                                                                        num_samples,
                                                                        batch_size,
                                                                        use_expectation,
                                                                        output_index,
                                                                        interaction_index)
                        interactions[output_index, i] = current_interactions
            else:
                current_interactions = self._single_interaction(current_input,
                                                                current_baseline,
                                                                current_alphas,
                                                                num_samples,
                                                                batch_size,
                                                                use_expectation,
                                                                None,
                                                                interaction_index)
                interactions[i] = current_interactions
        return interactions
