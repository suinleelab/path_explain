"""
A module that sub-classes the original explainer to explain language
models through an embedding layer.
"""
import tensorflow as tf
import numpy as np
from .path_explainer_tf import PathExplainerTF

class EmbeddingExplainerTF(PathExplainerTF):
    """
    This class is designed to explain models that use an embedding layer,
    e.g. language models. It is very similar to the original path explainer,
    except that it sums over the embedding dimension at convient places
    to reduce dimensionality.
    """

    def __init__(self, model, embedding_axis=2):
        """
        Initialize the TF explainer class. This class
        will handle both the eager and the
        old-school graph-based tensorflow models.

        Args:
            model: A tf.keras.Model instance if executing eagerly.
                   A tuple (input_tensor, output_tensor) otherwise.
            embedding_dimension: The axis corresponding to the embeddings.
                                 Usually this is 2.
        """
        self.model = model
        self.eager_mode = False
        self.embedding_axis = embedding_axis
        try:
            self.eager_mode = tf.executing_eagerly()
        except AttributeError:
            pass

    def accumulation_function(self,
                              batch_input,
                              batch_baseline,
                              batch_alphas,
                              output_index=None,
                              second_order=False,
                              interaction_index=None):
        """
        A function that computes the logic of combining gradients and
        the difference from reference. This function is meant to
        be overloaded in the case of custom gradient logic. See PathExplainerTF
        for a description of the input.
        """
        if not second_order:
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

            ################################
            # This line is the only difference
            # for attributions. We sum over the embedding dimension.
            batch_attributions = tf.reduce_sum(batch_attributions, axis=self.embedding_axis)
            ################################

            return batch_attributions

        batch_alpha, batch_beta = batch_alphas
        batch_difference = batch_input - batch_baseline
        batch_interpolated_beta = batch_beta * batch_input + (1.0 - batch_beta) * batch_baseline

        with tf.GradientTape() as second_order_tape:
            second_order_tape.watch(batch_interpolated_beta)

            batch_difference_beta = batch_interpolated_beta - batch_baseline
            batch_interpolated_alpha = batch_alpha * batch_interpolated_beta + (1.0 - batch_alpha) * batch_baseline
            with tf.GradientTape() as first_order_tape:
                first_order_tape.watch(batch_interpolated_alpha)

                batch_predictions = self.model(batch_interpolated_alpha)
                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]

            batch_gradients = first_order_tape.gradient(batch_predictions, batch_interpolated_alpha)
            batch_gradients = batch_gradients * batch_difference_beta

            if interaction_index is not None:
                batch_gradients = batch_gradients[tuple([slice(None)] + interaction_index)]
            else:
                ################################
                # The first of two modifications for interactions.
                batch_gradients = tf.reduce_sum(batch_gradients, axis=self.embedding_axis)
                ################################

        if interaction_index is not None:
            batch_hessian = second_order_tape.gradient(batch_gradients, batch_interpolated_beta)
        else:
            batch_hessian = second_order_tape.batch_jacobian(batch_gradients, batch_interpolated_beta)

        if interaction_index is not None:
            batch_difference = batch_difference[tuple([slice(None)] + \
                                                                 interaction_index)]
            for _ in range(len(batch_input.shape) - 1):
                batch_difference = tf.expand_dims(batch_difference, axis=-1)
        else:
            batch_difference = tf.expand_dims(batch_difference, axis=1)

        batch_interactions = batch_hessian * batch_difference

        ################################
        # The second of two modifications for interactions.
        if interaction_index is None:
            # This axis computation is really len(input.shape) - 1 + self.embedding_axis - 1
            # The -1's are because we squashed a batch dimension and the first embedding dimension.

            hessian_embedding_axis = len(batch_input.shape) + self.embedding_axis - 2
            batch_interactions = tf.reduce_sum(batch_interactions, axis=hessian_embedding_axis)
        ################################

        return batch_interactions

    def _init_array(self,
                    inputs,
                    output_indices,
                    interaction_index=None,
                    as_interactions=False):
        """
        Internal helper function to get an
        array of the proper shape. This needs
        to be overloaded because the input shape is the
        embedding size, but we will be squashing the embedding dimensions.
        """
        test_output = self._get_test_output(inputs)
        is_multi_output = len(test_output.shape) > 1
        shape_tuple = inputs.shape[:2]
        num_classes = test_output.shape[-1]

        if as_interactions and interaction_index is None:
            shape_tuple = [inputs.shape[0], inputs.shape[1], inputs.shape[1]]
            shape_tuple = tuple(shape_tuple)

        if is_multi_output and output_indices is None:
            num_classes = test_output.shape[-1]
            attributions = np.zeros((num_classes,) + shape_tuple)
        elif not is_multi_output and output_indices is not None:
            raise ValueError('Provided output_indices but ' + \
                             'model is not multi output!')
        else:
            attributions = np.zeros(shape_tuple)

        return attributions, is_multi_output, num_classes
