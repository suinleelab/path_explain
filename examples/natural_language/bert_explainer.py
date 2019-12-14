"""
This file provides an example of how to sub-class the
PathExplainerTF class in order to use a custom gradient function.
"""
import tensorflow as tf
import numpy as np
from path_explain.path_explainer_tf import PathExplainerTF

class BertExplainerTF(PathExplainerTF):
    """
    A class for explaining the BERT-based models
    (https://arxiv.org/abs/1810.04805)
    from the https://github.com/huggingface/transformers
    repository.
    """

    def _get_test_output(self,
                        inputs):
        int_inputs = tf.cast(inputs[0:1], tf.int32)
        return self.model(int_inputs)[0]

    def _prepare_input(self,
                       batch_input):
        """
        A helper function to prepare input
        into the language model.
        """
        ########################
        # These lines are just lines to shape
        # the input to be fed into the model
        batch_ids = tf.cast(batch_input, tf.int64)
        batch_masks = tf.cast(tf.cast(batch_ids, tf.bool), tf.int64)
        batch_token_types = tf.zeros(batch_ids.shape)

        extended_attention_mask = batch_masks[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.model.bert.num_hidden_layers
        ########################
        return batch_ids, batch_token_types, \
               extended_attention_mask, head_mask

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
        be overloaded in the case of custom gradient logic.

        Args:
            batch_input: A batch of input to the model. The model
                         should be able to be called as self.model(batch_input).
            batch_baseline: A batch of input to the model representing the
                            baseline input.
            batch_alphas: A batch of interpolation constants.
            output_index: An integer. Which output to index into. If None,
                          will take the gradient with respect to the
                          sum of the outputs.
            second_order: Set to True to return the hessian rather than
                          the gradient.
            interaction_index: An index into the features of the input. See
                               self.interactions for a complete description
                               of what this argument should be.
        """
        batch_ids, batch_token_types, \
        extended_attention_mask, head_mask = self._prepare_input(batch_input)
        batch_embedding = self.model.bert.embeddings([batch_ids,
                                                      None,
                                                      batch_token_types])
        batch_embedding = tf.cast(batch_embedding, tf.float64)
        baseline_ids = tf.cast(batch_baseline, tf.int64)
        baseline_embedding = self.model.bert.embeddings([baseline_ids,
                                                         None,
                                                         batch_token_types])
        baseline_embedding = tf.cast(baseline_embedding, tf.float64)

        if not second_order:
            # Add a dimension to account for the embedding dimension
            batch_alphas = tf.expand_dims(batch_alphas, axis=-1)
            batch_difference = batch_embedding - baseline_embedding
            batch_interpolated = batch_alphas * batch_embedding + \
                                 (1.0 - batch_alphas) * baseline_embedding
            batch_grad_input = (batch_interpolated, extended_attention_mask, head_mask)

            batch_gradients = self.gradient_function(batch_grad_input,
                                                     output_index)
            batch_attributions = batch_gradients * batch_difference
            batch_attributions = np.sum(batch_attributions, axis=-1)
            return batch_attributions

        batch_alpha, batch_beta = batch_alphas
        batch_product = batch_alpha * batch_beta
        batch_product = tf.expand_dims(batch_product, axis=-1)

        batch_difference = batch_embedding - baseline_embedding
        batch_interpolated = batch_product * batch_embedding + \
                             (1.0 - batch_product) * baseline_embedding
        batch_grad_input = (batch_interpolated, extended_attention_mask, head_mask)

        batch_gradients, batch_hessian = self.gradient_function(batch_grad_input,
                                               output_index,
                                               second_order=True,
                                               interaction_index=interaction_index)

        ########################
        # This code is just preparing arrays to be the right shape for broadcasting.
        if interaction_index is not None:
            batch_differences_secondary = batch_difference[tuple([slice(None)] + \
                                                                 interaction_index + \
                                                                 [slice(None)])]
            batch_differences_secondary = tf.expand_dims(batch_differences_secondary,
                                                         axis=1)
        else:
            batch_differences_secondary = batch_difference
            for _ in range(len(batch_embedding.shape) - 1):
                batch_product = tf.expand_dims(batch_product, axis=-1)
                batch_difference = tf.expand_dims(batch_difference, axis=-1)
                batch_differences_secondary = tf.expand_dims(batch_differences_secondary,
                                                             axis=1)
        ########################
        off_diagonal_attributions = batch_hessian * batch_product * \
                                    batch_difference * batch_differences_secondary
        diagonal_derivative = self._get_diagonal_derivatives(batch_hessian,
                                                             batch_gradients,
                                                             interaction_index)
        batch_attributions = off_diagonal_attributions + \
                             batch_difference * diagonal_derivative

        if interaction_index is not None:
            batch_attributions = np.sum(batch_attributions, axis=-1)
        else:
            batch_attributions = np.sum(batch_attributions, axis=(2, 4))
        return batch_attributions

    def gradient_function(self, batch_input,
                           output_index=None,
                           second_order=False,
                           interaction_index=None):
        """
        We overload the gradient function because you
        can't directly take the gradient through the
        embedding layer of a language model. Instead,
        we write some custom logic to take the gradient
        first with respect to the embedding, and then
        we sum the gradients with respect to each embedding.

        Args:
            batch_input: A batch of input. A dictionary containing
                         the keys 'input_ids', 'attention_mask',
                         'token_type_ids'.
        """
        batch_embedding, extended_attention_mask, head_mask = batch_input

        if not second_order:
            with tf.GradientTape() as tape:
                tape.watch(batch_embedding)

                batch_encoded = self.model.bert.encoder([batch_embedding,
                                                         extended_attention_mask,
                                                         head_mask])
                batch_sequence = batch_encoded[0]
                batch_pooled = self.model.bert.pooler(batch_sequence)
                batch_predictions = self.model.classifier(batch_pooled)

                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]
            batch_gradients = tape.gradient(batch_predictions, batch_embedding)
            return batch_gradients

        ## Handle the second order derivatives here
        with tf.GradientTape() as second_order_tape:
            second_order_tape.watch(batch_embedding)
            with tf.GradientTape() as first_order_tape:
                first_order_tape.watch(batch_embedding)

                batch_encoded = self.model.bert.encoder([batch_embedding,
                                                         extended_attention_mask,
                                                         head_mask])
                batch_sequence = batch_encoded[0]
                batch_pooled = self.model.bert.pooler(batch_sequence)
                batch_predictions = self.model.classifier(batch_pooled)

                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]

            batch_gradients = first_order_tape.gradient(batch_predictions, batch_embedding)
            # Same shape as the embedding layer
            if interaction_index is not None:
                batch_gradients = batch_gradients[tuple([slice(None)] + \
                                                        interaction_index + \
                                                        [slice(None)])]
        if interaction_index is not None:
            batch_hessian = second_order_tape.gradient(batch_gradients, batch_embedding)
        else:
            batch_hessian = second_order_tape.jacobian(batch_gradients, batch_embedding)
            batch_hessian = batch_hessian.numpy()

            hessian_index_array = [np.arange(batch_input.shape[0])] + \
                                  [slice(None)] * (len(batch_embedding.shape) - 1)
            hessian_index_array = 2 * hessian_index_array
            batch_hessian = batch_hessian[tuple(hessian_index_array)]

        return batch_gradients, batch_hessian