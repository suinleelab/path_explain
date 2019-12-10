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
        ########################
        # These lines are just lines to shape
        # the input to be fed into the model
        batch_ids = tf.cast(batch_input, tf.int32)
        batch_masks = tf.cast(tf.cast(batch_ids, tf.bool), tf.int32)
        batch_token_types = tf.zeros(batch_ids.shape)

        extended_attention_mask = batch_masks[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.model.bert.num_hidden_layers
        ########################

        batch_embedding = self.model.bert.embeddings([batch_ids,
                                                      None,
                                                      batch_token_types])
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
            batch_gradients = np.sum(batch_gradients, axis=-1)
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
            batch_gradients = tf.reduce_sum(batch_gradients, axis=-1)

            # Same shape as the embedding layer
            if interaction_index is not None:
                batch_gradients = batch_gradients[tuple([slice(None)] + interaction_index)]

        if interaction_index is not None:
            batch_hessian = second_order_tape.gradient(batch_gradients, batch_input)
            batch_hessian = np.sum(batch_hessian, axis=-1)
        else:
            batch_hessian = second_order_tape.jacobian(batch_gradients, batch_input)
            batch_hessian = batch_hessian.numpy()

            hessian_index_array = [np.arange(batch_input.shape[0])] + \
                                  [slice(None)] * (len(batch_embedding.shape) - 1)
            hessian_index_array = 2 * hessian_index_array
            batch_hessian = batch_hessian[tuple(hessian_index_array)]
            batch_hessian = np.sum(batch_hessian, axis=(2, 4))

        return batch_gradients, batch_hessian