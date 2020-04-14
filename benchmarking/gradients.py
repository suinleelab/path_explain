"""
A module to explain models using the input gradient
or the input hessian. Using the input hessian as
an interaction method hasn't been published, but seems
like an obvious if naive baseline.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

class GradientExplainerTF():
    def __init__(self, model):
        """
        Args:
            model: A tf.keras.Model instance.
        """
        self.model = model

    def attributions(self,
                     inputs,
                     multiply_by_input=False,
                     batch_size=50,
                     output_index=None,
                     verbose=False):
        """
        A function to compute attributions using the input gradient on the given
        inputs.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            multiply_by_input: Whether or not to multiply the gradient by the input.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            output_index: Set to an integer to index into the output. Otherwise sums the outputs.
        """
        attributions = np.zeros(inputs.shape)

        iterable = range(0, len(inputs), batch_size)
        if verbose:
            iterable = tqdm(iterable)

        for i in iterable:
            number_to_draw = min(batch_size, len(inputs) - i)
            batch_inputs = inputs[i:(i + number_to_draw)]
            batch_inputs = tf.convert_to_tensor(batch_inputs)
            if batch_inputs.dtype != tf.float32:
                batch_inputs = tf.cast(batch_inputs, tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(batch_inputs)
                batch_predictions = self.model(batch_inputs)
                if output_index is not None:
                    batch_predictions = batch_predictions[:, output_index]

            batch_gradients = tape.gradient(batch_predictions, batch_inputs)
            if multiply_by_input:
                batch_gradients = batch_gradients * batch_inputs

            attributions[i:(i + number_to_draw)] = batch_gradients
        return attributions

    def interactions(self,
                     inputs,
                     multiply_by_input=False,
                     batch_size=50,
                     output_index=None,
                     verbose=False,
                     interaction_index=None):
        """
        A function to compute interactions using the input hessian on the given input.

        Args:
            inputs: A tensor of inputs to the model of shape (batch_size, ...).
            multiply_by_input: Whether or not to multiply the gradient by the input.
            batch_size: The maximum number of inputs to input into the model
                        simultaneously.
            output_index: Set to an integer to index into the output. Otherwise sums the outputs.
        """
        num_samples = inputs.shape[0]
        num_features = inputs.shape[1]

        if interaction_index is not None:
            interactions = np.zeros((num_samples, num_features))
        else:
            interactions = np.zeros((num_samples, num_features, num_features))

        iterable = range(0, len(inputs), batch_size)
        if verbose:
            iterable = tqdm(iterable)

        for i in iterable:
            number_to_draw = min(batch_size, len(inputs) - i)
            batch_inputs = inputs[i:(i + number_to_draw)]
            batch_inputs = tf.convert_to_tensor(batch_inputs)
            if batch_inputs.dtype != tf.float32:
                batch_inputs = tf.cast(batch_inputs, tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(batch_inputs)
                with tf.GradientTape() as second_tape:
                    second_tape.watch(batch_inputs)

                    batch_predictions = self.model(batch_inputs)
                    if output_index is not None:
                        batch_predictions = batch_predictions[:, output_index]

                batch_gradients = second_tape.gradient(batch_predictions, batch_inputs)

                if interaction_index is not None:
                    batch_gradients = batch_gradients[:, interaction_index]

            if interaction_index is not None:
                batch_hessians = tape.gradient(batch_gradients, batch_inputs)
            else:
                batch_hessians = tape.batch_jacobian(batch_gradients, batch_inputs)

            if multiply_by_input:
                if interaction_index is not None:
                    batch_hessians = batch_hessians * batch_inputs
                else:
                    batch_hessians = batch_hessians * tf.expand_dims(batch_inputs, axis=2) * tf.expand_dims(batch_inputs, axis=1)

            interactions[i:(i + number_to_draw)] = batch_hessians
        return interactions