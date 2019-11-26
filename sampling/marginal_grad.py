import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class MarginalGradExplainer(object):
    '''
    A class for computing the conditional expectation of a function
    (model) given knowledge of certain features
    '''
    
    def __init__(self, model, data, nsamples, representation='mobius'):
        '''
        Initializes the class object.
        
        Args:
            model: A function or callable that can be called on input data to produce output.
            data:  A numpy array of input data to sample background references from.
                   Typically this represents the same matrix the model was trained on.
                   If len(data) < nsamples, then we will only draw len(data) samples. This
                   behavior is useful if you want a constant reference, e.g. a black image.
            nsamples: The number of samples to draw from the data when computing the expectation.
            
            representation: One of `mobius`, `comobius`, or `average`. Using `mobius`,
                            the main effects are represented as E[f|x_i]. Using `comobius`,
                            they are E[f|X_{N\i}]. Using `average` averages the two cases.
        '''
        self.model    = model
        self.data     = data
        self.nsamples = min(nsamples, len(data))
        self.representation     = representation
        
        if len(self.data) < self.nsamples:
            raise ValueError('Requested more samples than amount of background data provided!')
        
        if self.representation not in ['mobius', 'comobius', 'average']:
            raise ValueError('''Unrecognized value `{}` for argument representation. 
            Must be one of `mobius`, `comobius`, `average`.'''.format(self.representation))
    
    def _sample_background(self, number_to_draw):
        '''
        An internal function to sample data from a background distribution.
        
        Args:
            number_to_draw: The number of samples to draw.
        Returns:
            A data matrix of number_to_draw samples sampled from the background distribution.
        '''
        sample_indices = np.random.choice(self.nsamples, number_to_draw, replace=False)
        return self.data[sample_indices]
    
    def _index_predictions(self, predictions, labels):
        '''
        Indexes predictions, a [batch_size, num_classes]-shaped tensor, 
        by labels, a [batch_size]-shaped tensor that indicates which
        class each sample should be indexed by.

        Args:
            predictions: A [batch_size, num_classes]-shaped tensor. The input to a model.
            labels: A [batch_size]-shaped tensor. The tensor used to index predictions.
        Returns:
            A tensor of shape [batch_size] representing the predictions indexed by the labels.
        '''
        current_batch_size = tf.shape(predictions)[0]
        sample_indices = tf.range(current_batch_size)
        indices_tensor = tf.stack([sample_indices, tf.cast(labels, tf.int32)], axis=1)
        predictions_indexed = tf.gather_nd(predictions, indices_tensor)
        return predictions_indexed
        
    def explain(self, X, batch_size=50, verbose=False, 
                index_outputs=False, labels=None):
        '''
        Computes the main effects of the model on data X. 
        
        Args:
            X: A data matrix. The samples you want to compute
               the main effects for. The code here assumes that the
               data is an array of shape [num_samples, ...] where
               ... represent the dimensions of the input.
            batch_size: The batch size to use while calling the model.
            verbose:    Whether or not to log progress while doing computation.
            index_true_class: Whether or not to index the output of the model
                              by output_indices. This parameter is 
                              used if you have a multi-class problem, in which
                              case you have to pass which output class you
                              want to index into.
            labels:           An integer array of shape (len(X),) representing
                              which output classes to index into when
                              doing the computation. If set to None,
                              defaults to the maximum output for each class.
            
        Returns:
            A list of length len(output_indices). Each entry in the list
            is a matrix of shape [num_samples, len(feature_indices)], representing
            the main effects of each feature with respect to the indicated output class.
        '''
        accumulated_gradients = []
        for i in range(0, len(X), batch_size):
            number_to_draw = min(batch_size, len(X) - i)
            background    = self._sample_background(number_to_draw)
            current_batch = X[i:min(i + batch_size, len(X))]
            
            alpha = 1.0 / self.nsamples
            interpolated_batch = alpha * current_batch + (1.0 - alpha) * background
            interpolated_batch = tf.constant(interpolated_batch)
            
            with tf.GradientTape() as tape:
                tape.watch(interpolated_batch)
                predictions = self.model(interpolated_batch)

                if index_outputs:
                    if labels is not None:
                        predictions_indexed = self._index_predictions(predictions, labels)
                    else:
                        model_outputs = np.argmax(self.model(current_batch), axis=-1)
                        predictions_indexed = self._index_predictions(predictions, model_outputs)
                else:
                    predictions_indexed = predictions

            input_gradients = tape.gradient(predictions_indexed, interpolated_batch)
            input_gradients = input_gradients * interpolated_batch
            accumulated_gradients.append(input_gradients)
        
        return np.concatenate(accumulated_gradients, axis=0)
