import warnings
import numpy as np
from tqdm import tqdm

class MarginalExplainer(object):
    '''
    A class for computing the conditional expectation of a function
    (model) given knowledge of certain features
    '''

    def __init__(self, model, data, nsamples, feature_dependence='independent', representation='mobius'):
        '''
        Initializes the class object.

        Args:
            model: A function or callable that can be called on input data to produce output.
            data:  A numpy array of input data to sample background references from.
                   Typically this represents the same matrix the model was trained on.
                   If len(data) < nsamples, then we will only draw len(data) samples. This
                   behavior is useful if you want a constant reference, e.g. a black image.
            nsamples: The number of samples to draw from the data when computing the expectation.
            feature_dependence: One of `independent`, `dependent`. This parameter
                                controls how the explainer samples background samples.
                                In the former case, it simply draws from the data uniformly at random.
                                In the latter case, it draws samples using the (approximate)
                                conditional distribution.
            representation: One of `mobius`, `comobius`, or `average`. Using `mobius`,
                            the main effects are represented as E[f|x_i]. Using `comobius`,
                            they are E[f|X_{N\i}]. Using `average` averages the two cases.
        '''
        self.model    = model
        self.data     = data
        self.nsamples = min(nsamples, len(data))
        self.feature_dependence = feature_dependence
        self.representation     = representation

        if len(self.data) < self.nsamples:
            raise ValueError('Requested more samples than amount of background data provided!')

        if self.feature_dependence not in ['independent', 'dependent']:
            raise ValueError('''Unrecognized value `{}` for argument feature_dependence.
            Must be one of `independent`, `dependent`.'''.format(self.feature_dependence))

        if self.feature_dependence == 'dependent':
            warnings.warn('You have specified dependent feature sampling. Note that this ' + \
                          'is current performed by sorting the data for every sample and ' + \
                          'every feature, which is quite slow.')

        if self.representation not in ['mobius', 'comobius', 'average']:
            raise ValueError('''Unrecognized value `{}` for argument representation.
            Must be one of `mobius`, `comobius`, `average`.'''.format(self.representation))

    def _sample_background(self, number_to_draw, target_example, feature_index):
        '''
        An internal function to sample data from a background distribution.

        Args:
            number_to_draw: The number of samples to draw.
            target_example: The current example x.
            feature_index: Which feature (or set of features) represent the
                           target set.
        Returns:
            A data matrix of number_to_draw samples sampled from the background distribution.
        '''
        if self.feature_dependence == 'independent':
            sample_indices = np.random.choice(self.nsamples, number_to_draw, replace=False)
            return self.data[sample_indices]
        else:
            #Right now the way I'm doing this is by sorting examples with respect to their
            #feature-wise difference to the target feature. This is slow and should be improved,
            #but it was easy to code so here we are.
            abs_diff_from_feature = np.abs(target_example[tuple([np.newaxis] + list(feature_index))] - \
                                           self.data[tuple([slice(None)] + list(feature_index))])
            #Note: this slice(None) business is used to index the sample_indices
            #tensor using the feature_index tensor, which represents a list
            #the same length as the number of dimensions the data.
            abs_diff_ranking      = np.argsort(abs_diff_from_feature)[::-1]
            return self.data[abs_diff_from_ranking[number_to_draw]]

    def _construct_sample_vector(self, target_example, background_samples, feature_index):
        '''
        An internal function to compute the vector x_S or x_{N\S}, or both.

        Args:
            target_example: The current example x.
            background_samples: A set of input vectors that represent
                                the background distribution.
            feature_index: Which feature (or set of features) represent the
                           target set

        Returns:
            The target vector x_S or x_{N\S}, or in the case of
            `average`, a tuple containing (x_S, x_{N\S}).
        '''
        mobius_vector = background_samples.copy()
        mobius_vector[tuple([slice(None)] + list(feature_index))] = \
            target_example[tuple(feature_index)]

        comobius_vector = target_example.copy()
        comobius_vector = np.expand_dims(comobius_vector, axis=0)
        comobius_vector = np.tile(comobius_vector, [background_samples.shape[0]] + [1] * len(feature_index))

        comobius_vector[tuple([slice(None)] + list(feature_index))] = \
            background_samples[tuple([slice(None)] + list(feature_index))]

        if self.representation == 'mobius':
            return mobius_vector
        elif self.representation == 'comobius':
            return comobius_vector
        else:
            return (mobius_vector, comobius_vector)

    def explain(self, X, feature_indices=None, batch_size=50, verbose=False,
                index_outputs=False, labels=None):
        '''
        Computes the main effects of the model on data X.

        Args:
            X: A data matrix. The samples you want to compute
               the main effects for. The code here assumes that the
               data is an array of shape [num_samples, ...] where
               ... represent the dimensions of the input.
            feature_indices: The indices of features whose main effects
                             you want to compute. Defaults to all features.
                             If your data X has multiple dimensions (e.g.
                             [width, height, channels] for images), this should
                             be a 2D array where the second dimension is
                             equal to the number of dimensions in the data
                             (something like [#indices, 3] for color images).
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
        if feature_indices is None:
            #This one-liner computes an array of size
            #[np.prod(X.shape), len(X.shape)] where the
            #rows represent all possible indices into X
            sample_shape = X.shape[1:]
            feature_indices = np.swapaxes(np.reshape(np.indices(sample_shape),
                                          (len(sample_shape), np.prod(sample_shape))), 0, 1)
        test_output = self.model(X[0:1])
        if len(test_output.shape) > 1 and test_output.shape[-1] > 1:
            is_multi_class = True
            if labels is None:
                labels = []
                for i in range(0, len(X), batch_size):
                    labels_batch = self.model(X[i:min(i + batch_size, len(X))])
                    labels.append(labels_batch)
                labels = np.concatenate(labels, axis=0)
                labels = np.argmax(labels, axis=-1)
        else:
            is_multi_class = False

        main_effects = np.full(X.shape, np.nan)

        data_iterable = enumerate(X)
        if verbose:
            data_iterable = enumerate(tqdm(X))

        for j, target_example in data_iterable:
            for feature_index in feature_indices:
                for i in range(0, self.nsamples, batch_size):
                    number_to_draw     = min(self.nsamples, i + batch_size) - i
                    background_samples = self._sample_background(number_to_draw, target_example, feature_index)
                    sample_vector      = self._construct_sample_vector(target_example, background_samples, feature_index)

                    if self.representation == 'mobius':
                        sample_vector_out  = self.model(sample_vector)
                        background_out     = self.model(background_samples)
                        if is_multi_class:
                            sample_vector_out = sample_vector_out[:, labels[j]]
                            background_out    = background_out[:, labels[j]]

                        difference    = np.sum(sample_vector_out) - np.sum(background_out)
                    elif self.representation == 'comobius':
                        #I've hacked a quick solution here: multiply the baseline v(N) by the number of samples
                        #drawn for v(N\{i}). This technically works to put them on the same magnitude,
                        #but is numerically unstable. It would be better to actually perform the mean
                        #calculations over the sampling and keep v(N) as a stable quantity.
                        target_vector_out = self.model(np.expand_dims(target_example, axis=0))
                        sample_vector_out = self.model(sample_vector)
                        if is_multi_class:
                            target_vector_out = target_vector_out[:, labels[j]]
                            sample_vector_out    = sample_vector_out[:, labels[j]]

                        difference    = number_to_draw * np.sum(target_vector_out) - \
                                        np.sum(sample_vector_out)
                    else:
                        sample_vector_out   = self.model(sample_vector[0])
                        background_out      = self.model(background_samples)
                        target_vector_out   = self.model(np.expand_dims(target_example, axis=0))
                        cosample_vector_out = self.model(sample_vector[1])
                        if is_multi_class:
                            sample_vector_out   = sample_vector_out[:, labels[j]]
                            background_out      = background_out[:,    labels[j]]
                            target_vector_out   = target_vector_out[:, labels[j]]
                            cosample_vector_out = cosample_vector_out[:, labels[j]]

                        mobius_diff   = np.sum(sample_vector_out) - np.sum(background_out)
                        comobius_diff = number_to_draw * np.sum(target_vector_out) - \
                                        np.sum(cosample_vector_out)
                        difference    = (mobius_diff + comobius_diff) * 0.5

                    # Since we can't add anything to np.nan, we have to first check
                    if np.isnan(main_effects[tuple([j] + list(feature_index))]):
                        main_effects[tuple([j] + list(feature_index))] = 0.0
                    main_effects[tuple([j] + list(feature_index))] += difference

        main_effects /= self.nsamples
        return main_effects
