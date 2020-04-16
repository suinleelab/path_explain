import sys
import os
sys.path.append('..')
sys.path.append('../../examples/tabular/')

from path_explain import PathExplainerTF
from contextual_decomposition import ContextualDecompositionExplainerTF
from gradients import GradientExplainerTF
from neural_interaction_detection import NeuralInteractionDetectionExplainerTF
from shapley_sampling import SamplingExplainerTF
from path_explain.utils import softplus_activation

from build_model import get_subnetwork

def return_interaction_function(interaction_type='integrated_hessians'):
    if interaction_type == 'integrated_hessians':
        def interaction_function(model, data, baseline=None):
            interpret_model = tf.keras.models.clone_model(model)

            for layer in interpret_model.layers:
                if layer.get_config().get('activation', None) == 'relu':
                    layer.activation = softplus_activation(beta=10.0)

            explainer = PathExplainerTF(interpret_model)

            baseline = np.zeros((1, data.shape[1]))
            interactions = explainer.interactions(inputs=data,
                                                  baseline=baseline,
                                                  batch_size=100,
                                                  num_samples=100,
                                                  use_expectation=False,
                                                  output_indices=None,
                                                  verbose=False,
                                                  interaction_index=None)
            return interactions
    elif interaction_type == 'expected_hessians':
        def interaction_function(model, data, baseline=None):
            interpret_model = tf.keras.models.clone_model(model)

            for layer in interpret_model.layers:
                if layer.get_config().get('activation', None) == 'relu':
                    layer.activation = softplus_activation(beta=10.0)

            explainer = PathExplainerTF(interpret_model)


            interactions = explainer.interactions(inputs=data,
                                                  baseline=baseline,
                                                  batch_size=100,
                                                  num_samples=100,
                                                  use_expectation=True,
                                                  output_indices=None,
                                                  verbose=False,
                                                  interaction_index=None)
            return interactions
    elif interaction_type == 'hessians':
        def interaction_function(model, data, baseline=None):
            interpret_model = tf.keras.models.clone_model(model)

            for layer in interpret_model.layers:
                if layer.get_config().get('activation', None) == 'relu':
                    layer.activation = softplus_activation(beta=10.0)

            explainer = GradientExplainerTF(interpret_model)

            interactions = explainer.interactions(inputs=data,
                                                  multiply_by_input=False,
                                                  batch_size=100,
                                                  output_index=None,
                                                  verbose=False,
                                                  interaction_index=None)
            return interactions
    elif interaction_type == 'hessians_times_inputs':
        def interaction_function(model, data, baseline=None):
            interpret_model = tf.keras.models.clone_model(model)

            for layer in interpret_model.layers:
                if layer.get_config().get('activation', None) == 'relu':
                    layer.activation = softplus_activation(beta=10.0)

            explainer = GradientExplainerTF(interpret_model)

            interactions = explainer.interactions(inputs=data,
                                                  multiply_by_input=True,
                                                  batch_size=100,
                                                  output_index=None,
                                                  verbose=False,
                                                  interaction_index=None)
            return interactions
    elif interaction_type == 'shapley_sampling':
        def interaction_function(model, data, baseline=None):
            explainer = SamplingExplainerTF(model)

            baseline = np.zeros((1, data.shape[1]))
            interactions = explainer.interactions(inputs,
                                                  baselines=baseline,
                                                  batch_size=100,
                                                  number_of_samples=200,
                                                  output_index=None,
                                                  verbose=False)
            return interactions
    elif interaction_type == 'contextual_decomposition':
        def interaction_function(model, data, baseline=None):
            interaction_matrix = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
            for i in range(data.shape[1]):
                for j in range(i + 1, data.shape[1]):
                    subnetwork = get_subnetwork(model, input_features=(i, j))
                    explainer = ContextualDecompositionExplainerTF(subnetwork)
                    attributions, _ = explainer.attributions(fixed_values, batch_size=100)
                    interactions, _ = explainer.interactions(fixed_values, batch_size=100)
                    interactions = interactions - attributions[:, np.newaxis, :] - attributions[:, :, np.newaxis]
                    interaction_matrix += interactions
            return interaction_matrix
    elif interaction_type == 'neural_interaction_detection':
        def interaction_function(model, data, baseline=None):
            interaction_matrix = np.zeros((data.shape[0], data.shape[1], data.shape[1]))

            for i in range(data.shape[1]):
                for j in range(i + 1, data.shape[1]):
                    subnetwork = get_subnetwork(model, input_features=(i, j))
                    explainer = NeuralInteractionDetectionExplainerTF(subnetwork)
                    interactions = nid_explainer.interactions()
                    interaction_matrix[:, i, j] = interactions[0, 1]
                    interaction_matrix[:, j, i] = interactions[1, 0]
            return interaction_matrix
    else:
        raise ValueError('Unrecognized interaction type `{}`'.format(interaction_type))

    return interaction_function

if __name__ == '__main__':
    interaction_function = return_interaction_function('integrated_hessians')
    interaction_function = return_interaction_function('expected_hessians')
    interaction_function = return_interaction_function('hessians')
    interaction_function = return_interaction_function('hessians_times_inputs')
    interaction_function = return_interaction_function('shapley_sampling')
    interaction_function = return_interaction_function('contextual_decomposition')
    interaction_function = return_interaction_function('neural_interaction_detection')
