import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from contextual_decomposition import ContextualDecompositionExplainerTF
from gradients import GradientExplainerTF
from neural_interaction_detection import NeuralInteractionDetectionExplainerTF
from path_explain import PathExplainerTF, softplus_activation


def build_model(num_features,
                units=[128, 128],
                activation_function=tf.keras.activations.relu,
                output_units=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_features,)))
    for unit in units:
        model.add(tf.keras.layers.Dense(unit))
        model.add(tf.keras.layers.Activation(activation_function))
    model.add(tf.keras.layers.Dense(output_units))

    return model

def get_data(num_samples,
             num_features):
    x = np.random.randn(num_samples, num_features).astype(np.float32)
    return x

def benchmark_time():
    number_of_layers   = np.arange(2, 6)
    number_of_samples  = [100, 1000, 10000]
    number_of_features = [10, 50, 100]

    layer_array = []
    sample_array = []
    feature_array = []

    time_dict = {}
    for method in ['ih', 'eh', 'cd', 'nid', 'hess', 'hess_in']:
        for eval_type in ['all', 'row', 'pair']:
            time_dict[method + '_' + eval_type] = []

    for layer_count in number_of_layers:
        for sample_count in number_of_samples:
            for feature_count in number_of_features:
                print('Number of layers: {} - Number of samples: {} - Number of features: {}'.format(layer_count, sample_count, feature_count))
                model = build_model(num_features=feature_count,
                                    activation_function=softplus_activation(beta=10.0))
                data = get_data(sample_count, feature_count)

                ###### Integrated and Expected Hessians ######
                path_explainer  = PathExplainerTF(model)
                start_time = time.time()
                _ = path_explainer.interactions(inputs=data,
                                                baseline=np.zeros((1, feature_count)).astype(np.float32),
                                                batch_size=100,
                                                num_samples=200,
                                                use_expectation=False,
                                                output_indices=0,
                                                verbose=False,
                                                interaction_index=None)
                end_time = time.time()
                time_dict['ih_all'].append(end_time - start_time)

                start_time = time.time()
                _ = path_explainer.interactions(inputs=data,
                                                baseline=np.zeros((1, feature_count)).astype(np.float32),
                                                batch_size=100,
                                                num_samples=200,
                                                use_expectation=False,
                                                output_indices=0,
                                                verbose=False,
                                                interaction_index=0)
                end_time = time.time()
                time_dict['ih_row'].append(end_time - start_time)
                time_dict['ih_pair'].append(end_time - start_time)

                start_time = time.time()
                _ = path_explainer.interactions(inputs=data,
                                                baseline=np.zeros((200, feature_count)).astype(np.float32),
                                                batch_size=100,
                                                num_samples=200,
                                                use_expectation=True,
                                                output_indices=0,
                                                verbose=False,
                                                interaction_index=None)
                end_time = time.time()
                time_dict['eh_all'].append(end_time - start_time)

                start_time = time.time()
                ih_interactions = path_explainer.interactions(inputs=data,
                                                              baseline=np.zeros((200, feature_count)).astype(np.float32),
                                                              batch_size=100,
                                                              num_samples=200,
                                                              use_expectation=True,
                                                              output_indices=0,
                                                              verbose=False,
                                                              interaction_index=0)
                end_time = time.time()
                time_dict['eh_row'].append(end_time - start_time)
                time_dict['eh_pair'].append(end_time - start_time)

                ###### Contextual Decomposition ######
                cd_explainer = ContextualDecompositionExplainerTF(model)
                start_time = time.time()
                _ = cd_explainer.interactions(inputs=data,
                                              batch_size=100,
                                              output_indices=0,
                                              interaction_index=None)
                end_time = time.time()
                time_dict['cd_all'].append(end_time - start_time)

                start_time = time.time()
                _ = cd_explainer.interactions(inputs=data,
                                              batch_size=100,
                                              output_indices=0,
                                              interaction_index=0)
                end_time = time.time()
                time_dict['cd_row'].append(end_time - start_time)

                start_time = time.time()
                _ = cd_explainer.interactions(inputs=data,
                                              batch_size=100,
                                              output_indices=0,
                                              interaction_index=(0, 1))
                end_time = time.time()
                time_dict['cd_pair'].append(end_time - start_time)

                ###### Neural Interaction Detection ######
                nid_explainer = NeuralInteractionDetectionExplainerTF(model)
                start_time = time.time()
                _ = nid_explainer.interactions(output_index=0,
                                               verbose=False,
                                               inputs=data,
                                               batch_size=100)
                end_time = time.time()
                time_dict['nid_all'].append(end_time - start_time)

                start_time = time.time()
                _ = nid_explainer.interactions(output_index=0,
                                               verbose=False,
                                               inputs=data,
                                               batch_size=100,
                                               interaction_index=0)
                end_time = time.time()
                time_dict['nid_row'].append(end_time - start_time)

                start_time = time.time()
                _ = nid_explainer.interactions(output_index=0,
                                               verbose=False,
                                               inputs=data,
                                               batch_size=100,
                                               interaction_index=(0, 1))
                end_time = time.time()
                time_dict['nid_pair'].append(end_time - start_time)

                ###### Input Hessian ######
                grad_explainer = GradientExplainerTF(model)

                start_time = time.time()
                hess_interactions = grad_explainer.interactions(inputs=data,
                                                                multiply_by_input=False,
                                                                batch_size=100,
                                                                output_index=0)
                end_time = time.time()
                time_dict['hess_all'].append(end_time - start_time)

                start_time = time.time()
                hess_interactions = grad_explainer.interactions(inputs=data,
                                                                multiply_by_input=False,
                                                                batch_size=100,
                                                                output_index=0,
                                                                interaction_index=0)
                end_time = time.time()
                time_dict['hess_row'].append(end_time - start_time)
                time_dict['hess_pair'].append(end_time - start_time)

                start_time = time.time()
                hess_interactions = grad_explainer.interactions(inputs=data,
                                                                multiply_by_input=True,
                                                                batch_size=100,
                                                                output_index=0)
                end_time = time.time()
                time_dict['hess_in_all'].append(end_time - start_time)

                start_time = time.time()
                hess_interactions = grad_explainer.interactions(inputs=data,
                                                                multiply_by_input=True,
                                                                batch_size=100,
                                                                output_index=0,
                                                                interaction_index=0)
                end_time = time.time()
                time_dict['hess_in_row'].append(end_time - start_time)
                time_dict['hess_in_pair'].append(end_time - start_time)

                layer_array.append(layer_count)
                sample_array.append(sample_count)
                feature_array.append(feature_count)

    time_dict['hidden_layers'] = layer_array
    time_dict['number_of_samples'] = sample_array
    time_dict['number_of_features'] = feature_array
    time_df = pd.DataFrame(time_dict)
    time_df.to_csv('time.csv', index=False)

if __name__ == '__main__':
    benchmark_time()