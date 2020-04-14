import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau

from contextual_decomposition import ContextualDecompositionExplainerTF
from gradients import GradientExplainerTF
from neural_interaction_detection import NeuralInteractionDetectionExplainerTF
from path_explain import PathExplainerTF, softplus_activation

def build_model(num_features,
                units=[64, 64, 64],
                activation_function=tf.keras.activations.relu,
                output_units=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_features,)))
    for unit in units:
        model.add(tf.keras.layers.Dense(unit))
        model.add(tf.keras.layers.Activation(activation_function))
    model.add(tf.keras.layers.Dense(output_units))

    return model

def build_data(num_features=20,
               num_interacting_pairs=10,
               num_samples=5000):

    x = np.random.randn(num_samples, num_features)
    y = np.zeros(num_samples)

    all_pairs = np.array(np.meshgrid(np.arange(num_features),
                                     np.arange(num_features))).T.reshape(-1, 2)
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
    pair_indices = np.random.choice(all_pairs.shape[0],
                                    size=num_interacting_pairs,
                                    replace=False)
    chosen_pairs = all_pairs[pair_indices]
    for pair in chosen_pairs:
        y += np.prod(x[:, pair], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test, chosen_pairs

def get_interactions(x_train, y_train, x_test, y_test, model):
    num_features = x_train.shape[1]

    ###### Integrated and Expected Hessians ######
    interpret_model = tf.keras.models.clone_model(model)
    for layer in interpret_model.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            layer.activation = softplus_activation(beta=10.0)
    print(interpret_model.summary())

    path_explainer  = PathExplainerTF(interpret_model)

    ih_interactions = path_explainer.interactions(inputs=x_test,
                                                  baseline=np.zeros((1, num_features)).astype(np.float32),
                                                  batch_size=100,
                                                  num_samples=200,
                                                  use_expectation=False,
                                                  output_indices=0,
                                                  verbose=True,
                                                  interaction_index=None)
    eh_interactions = path_explainer.interactions(inputs=x_test,
                                                  baseline=x_train,
                                                  batch_size=100,
                                                  num_samples=200,
                                                  use_expectation=True,
                                                  output_indices=0,
                                                  verbose=True,
                                                  interaction_index=None)

    ###### Contextual Decomposition ######
    cd_explainer = ContextualDecompositionExplainerTF(model)
    cd_attr_beta, cd_attr_gamma = cd_explainer.attributions(inputs=x_test,
                                                batch_size=100,
                                                output_indices=0)
    cd_group_beta, cd_group_gamma = cd_explainer.interactions(inputs=x_test,
                                                              batch_size=100,
                                                              output_indices=0)

    #Subtract feature attributions from group attributions, as discussed in the original paper
    cd_interactions_beta = cd_group_beta - cd_attr_beta[:, :, np.newaxis] - cd_attr_beta[:, np.newaxis, :]
    cd_interactions_beta[:, np.arange(num_features), np.arange(num_features)] = cd_attr_beta

    cd_interactions_gamma = cd_group_gamma - cd_attr_gamma[:, :, np.newaxis] - cd_attr_gamma[:, np.newaxis, :]
    cd_interactions_gamma[:, np.arange(num_features), np.arange(num_features)] = cd_attr_gamma

    ###### Neural Interaction Detection ######
    nid_explainer = NeuralInteractionDetectionExplainerTF(model)
    nid_interactions = nid_explainer.interactions(output_index=0,
                                                  verbose=False,
                                                  inputs=x_test,
                                                  batch_size=100)

    ###### Input Hessian ######
    grad_explainer = GradientExplainerTF(interpret_model)
    hess_interactions = grad_explainer.interactions(inputs=x_test,
                                                    multiply_by_input=False,
                                                    batch_size=100,
                                                    output_index=0)
    hess_times_inp_interactions = grad_explainer.interactions(inputs=x_test,
                                                              multiply_by_input=True,
                                                              batch_size=100,
                                                              output_index=0)

    interaction_dict = {
        'integrated_hessians': ih_interactions,
        'expected_hessians': eh_interactions,
        'contextual_decomposition_beta': cd_interactions_beta,
        'contextual_decomposition_gamma': cd_interactions_gamma,
        'neural_interaction_detection': nid_interactions,
        'hessian': hess_interactions,
        'hessian_times_input': hess_times_inp_interactions
    }

    # Zero diagonals
    for key in interaction_dict:
        interaction_dict[key][:, np.arange(num_features), np.arange(num_features)] = 0.0

    return interaction_dict

def get_metrics(x_test,
                interaction_dict,
                chosen_pairs):
    pair_interactions = []
    for pair in chosen_pairs:
        pair_interactions.append(np.prod(x_test[:, pair], axis=1))
    pair_interactions = np.stack(pair_interactions, axis=1)

    interaction_ordering = np.argsort(np.abs(pair_interactions), axis=1)
    maximum_interaction_index = interaction_ordering[:, -1]
    maximum_interaction_pair = chosen_pairs[maximum_interaction_index]

    metric_dict = {}
    for key in interaction_dict:
        interaction = interaction_dict[key]
        abs_interaction = np.abs(interaction)

        current_max_pair = abs_interaction.reshape(abs_interaction.shape[0], -1).argmax(1)
        current_max_pair = np.column_stack(np.unravel_index(current_max_pair,
                                                            abs_interaction[0].shape))

        mask_pairs = (current_max_pair == maximum_interaction_pair)
        top_1_accuracy = np.sum(np.sum(mask_pairs, axis=1) == 2) / len(mask_pairs)
        metric_dict[key] = top_1_accuracy

    return metric_dict

def train(x_train, y_train, x_test, y_test, model,
          learning_rate=0.01, epochs=30, batch_size=100):
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Variance of the test labels: {:.4f}'.format(np.var(y_test)))

def test_simple_multiplicative():
    x_train, y_train, x_test, y_test, chosen_pairs = build_data(num_features=5,
                                                                num_interacting_pairs=2,
                                                                num_samples=5000)
    model = build_model(num_features=x_train.shape[1])
    train(x_train, y_train, x_test, y_test, model)
    interaction_dict = get_interactions(x_train, y_train, x_test, y_test, model)
    metric_dict = get_metrics(x_test,
                              interaction_dict,
                              chosen_pairs)
    for key in metric_dict:
        print('{}: {:.4f}'.format(key, metric_dict[key]))

if __name__ == '__main__':
    test_simple_multiplicative()