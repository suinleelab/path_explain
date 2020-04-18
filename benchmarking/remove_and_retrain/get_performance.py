import sys
import os
sys.path.append('..')
sys.path.append('../../examples/tabular/')

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from time import sleep

from utils import get_performance, get_interactions, get_interaction_model
from build_model import interaction_model
from path_explain.utils import set_up_environment, softplus_activation
from heart_disease.preprocess import heart_dataset
from pulsar.preprocess import pulsar_dataset
from interactions import return_interaction_function

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'heart_disease',
                    'One of `heart_disease`, `pulsar`, `simulated_5`, `simlated_10`')
flags.DEFINE_string('visible_devices', '0', 'GPU to use')
flags.DEFINE_integer('n_iter', 50, 'Number of hyper parameter settings to try')
flags.DEFINE_string('interaction_type', 'integrated_hessians', 'type to use')
flags.DEFINE_boolean('train_interaction_model', False, 'Set to true to train the interaction model from scratch.')

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_data():
    if FLAGS.dataset == 'heart_disease':
        x_train, y_train, x_test, y_test, _, _, _ = \
            heart_dataset(dir='/homes/gws/psturm/path_explain/examples/tabular/heart_disease/heart.csv')
    elif FLAGS.dataset == 'pulsar':
        x_train, y_train, x_test, y_test, _, _, _ = \
            pulsar_dataset(dir='/homes/gws/psturm/path_explain/examples/tabular/pulsar/pulsar_stars.csv')
    elif FLAGS.dataset == 'simulated_5':
        x = np.random.randn(2000, 5).astype(np.float32)
        y = 2 * x[:, 0] * x[:, 1] - \
            x[:, 0] * x[:, 2] + \
            1.5 * x[:, 2] * x[:, 3] + \
            0.5 * x[:, 1] * x[:, 3] + \
            x[:, 4]
        return x, y
    elif FLAGS.dataset == 'simulated_10':
        x = np.random.randn(2000, 10).astype(np.float32)
        y = 2 * x[:, 0] * x[:, 1] - \
            x[:, 0] * x[:, 2] + \
            1.5 * x[:, 2] * x[:, 3] + \
            0.5 * x[:, 1] * x[:, 3] + \
            3 * x[:, 6] * x[:, 7] - \
            2 * x[:, 8] * x[:, 9] + \
            x[:, 4] - x[:, 5]
        return x, y
    else:
        raise ValueError('Unrecognized value `{}` for parameter `dataset`'.format(FLAGS.dataset))

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y

def get_performance_curve(interaction_function, x, y, best_param, interactions):
    print('Getting standard performance')
    mean_test_auc, sd_test_auc, = get_performance(x,
                                                  y,
                                                  random_seed=0,
                                                  n_iter=None,
                                                  interactions_to_ignore=None,
                                                  best_param=best_param)
    interactions = np.mean(np.abs(interactions), axis=0)
    first_indices, second_indices = np.triu_indices(interactions.shape[0], k=1)
    flattened_interactions = interactions[first_indices, second_indices]

    sorted_indices = np.argsort(flattened_interactions)[::-1]

    first_indices  = first_indices[sorted_indices]
    second_indices = second_indices[sorted_indices]

    mean_performances = [mean_test_auc]
    sd_performances   = [sd_test_auc]

    cumulative_pairs = [(),]
    num_removed = [0]
    for num, pair in enumerate(zip(first_indices, second_indices)):
        print('Removing pair {}, {}/{}'.format(pair, num, len(first_indices)))
        cumulative_pairs.append(pair)
        num_removed.append(num)
        mean_test_auc, sd_test_auc = get_performance(x,
                                                     y,
                                                     random_seed=0,
                                                     n_iter=FLAGS.n_iter,
                                                     interactions_to_ignore=cumulative_pairs,
                                                     best_param=best_param)
        mean_performances.append(mean_test_auc)
        sd_performances.append(sd_test_auc)

    return mean_performances, sd_performances, cumulative_pairs, num_removed

def train_interaction_model():
    print('Getting best hyper parameters')
    best_param, _ = get_performance(x,
                                    y,
                                    random_seed=0,
                                    n_iter=FLAGS.n_iter,
                                    interactions_to_ignore=None,
                                    best_param=None)
    save_obj(best_param, 'models/param_{}.pkl'.format(FLAGS.dataset))

    print('Training interaction model')
    set_up_environment(visible_devices=FLAGS.visible_devices)
    x, y = get_data()
    interaction_model = get_interaction_model(x, y)
    tf.keras.models.save_model('models/{}.h5'.format(FLAGS.dataset))

def load_interaction_model():
    counter = 0
    while True:
        counter += 1
        try:
            model = tf.keras.models.load_model('models/{}.h5'.format(FLAGS.datset))
            best_param = load_obj('models/param_{}.pkl'.format(FLAGS.dataset))
            print('Successfully restored trained model and training parameters.')
            break
        except FileNotFoundError:
            print('({}) Did not find saved model or parameters. Will try again in 30 seconds...'.format(counter))
        sleep(30)
    return model, best_param

def main(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)
    x, y = get_data()

    if FLAGS.train_interaction_model:
        train_interaction_model()

    interaction_model, best_param = load_interaction_model()

    interaction_types = ['integrated_hessians',
                         'expected_hessians',
                         'hessians',
                         'hessians_times_inputs',
                         'shapley_sampling',
                         'contextual_decomposition',
                         'neural_interaction_detection']
    if FLAGS.interaction_type not in interaction_types:
        raise ValueError('Invalid interaction type `{}`'.format(FLAGS.interaction_type))

#     for interaction_type in interaction_types:
    type_list = []
    perf_list = []
    sd_list   = []
    pairs_list = []
    num_list  = []
    print('Evaluating {}'.format(FLAGS.interaction_type))
    print('Getting interactions')
    interaction_function = return_interaction_function(FLAGS.interaction_type)
    interactions = get_interactions(x,
                                    y,
                                    interaction_model,
                                    interaction_function)

    mean_performances, sd_performances, cumulative_pairs, num_removed = \
        get_performance_curve(interaction_function, x, y, best_param, interactions)

    perf_list += mean_performances
    sd_list += sd_performances
    pairs_list += cumulative_pairs
    num_list += num_removed
    type_list += [interaction_type] * len(mean_performances)

    data = pd.DataFrame({
        'interaction_type': type_list,
        'mean_auc': perf_list,
        'sd_auc': sd_list,
        'cumulative_pairs': pairs_list,
        'num_interactions_removed': num_list
    })
    data.to_csv('results/{}_{}.csv'.format(FLAGS.dataset, FLAGS.interaction_type))

if __name__ == '__main__':
    tf.autograph.set_verbosity(0)
    app.run(main)
