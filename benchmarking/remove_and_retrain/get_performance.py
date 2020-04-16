import sys
import os
sys.path.append('..')
sys.path.append('../../examples/tabular/')

import tensorflow as tf
import numpy as np
import pandas as pd

from utils import get_performance
from path_explain.utils import set_up_environment, softplus_activation
from heart_disease.preprocess import heart_dataset
from pulsar.preprocess import pulsar_dataset
from interactions import return_interaction_function

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'heart_disease',
                    'One of `heart_disease`, `pulsar`, `simulated`')
flags.DEFINE_string('visible_devices', '0', 'GPU to use')
flags.DEFINE_integer('n_iter', 50, 'Number of hyper parameter settings to try')

def get_data():
    if FLAGS.dataset == 'heart_disease':
        x_train, y_train, x_test, y_test, _, _, _ = \
            heart_dataset(dir='/homes/gws/psturm/path_explain/examples/tabular/heart_disease/heart.csv')
    elif FLAGS.dataset == 'pulsar':
        x_train, y_train, x_test, y_test, _, _, _ = \
            pulsar_dataset(dir='/homes/gws/psturm/path_explain/examples/tabular/pulsar/pulsar_stars.csv')
    elif FLAGS.dataset == 'simulated':
        x = np.random.randn(2000, 10)
        y = np.logical_or.reduce([np.prod(x[0:2], axis=1) > 0,
                                  np.prod(x[1:3], axis=1) > 0,
                                  np.prod(x[4:6], axis=1) < 0])
        return x, y
    else:
        raise ValueError('Unrecognized value `{}` for parameter `dataset`'.format(FLAGS.dataset))

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y

def get_performance_curve(interaction_function, x, y):
    print('Getting standard performance')
    mean_test_auc, sd_test_auc, interactions = get_performance(x,
                                                               y,
                                                               random_seed=0,
                                                               n_iter=FLAGS.n_iter,
                                                               interactions_to_ignore=None,
                                                               interaction_function=interaction_function)
    interactions = np.mean(np.abs(interactions), axis=0)
    first_indices, second_indices = np.triu_indices(interactions.shape[0], k=1)
    flattened_interactions = interactions[first_indices, second_indices]

    sorted_indices = np.argsort(flattened_interactions)[::-1]

    first_indices  = first_indices[sorted_indices]
    second_indices = second_indices[sorted_indices]

    mean_performances = [mean_test_auc]
    sd_performances   = [sd_test_auc]

    cumulative_pairs = [(),]
    num_removed = []
    for num, pair in enumerate(zip(first_indices, second_indices)):
        print('Removing pair {}/{}'.format(num, len(first_indices)))
        cumulative_pairs.append(pair)
        num_removed.append(num)
        mean_test_auc, sd_test_auc = get_performance(x,
                                                     y,
                                                     random_seed=0,
                                                     n_iter=FLAGS.n_iter,
                                                     interactions_to_ignore=cumulative_pairs,
                                                     interaction_function=None)
        mean_performances.append(mean_test_auc)
        sd_performances.append(sd_test_auc)

    return mean_performances, sd_performances, cumulative_pairs, num_removed

def main(argv=None):
    x, y = get_data()
    interaction_types = ['integrated_hessians']
#                          'expected_hessians',
#                          'hessians',
#                          'hessians_times_inputs',
#                          'shapley_sampling',
#                          'contextual_decomposition',
#                          'neural_interaction_detection']
    type_list = []
    perf_list = []
    sd_list   = []
    pairs_list = []
    num_list  = []
    for interaction_type in interaction_types:
        print('Evaluating {}'.format(interaction_type))
        interaction_function = return_interaction_function(interaction_type)
        mean_performances, sd_performances, cumulative_pairs, num_removed = \
            get_performance_curve(interaction_function, x, y)

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
    data.to_csv('{}.csv'.format(FLAGS.dataset))

if __name__ == '__main__':
    tf.autograph.set_verbosity(0)
    app.run(main)