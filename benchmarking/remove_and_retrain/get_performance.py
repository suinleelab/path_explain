import sys
import os
sys.path.append('..')
sys.path.append('../../examples/tabular/')

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from time import sleep
from sklearn.model_selection import train_test_split

from utils import get_performance, get_interaction_model
from build_model import interaction_model
from path_explain.utils import set_up_environment, softplus_activation
from heart_disease.preprocess import heart_dataset
from pulsar.preprocess import pulsar_dataset
from interactions import return_interaction_function

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'heart_disease',
                    'One of `heart_disease`, `pulsar`, `simulated_2`, `simulated_5`, `simlated_10`')
flags.DEFINE_string('visible_devices', '0', 'GPU to use')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train for')
flags.DEFINE_string('interaction_type', 'integrated_hessians', 'type to use')
flags.DEFINE_boolean('train_interaction_model', False, 'Set to true to train the interaction model from scratch.')

def get_data(dataset):
    if dataset == 'heart_disease':
        x_train, y_train, x_test, y_test, _, _, _ = \
            heart_dataset(dir='/homes/gws/psturm/path_explain/examples/tabular/heart_disease/heart.csv')
    elif dataset == 'pulsar':
        x_train, y_train, x_test, y_test, _, _, _ = \
            pulsar_dataset(dir='/homes/gws/psturm/path_explain/examples/tabular/pulsar/pulsar_stars.csv')
    elif 'simulated' in dataset:
        dataset = np.load('data/{}.npz'.format(dataset))
        x_train, y_train = dataset['x_train'], dataset['y_train']
        x_test,  y_test  = dataset['x_test'],  dataset['y_test']
    else:
        raise ValueError('Unrecognized value `{}` for parameter `dataset`'.format(FLAGS.dataset))

    return x_train, y_train, x_test, y_test

def get_performance_curve(x_train, y_train, x_test, y_test,
                          best_param, interactions):
    print('Getting standard performance')
    mean_test_auc, sd_test_auc = get_performance(x_train, y_train, x_test, y_test,
                                                 best_param=best_param,
                                                 regression=('simulated' in FLAGS.dataset),
                                                 interactions_to_ignore=None)

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
        num_removed.append(num + 1)
        mean_test_auc, sd_test_auc = get_performance(x_train, y_train, x_test, y_test,
                                                     best_param=best_param,
                                                     regression=('simulated' in FLAGS.dataset),
                                                     interactions_to_ignore=cumulative_pairs)
        mean_performances.append(mean_test_auc)
        sd_performances.append(sd_test_auc)

    return mean_performances, sd_performances, cumulative_pairs, num_removed

def train_interaction_model(x_train, y_train, x_test, y_test):
    print('Training interaction model')
    set_up_environment(visible_devices=FLAGS.visible_devices)
    interaction_model = get_interaction_model(x_train, y_train, x_test, y_test,
                                              regression=('simulated' in FLAGS.dataset))
    tf.keras.models.save_model(interaction_model, 'models/{}.h5'.format(FLAGS.dataset))

def load_interaction_model():
    counter = 0
    while True:
        counter += 1
        try:
            model = tf.keras.models.load_model('models/{}.h5'.format(FLAGS.dataset))
            print('Successfully restored trained model.')
            break
        except (FileNotFoundError, OSError):
            print('({}) Did not find saved model. Will try again in 30 seconds...'.format(counter))
        sleep(60)
    return model

def main(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)
    x_train, y_train, x_test, y_test = get_data(FLAGS.dataset)

    if FLAGS.train_interaction_model:
        train_interaction_model(x_train, y_train, x_test, y_test)

    interaction_model = load_interaction_model()
    best_param = {
        'learning_rate': 0.005,
        'epochs': FLAGS.epochs,
        'num_layers': 2,
        'hidden_layer_size': 4
    }

    interaction_types = ['integrated_hessians',
                         'expected_hessians',
                         'hessians',
                         'hessians_times_inputs',
                         'shapley_sampling',
                         'contextual_decomposition',
                         'neural_interaction_detection']
    if FLAGS.interaction_type not in interaction_types:
        raise ValueError('Invalid interaction type `{}`'.format(FLAGS.interaction_type))

    type_list = []
    perf_list = []
    sd_list   = []
    pairs_list = []
    num_list  = []
    print('Evaluating {}'.format(FLAGS.interaction_type))
    print('Getting interactions')
    interaction_function = return_interaction_function(FLAGS.interaction_type)
    interactions = interaction_function(interaction_model, x_test, baseline=x_train)

    mean_performances, sd_performances, cumulative_pairs, num_removed = \
        get_performance_curve(x_train, y_train, x_test, y_test,
                              best_param, interactions)

    perf_list += mean_performances
    sd_list += sd_performances
    pairs_list += cumulative_pairs
    num_list += num_removed
    type_list += [FLAGS.interaction_type] * len(mean_performances)

    data = pd.DataFrame({
        'interaction_type': type_list,
        'mean_perf': perf_list,
        'sd_perf': sd_list,
        'cumulative_pairs': pairs_list,
        'num_interactions_removed': num_list
    })
    data.to_csv('results/{}_{}.csv'.format(FLAGS.dataset, FLAGS.interaction_type))

if __name__ == '__main__':
    tf.autograph.set_verbosity(0)
    app.run(main)
