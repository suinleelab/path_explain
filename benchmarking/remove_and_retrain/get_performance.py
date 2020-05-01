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

from utils import get_performance, get_default_model, compile_model
from path_explain.utils import set_up_environment, softplus_activation
from heart_disease.preprocess import heart_dataset
from pulsar.preprocess import pulsar_dataset
from interactions import return_interaction_function

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'simulated_compile_test', 'See data directory for results')
flags.DEFINE_string('visible_devices', '0', 'GPU to use')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train for')
flags.DEFINE_integer('num_iters', 25, 'Number of iterations to average over while getting performance')
flags.DEFINE_string('interaction_type', 'integrated_hessians', 'type to use')
flags.DEFINE_boolean('train_interaction_model', False, 'Set to true to train the interaction model from scratch.')
flags.DEFINE_boolean('use_random_draw', False, 'Set to True to replace interactions with random guassian noise rather than the interaction distribution')

def get_data(dataset_name):
    dataset = np.load('data/{}.npz'.format(dataset_name))
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_test,  y_test  = dataset['x_test'],  dataset['y_test']

    spec_df = pd.read_csv('data/{}_spec.csv'.format(dataset_name))
    return x_train, y_train, x_test, y_test, spec_df

def get_performance_curve(x_train, x_test, model, spec_df,
                          interactions_train, interactions_test, random_weights):
    total_possible_pairs = int(x_train.shape[1] * (x_train.shape[1] - 1) / 2)
    print('Getting standard performance')


    mean_performances = []
    sd_performances = []
    for k in range(total_possible_pairs):
        print('Iteration {}/{}'.format(k, total_possible_pairs))
        mean_perf, sd_perf = get_performance(x_train,
                                             x_test,
                                             model,
                                             random_weights,
                                             spec_df,
                                             interactions_train=interactions_train,
                                             interactions_test=interactions_test,
                                             k=k,
                                             num_iters=FLAGS.num_iters,
                                             use_random_draw=FLAGS.use_random_draw)
        mean_performances.append(mean_perf)
        sd_performances.append(sd_perf)

    return mean_performances, sd_performances

def train_interaction_model(x_train, y_train, x_test, y_test):
    print('Training interaction model')
    model = get_default_model(x_train.shape[1])
    compile_model(model)
    tf.keras.models.save_model(model, 'models/{}_random.h5'.format(FLAGS.dataset))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5,
                                                mode='min')
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=FLAGS.epochs,
              verbose=0,
              validation_split=0.2,
              callbacks=[callback])
    tf.keras.models.save_model(model, 'models/{}.h5'.format(FLAGS.dataset))

def load_interaction_model():
    counter = 0
    while True:
        counter += 1
        try:
            model = tf.keras.models.load_model('models/{}.h5'.format(FLAGS.dataset))
            random_model = tf.keras.models.load_model('models/{}_random.h5'.format(FLAGS.dataset))
            random_weights = random_model.get_weights()

            print('Successfully restored trained model.')
            break
        except (FileNotFoundError, OSError):
            print('({}) Did not find saved model. Will try again in 60 seconds...'.format(counter))
        sleep(60)
    return model, random_weights

def main(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)
    x_train, y_train, x_test, y_test, spec_df = get_data(FLAGS.dataset)

    if FLAGS.train_interaction_model:
        train_interaction_model(x_train, y_train, x_test, y_test)

    model, random_weights = load_interaction_model()

    interaction_types = ['integrated_hessians',
                         'expected_hessians',
                         'hessians',
                         'hessians_times_inputs',
                         'shapley_sampling',
                         'contextual_decomposition',
                         'neural_interaction_detection']
    if FLAGS.interaction_type not in interaction_types:
        raise ValueError('Invalid interaction type `{}`'.format(FLAGS.interaction_type))

    print('Evaluating {}'.format(FLAGS.interaction_type))
    print('Getting interactions')
    interaction_function = return_interaction_function(FLAGS.interaction_type)

    interactions_train = interaction_function(model, x_train, baseline=x_test)
    interactions_test  = interaction_function(model, x_test, baseline=x_train)

    mean_performances, sd_performances = \
        get_performance_curve(x_train,
                              x_test,
                              model,
                              spec_df,
                              interactions_train,
                              interactions_test,
                              random_weights)

    num_removed = np.arange(len(mean_performances))
    type_list = [FLAGS.interaction_type] * len(mean_performances)

    data = pd.DataFrame({
        'interaction_type': type_list,
        'mean_perf': mean_performances,
        'sd_perf': sd_performances,
        'num_interactions_removed': num_removed
    })
    if FLAGS.use_random_draw:
        data.to_csv('results_random_draw/{}_{}.csv'.format(FLAGS.dataset, FLAGS.interaction_type))
    else:
        data.to_csv('results/{}_{}.csv'.format(FLAGS.dataset, FLAGS.interaction_type))

if __name__ == '__main__':
    tf.autograph.set_verbosity(0)
    app.run(main)
