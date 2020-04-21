import tensorflow as tf
import numpy as np
from tqdm import tqdm

from build_model import interaction_model

def compile_model(model, learning_rate=0.0001, regression=False):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if regression:
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanSquaredError()]
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.AUC()]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

def get_interactions(x_train, x_test,
                     model,
                     interaction_function,
                     random_seed=0):
    interactions = interaction_function(model, x_test, baseline=x_train)

def get_interaction_model(x_train, y_train, x_test, y_test, regression=True):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=1,  activation=None))

    if not regression:
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

    compile_model(model, regression=regression, learning_rate=0.001)
    model.fit(x_train, y_train, batch_size=300, epochs=200, verbose=0)
    model.evaluate(x_test, y_test, batch_size=300, verbose=2)
    return model

def get_performance(x_train, y_train, x_test, y_test,
                    best_param, regression=True,
                    interactions_to_ignore=None):
    test_performances = []
    for _ in range(10):
        model = interaction_model(num_features=x_train.shape[1],
                                  num_layers=best_param['num_layers'],
                                  hidden_layer_size=best_param['hidden_layer_size'],
                                  interactions_to_ignore=interactions_to_ignore,
                                  regression=regression)
        compile_model(model,
                      learning_rate=best_param['learning_rate'],
                      regression=regression)
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=128,
                  epochs=best_param['epochs'],
                  verbose=0)
        _, test_perf = model.evaluate(x=x_test,
                                      y=y_test,
                                      batch_size=128,
                                      verbose=0)
        test_performances.append(test_perf)
        del model

    mean_test_performance = np.mean(test_performances)
    sd_test_performance = np.std(test_performances)
    print('Finished training ({:.3f} +- {:.3f})'.format(mean_test_performance, sd_test_performance))

    return mean_test_performance, sd_test_performance
