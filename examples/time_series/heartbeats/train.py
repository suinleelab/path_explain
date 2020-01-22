"""
Trains a sequence classification model. Taken from
https://www.kaggle.com/coni57/model-from-arxiv-1805-00794
"""
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error
from sklearn.utils import shuffle
from scipy.signal import resample
from sklearn.preprocessing import OneHotEncoder
from path_explain.utils import set_up_environment

from model import cnn_model
from preprocess import mitbih_dataset

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate to use for training')
flags.DEFINE_float('beta_1', 0.9, 'Parameters for adam')
flags.DEFINE_float('beta_2', 0.999, 'Parameters for adam')
flags.DEFINE_float('decay_rate', 0.95, 'Exponential learning rate decay parameter')
flags.DEFINE_string('visible_devices', '0', 'Which gpu to train on')
flags.DEFINE_integer('batch_size', 500, 'Batch size for training')
flags.DEFINE_integer('epochs', 75, 'Number of epochs to train for')
flags.DEFINE_boolean('evaluate', False, 'Set to true to evaluate a saved model')

def exp_decay(epoch):
    initial_lrate = 0.001
    k = 0.75
    t = n_obs//(10000 * batch_size)
    lrate = initial_lrate * math.exp(-k*t)
    return lrate

def train(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)

    print('Reading data...')
    x_train, y_train, x_test, y_test = mitbih_dataset()
    print('Dataset shape: {}'.format(x_train.shape))

    if FLAGS.evaluate:
        model = tf.keras.models.load_model('model.h5')
        print('Evaluating on the training data...')
        model.evaluate(x_train, y_train, verbose=2)
        print('Evaluating on the test data...')
        model.evaluate(x_test, y_test, verbose=2)
        return

    print('Building model...')
    model = cnn_model()

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=FLAGS.learning_rate,
                                                                   decay_steps=int(x_train.shape[0] / FLAGS.batch_size),
                                                                   decay_rate=FLAGS.decay_rate,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=FLAGS.beta_1,
                                         beta_2=FLAGS.beta_2)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print('Training model...')
    model.fit(x_train, y_train,
              epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              verbose=1,
              validation_data=(x_test, y_test))

    tf.keras.models.save_model(model, 'model.h5')

if __name__ == '__main__':
    app.run(train)