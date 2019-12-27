import joblib
import tensorflow as tf
import numpy as np
from path_explain.utils import set_up_environment
from path_explain.path_explainer_tf import PathExplainerTF

from preprocess import higgs_dataset

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_layers', 4, 'Number of dense layers')
flags.DEFINE_integer('hidden_units', 300, 'Number of units in each dense layer')
flags.DEFINE_integer('batch_size', 1000, 'Batch size to use while training')
flags.DEFINE_integer('epochs', 200, 'Maximum number of epochs to train for')
flags.DEFINE_string('visible_devices', '0', 'Which gpu to use')
flags.DEFINE_float('dropout_rate', 0.25, 'Probability of dropping out a unit while training')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate to use for training')
flags.DEFINE_float('decay_rate', 0.98, 'Amount to decay the learning rate every epoch')
flags.DEFINE_float('momentum', 0.9, 'Momentum to use for SGD while training')

def build_model(dropout_rate=0.25,
                num_layers=4,
                hidden_units=300,
                for_interpretation=False):
    if for_interpretation:
        activation = tf.keras.activations.softplus
    else:
        activation = tf.keras.activations.relu

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28,),
                                    dtype=tf.float32,
                                    name='input'))

    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(units=hidden_units,
                                        activation=activation,
                                        name=f'dense_{i}'))
        model.add(tf.keras.layers.BatchNormalization(name=f'batchnorm_{i}'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate,
                                          name=f'dropout_{i}'))

    model.add(tf.keras.layers.Dense(units=1,
                                    activation=None,
                                    name='dense_out'))

    if not for_interpretation:
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid,
                                             name='sigmoid'))
    return model

def train(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)

    train_set, test_set, vald_set = higgs_dataset(batch_size=FLAGS.batch_size,
                                                  num_parallel_calls=8,
                                                  buffer_size=10000,
                                                  seed=0,
                                                  scale=True,
                                                  include_vald=True)

    model = build_model(dropout_rate=FLAGS.dropout_rate,
                        num_layers=FLAGS.num_layers,
                        hidden_units=FLAGS.hidden_units,
                        for_interpretation=False)

    steps_per_epoch = int(10000000 / FLAGS.batch_size)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=FLAGS.learning_rate,
                                                                   decay_steps=steps_per_epoch,
                                                                   decay_rate=FLAGS.decay_rate,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                        momentum=FLAGS.momentum,
                                        nesterov=True)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC()])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_set,
                        epochs=FLAGS.epochs,
                        validation_data=vald_set,
                        callbacks=[callback])
    tf.keras.models.save_model(model, 'model.h5')
    joblib.dump(history.history, 'history.pickle')

if __name__ == '__main__':
    app.run(train)
